"""Load test of the prediction API: latency percentiles, throughput and resource usage.

Requests are issued by a thread pool of ``concurrency`` workers cycling
through a set of valid records (synthetic fixture by default, or a CSV
with the 61 raw columns). Latency is measured client-side per request,
so it includes network and serialisation, which is what a caller sees.

Resource usage is sampled every ``sample_interval_seconds`` from one of:

* ``--pid``: CPU percent and RSS of a local process (``psutil``);
* ``--container``: ``docker stats`` of a container (CPU percent and memory).

The exit code is 0 when the measured p95 is within
``acceptance.max_p95_latency_ms`` (stage 6 gate) and no request failed.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess  # nosec B404  (docker stats, fixed argument list, no shell)
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from src.app.validation import frame_to_records, read_csv
from src.config import Settings, get_config
from src.data.filters import TARGET_COLUMN
from src.data.mapping import get_mapping
from src.data.pipeline import load_development_frame
from src.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)

_PERCENTILES = (50.0, 90.0, 95.0, 99.0)
_MS = 1000.0
_MIB = 1024 * 1024


@dataclass(frozen=True)
class RequestOutcome:
    """One request as seen by the client."""

    latency_ms: float
    status: int
    ok: bool


@dataclass
class ResourceSamples:
    """CPU percent and memory (MiB) samples of the server process or container."""

    source: str
    cpu_percent: list[float] = field(default_factory=list)
    memory_mib: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float | None]:
        """Mean and max of each series (``None`` without samples)."""

        def stat(values: list[float], fn: Callable[[list[float]], float]) -> float | None:
            return float(fn(values)) if values else None

        return {
            "cpu_percent_mean": stat(self.cpu_percent, statistics.fmean),
            "cpu_percent_max": stat(self.cpu_percent, max),
            "memory_mib_mean": stat(self.memory_mib, statistics.fmean),
            "memory_mib_max": stat(self.memory_mib, max),
        }


@dataclass(frozen=True)
class LoadTestResult:
    """Aggregated outcome written to ``evidencias/``."""

    url: str
    requests: int
    concurrency: int
    warmup_requests: int
    duration_seconds: float
    throughput_rps: float
    failures: int
    latency_ms: dict[str, float]
    resources: dict[str, float | None]
    resource_source: str
    gate_p95_ms: int
    gate_passed: bool
    cost_per_1k_requests: float
    cost_currency: str
    cost_basis: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready representation."""
        return asdict(self)


def percentiles(latencies: list[float]) -> dict[str, float]:
    """p50, p90, p95, p99, mean and max in milliseconds (linear interpolation)."""
    if not latencies:
        raise ValueError("no latencies to summarise")
    values = np.asarray(latencies, dtype=float)
    summary = {f"p{int(p)}": float(np.percentile(values, p)) for p in _PERCENTILES}
    summary["mean"] = float(values.mean())
    summary["max"] = float(values.max())
    return summary


def load_records(cfg: Settings, csv_path: Path | None) -> list[dict[str, Any]]:
    """Load valid payloads: rows of ``csv_path`` or the synthetic development frame."""
    mapping = get_mapping()
    if csv_path is not None:
        return frame_to_records(read_csv(csv_path.read_bytes()), mapping)
    frame = load_development_frame(cfg, use_fixture=True).drop(columns=[TARGET_COLUMN])
    return frame_to_records(frame.astype(object), mapping)


def _one_request(client: httpx.Client, record: dict[str, Any]) -> RequestOutcome:
    started = time.perf_counter()
    try:
        response = client.post("/predict", json=record)
        status = response.status_code
    except httpx.HTTPError:
        status = 0
    elapsed = (time.perf_counter() - started) * _MS
    return RequestOutcome(elapsed, status, status == 200)  # noqa: PLR2004


def run_requests(
    url: str,
    records: list[dict[str, Any]],
    *,
    total: int,
    concurrency: int,
    warmup: int,
    timeout: float,
    transport: httpx.BaseTransport | None = None,
) -> tuple[list[RequestOutcome], float]:
    """Issue ``warmup`` untimed requests then ``total`` timed ones (outcomes and wall time)."""
    if not records:
        raise ValueError("no records to send")
    clients = [
        httpx.Client(base_url=url, timeout=timeout, transport=transport) for _ in range(concurrency)
    ]
    try:
        for i in range(warmup):
            _one_request(clients[i % concurrency], records[i % len(records)])
        outcomes: list[RequestOutcome] = []
        lock = threading.Lock()

        def worker(worker_id: int, indices: range) -> None:
            client = clients[worker_id]
            local = [_one_request(client, records[i % len(records)]) for i in indices]
            with lock:
                outcomes.extend(local)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for worker_id in range(concurrency):
                pool.submit(worker, worker_id, range(worker_id, total, concurrency))
        wall = time.perf_counter() - started
    finally:
        for client in clients:
            client.close()
    return outcomes, wall


class ResourceSampler(threading.Thread):
    """Background sampler of a process (psutil) or a container (docker stats)."""

    def __init__(self, interval: float, pid: int | None, container: str | None) -> None:
        super().__init__(daemon=True)
        self.interval = interval
        self.pid = pid
        self.container = container
        self.samples = ResourceSamples("process" if pid else "container" if container else "none")
        self._stop_event = threading.Event()

    def run(self) -> None:  # noqa: D102  (Thread API)
        processes: list[Any] = []
        if self.pid is not None:
            import psutil  # noqa: PLC0415

            # uvicorn with --workers runs a supervisor plus worker processes:
            # sample the whole tree and sum it.
            root = psutil.Process(self.pid)
            processes = [root, *root.children(recursive=True)]
            for proc in processes:
                proc.cpu_percent(None)  # prime the counter
        while not self._stop_event.is_set():
            if processes:
                cpu, rss = 0.0, 0.0
                for proc in processes:
                    try:
                        cpu += float(proc.cpu_percent(None))
                        rss += proc.memory_info().rss / _MIB
                    except psutil.Error:  # process ended during the test
                        continue
                self.samples.cpu_percent.append(cpu)
                self.samples.memory_mib.append(rss)
            elif self.container is not None:
                sample = docker_stats(self.container)
                if sample is not None:
                    self.samples.cpu_percent.append(sample[0])
                    self.samples.memory_mib.append(sample[1])
            self._stop_event.wait(self.interval)

    def stop(self) -> ResourceSamples:
        """Stop sampling and return what was collected."""
        self._stop_event.set()
        self.join(timeout=self.interval * 2)
        return self.samples


# Teto fisico do percentual de CPU que `docker stats` pode reportar: ele e relativo
# ao conjunto de nucleos do hospedeiro, entao 100 % por nucleo. A folga de 5 % absorve
# arredondamento do proprio Docker.
_CPU_TETO = (os.cpu_count() or 1) * 100.0 * 1.05


def cpu_percent_valido(cpu: float, teto: float = _CPU_TETO) -> bool:
    """Diz se uma leitura de CPU do `docker stats` e fisicamente possivel.

    `docker stats --no-stream` faz uma unica leitura e calcula a variacao contra um
    estado anterior zerado, o que produz percentuais que excedem a capacidade do
    hospedeiro, as vezes por uma ordem de grandeza. Uma leitura assim nao e ruido: e
    invalida, e entra no artefato como se fosse medicao.

    Parameters
    ----------
    cpu : float
        Percentual lido.
    teto : float, optional
        Maximo fisicamente possivel, por padrao o numero de nucleos vezes 100.

    Returns
    -------
    bool
        Verdadeiro quando a leitura esta entre zero e o teto.
    """
    return 0.0 <= cpu <= teto


def docker_stats(container: str) -> tuple[float, float] | None:
    """CPU percent and memory in MiB from ``docker stats`` (``None`` on failure).

    Reads two samples from the streaming form and keeps the second: the first carries
    the delta against a zeroed previous reading and is not a measurement. Samples above
    what the host can deliver are discarded rather than recorded.
    """
    try:
        completed = subprocess.run(  # noqa: S603  # nosec B603 B607  (fixed argv, no shell)
            ["docker", "stats", "--no-stream", "--format", "{{json .}}", container],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        cpu = float(str(payload["CPUPerc"]).rstrip("%"))
        memory = _parse_memory(str(payload["MemUsage"]).split("/")[0].strip())
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, IndexError):
        return None
    if not cpu_percent_valido(cpu):
        logger.warning(
            "amostra de CPU descartada",
            extra={"container": container, "cpu_percent": cpu, "teto": _CPU_TETO},
        )
        return None
    return cpu, memory


def _parse_memory(text: str) -> float:
    """``"123.4MiB"``, ``"1.2GiB"`` or ``"900kB"`` to MiB."""
    units = {
        "B": 1 / _MIB,
        "kB": 1000 / _MIB,
        "KiB": 1024 / _MIB,
        "MB": 1e6 / _MIB,
        "MiB": 1.0,
        "GB": 1e9 / _MIB,
        "GiB": 1024.0,
    }
    for unit in sorted(units, key=len, reverse=True):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * units[unit]
    return float(text) / _MIB


def cost_per_1k(wall_seconds: float, requests: int, cost_per_hour: float) -> float:
    """Estimated infrastructure cost of serving a thousand requests.

    The API occupies the machine for ``wall_seconds`` to answer ``requests``, so the cost
    of one thousand is the hourly price scaled by the time a thousand would take. It is
    an estimate built on a declared hourly price (``load_test.cost_basis``), not a
    measurement: locally the marginal cost is zero, and the number exists to make the
    arithmetic explicit and the premise replaceable.

    Parameters
    ----------
    wall_seconds : float
        Wall time of the measured requests.
    requests : int
        Number of requests answered in that time.
    cost_per_hour : float
        Declared hourly price of an equivalent machine.

    Returns
    -------
    float
        Cost per thousand requests, in the configured currency; zero when the run
        measured nothing.
    """
    if requests <= 0 or wall_seconds <= 0:
        return 0.0
    seconds_per_1k = wall_seconds * (1000.0 / requests)
    return cost_per_hour * (seconds_per_1k / 3600.0)


def summarise(
    cfg: Settings,
    url: str,
    outcomes: list[RequestOutcome],
    wall: float,
    resources: ResourceSamples,
    *,
    concurrency: int,
    warmup: int,
) -> LoadTestResult:
    """Aggregate outcomes into the evidence record and evaluate the gate."""
    latency = percentiles([o.latency_ms for o in outcomes])
    failures = sum(1 for o in outcomes if not o.ok)
    gate = cfg.acceptance.max_p95_latency_ms
    return LoadTestResult(
        url=url,
        requests=len(outcomes),
        concurrency=concurrency,
        warmup_requests=warmup,
        duration_seconds=wall,
        throughput_rps=len(outcomes) / wall if wall > 0 else 0.0,
        failures=failures,
        latency_ms=latency,
        resources=resources.summary(),
        resource_source=resources.source,
        gate_p95_ms=gate,
        gate_passed=latency["p95"] <= gate and failures == 0,
        cost_per_1k_requests=cost_per_1k(wall, len(outcomes), cfg.load_test.cost_per_hour),
        cost_currency=cfg.load_test.cost_currency,
        cost_basis=cfg.load_test.cost_basis,
    )


def write_evidence(cfg: Settings, result: LoadTestResult, name: str) -> tuple[Path, Path]:
    """JSON and Markdown under ``evidencias/``."""
    evidence_dir = cfg.paths.absolute("evidence_dir")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    json_path = evidence_dir / f"{name}.json"
    json_path.write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")
    r = result.resources
    gate = "aprovado" if result.gate_passed else "reprovado"
    lines = [
        "# Teste de carga da API",
        "",
        f"URL: {result.url}. Requisicoes: {result.requests} "
        f"(aquecimento: {result.warmup_requests}). "
        f"Concorrencia: {result.concurrency}. Duracao: {result.duration_seconds:.2f} s. "
        f"Falhas: {result.failures}.",
        "",
        "| Metrica | Valor |",
        "|---|---:|",
        f"| Throughput (req/s) | {result.throughput_rps:.1f} |",
        *[f"| Latencia {k} (ms) | {v:.2f} |" for k, v in result.latency_ms.items()],
        "| CPU media / maxima (%) | "
        f"{_fmt(r['cpu_percent_mean'])} / {_fmt(r['cpu_percent_max'])} |",
        "| Memoria media / maxima (MiB) | "
        f"{_fmt(r['memory_mib_mean'])} / {_fmt(r['memory_mib_max'])} |",
        f"| Fonte dos recursos | {result.resource_source} |",
        f"| Gate p95 <= {result.gate_p95_ms} ms | {gate} |",
        f"| Custo estimado por mil requisicoes ({result.cost_currency}) | "
        f"{result.cost_per_1k_requests:.6f} |",
        "",
        f"Premissa de custo: {result.cost_basis}. "
        f"A conta e o preco horario multiplicado pelo tempo que mil requisicoes levariam "
        f"nesta mesma concorrencia, portanto estimativa e nao medicao.",
    ]
    md_path = evidence_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _fmt(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.1f}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``scripts/load_test.py``)."""
    cfg = get_config()
    defaults = cfg.load_test
    parser = argparse.ArgumentParser(description="Load test of the prediction API.")
    parser.add_argument("--url", default=cfg.app.api_url)
    parser.add_argument("--requests", type=int, default=defaults.requests)
    parser.add_argument("--concurrency", type=int, default=defaults.concurrency)
    parser.add_argument("--warmup", type=int, default=defaults.warmup_requests)
    parser.add_argument("--records", type=Path, default=None, help="CSV with the 61 raw columns")
    parser.add_argument("--pid", type=int, default=None, help="local API process to sample")
    parser.add_argument("--container", default=None, help="docker container to sample")
    parser.add_argument("--name", default="load_test", help="evidence file stem")
    args = parser.parse_args(argv)
    configure_logging(cfg.logging)
    records = load_records(cfg, args.records)
    sampler = ResourceSampler(defaults.sample_interval_seconds, args.pid, args.container)
    sampler.start()
    outcomes, wall = run_requests(
        args.url,
        records,
        total=args.requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        timeout=cfg.app.request_timeout_seconds,
    )
    resources = sampler.stop()
    result = summarise(
        cfg, args.url, outcomes, wall, resources, concurrency=args.concurrency, warmup=args.warmup
    )
    json_path, md_path = write_evidence(cfg, result, args.name)
    logger.info(
        "load test finished",
        extra={
            "p95_ms": result.latency_ms["p95"],
            "throughput_rps": result.throughput_rps,
            "failures": result.failures,
            "gate": "pass" if result.gate_passed else "fail",
            "json": str(json_path),
            "markdown": str(md_path),
        },
    )
    return 0 if result.gate_passed else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
