"""Load test helpers with a mocked transport (no server needed)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.config import Settings
from src.monitoring.load_test import (
    RequestOutcome,
    ResourceSamples,
    _parse_memory,
    cost_per_1k,
    cpu_percent_valido,
    load_records,
    percentiles,
    run_requests,
    summarise,
    write_evidence,
)


def test_percentiles_known_values() -> None:
    values = [float(v) for v in range(1, 101)]
    summary = percentiles(values)
    assert summary["p50"] == pytest.approx(50.5) and summary["p95"] == pytest.approx(95.05)
    assert summary["max"] == 100.0 and summary["mean"] == pytest.approx(50.5)
    with pytest.raises(ValueError):
        percentiles([])


def test_run_requests_counts_and_warmup(cfg: Settings) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        body = json.loads(request.content)
        status = 422 if body.get("age") == 999 else 200
        return httpx.Response(status, json={"probability": 0.1})

    records = load_records(cfg, None)
    assert len(records) > 10 and set(records[0]) >= {"age", "sex"}
    records[0] = dict(records[0], age=999)
    outcomes, wall = run_requests(
        "http://api",
        records,
        total=20,
        concurrency=4,
        warmup=3,
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    assert len(outcomes) == 20 and len(calls) == 23 and wall > 0
    failures = [o for o in outcomes if not o.ok]
    assert failures and all(o.status == 422 for o in failures)
    result = summarise(
        cfg, "http://api", outcomes, wall, ResourceSamples("none"), concurrency=4, warmup=3
    )
    assert result.failures == len(failures) and not result.gate_passed
    assert result.throughput_rps == pytest.approx(20 / wall)
    with pytest.raises(ValueError):
        run_requests("http://api", [], total=1, concurrency=1, warmup=0, timeout=1)


def test_summary_gate_and_evidence(cfg: Settings, tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    records = load_records(cfg, None)[:5]
    outcomes, wall = run_requests(
        "http://api",
        records,
        total=10,
        concurrency=2,
        warmup=0,
        timeout=5,
        transport=httpx.MockTransport(handler),
    )
    samples = ResourceSamples("process", cpu_percent=[10.0, 30.0], memory_mib=[100.0, 120.0])
    assert samples.summary() == {
        "cpu_percent_mean": 20.0,
        "cpu_percent_max": 30.0,
        "memory_mib_mean": 110.0,
        "memory_mib_max": 120.0,
    }
    assert ResourceSamples("none").summary()["cpu_percent_mean"] is None
    result = summarise(cfg, "http://api", outcomes, wall, samples, concurrency=2, warmup=0)
    assert result.gate_passed and result.gate_p95_ms == cfg.acceptance.max_p95_latency_ms
    local_cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"evidence_dir": tmp_path})}
    )
    json_path, md_path = write_evidence(local_cfg, result, "load_test_unit")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["requests"] == 10 and payload["resources"]["cpu_percent_max"] == 30.0
    assert "| Throughput (req/s) |" in md_path.read_text(encoding="utf-8")


def test_parse_memory_units() -> None:
    assert _parse_memory("100MiB") == 100.0
    assert _parse_memory("1GiB") == 1024.0
    assert _parse_memory("1000MB") == pytest.approx(1000 * 1e6 / (1024 * 1024))
    assert _parse_memory("1024kB") == pytest.approx(1024 * 1000 / (1024 * 1024))
    assert _parse_memory("1048576") == 1.0


def test_main_with_stubbed_requests(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.monitoring import load_test  # noqa: PLC0415

    local_cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"evidence_dir": tmp_path})}
    )
    monkeypatch.setattr(load_test, "get_config", lambda: local_cfg)

    def fake_run_requests(url: str, records: list[dict[str, object]], **kwargs: object):  # type: ignore[no-untyped-def]
        total = int(kwargs["total"])  # type: ignore[call-overload]
        return [load_test.RequestOutcome(5.0, 200, True) for _ in range(total)], 0.5

    monkeypatch.setattr(load_test, "run_requests", fake_run_requests)
    code = load_test.main(["--url", "http://stub", "--requests", "20", "--name", "lt_unit"])
    assert code == 0
    payload = json.loads((tmp_path / "lt_unit.json").read_text(encoding="utf-8"))
    assert (
        payload["requests"] == 20
        and payload["gate_passed"]
        and payload["resource_source"] == "none"
    )


# ------------------------------------------------------------------------ cost
def test_cost_per_1k_scales_the_hourly_price_by_time() -> None:
    # 100 requests in 36 s means 360 s for a thousand, a tenth of an hour.
    assert cost_per_1k(36.0, 100, 1.0) == pytest.approx(0.1)
    assert cost_per_1k(36.0, 100, 0.5) == pytest.approx(0.05)


def test_cost_per_1k_is_zero_without_a_measurement() -> None:
    assert cost_per_1k(0.0, 100, 1.0) == 0.0
    assert cost_per_1k(10.0, 0, 1.0) == 0.0


def test_faster_service_costs_less_for_the_same_work() -> None:
    fast = cost_per_1k(10.0, 1000, 0.5)
    slow = cost_per_1k(20.0, 1000, 0.5)
    assert fast < slow


def test_result_carries_the_cost_and_its_premise(cfg: Settings) -> None:
    outcomes = [RequestOutcome(10.0, 200, True) for _ in range(50)]
    result = summarise(
        cfg,
        "http://x",
        outcomes,
        wall=5.0,
        resources=ResourceSamples("none"),
        concurrency=2,
        warmup=0,
    )
    assert result.cost_per_1k_requests == pytest.approx(
        cost_per_1k(5.0, 50, cfg.load_test.cost_per_hour)
    )
    assert result.cost_currency == cfg.load_test.cost_currency
    assert result.cost_basis


def test_evidence_shows_the_cost_and_says_it_is_an_estimate(cfg: Settings, tmp_path: Path) -> None:
    outcomes = [RequestOutcome(10.0, 200, True) for _ in range(10)]
    result = summarise(
        cfg,
        "http://x",
        outcomes,
        wall=1.0,
        resources=ResourceSamples("none"),
        concurrency=1,
        warmup=0,
    )
    local_cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"evidence_dir": tmp_path})}
    )
    _, md_path = write_evidence(local_cfg, result, "load_test_cost_unit")
    text = md_path.read_text(encoding="utf-8")
    assert "Custo estimado por mil requisicoes" in text
    assert "estimativa e nao medicao" in text


# ----------------------------------------------------------- amostragem de CPU
def test_cpu_percent_valido_aceita_ate_o_numero_de_nucleos() -> None:
    # `docker stats` reporta percentual relativo ao conjunto de nucleos do
    # hospedeiro: 100 % por nucleo e o maximo fisicamente possivel.
    assert cpu_percent_valido(94.2, teto=210.0)
    assert cpu_percent_valido(200.0, teto=210.0)
    assert cpu_percent_valido(0.0, teto=210.0)


def test_cpu_percent_valido_descarta_leitura_impossivel() -> None:
    # Leitura observada na pratica com `--no-stream` em Docker Desktop: a variacao
    # e calculada contra um estado zerado e produz percentuais irreais.
    assert not cpu_percent_valido(2250.6, teto=210.0)
    assert not cpu_percent_valido(-1.0, teto=210.0)
