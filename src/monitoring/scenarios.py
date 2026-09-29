"""Reproducible drift scenarios: baseline (no drift) and simulated drift.

Both scenarios use the training split as reference and the 20 % holdout
as the "current" window. The simulated scenario applies the deterministic
shifts declared in ``monitoring.scenarios.simulated`` (seeded row
selection for categorical overrides). Two execution modes:

* offline (default): PSI and KS computed directly with
  :mod:`src.monitoring.drift`;
* online (``--api-url``): the window is sent to ``/predict/batch`` and the
  statistics are read back from the ``/metrics`` gauges, exercising the
  same path Prometheus and the Streamlit panel use.

Outputs go to ``evidencias/drift_<scenario>.json`` and ``.md``. The exit
code is 0 when the observed alerts match ``expect_alerts`` (the stage 6
gate: drift detected in the simulated scenario, absent in the baseline).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ScenarioConfig, Settings, get_config
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import VariableMapping, get_mapping
from src.data.pipeline import build_development_frame, load_development_frame
from src.data.synthetic import make_synthetic_frame
from src.logging_setup import configure_logging, get_logger
from src.monitoring.drift import (
    DriftError,
    DriftReport,
    ReferenceWindow,
    build_reference,
    compute_drift,
    load_reference,
    report_markdown,
)
from src.monitoring.prometheus_text import parse_exposition, select
from src.paths import resolve

logger = get_logger(__name__)


def apply_scenario(frame: pd.DataFrame, scenario: ScenarioConfig, seed: int) -> pd.DataFrame:
    """Return a shifted copy of ``frame`` (pure: the input is not modified).

    Parameters
    ----------
    frame : pandas.DataFrame
        Current window with English raw columns.
    scenario : ScenarioConfig
        Numeric shifts (``x * multiply + add``) and categorical overrides.
    seed : int
        Seed of the row selection for categorical overrides.
    """
    shifted = frame.copy()
    for name, shift in scenario.numeric_shifts.items():
        if name not in shifted.columns:
            raise DriftError(f"scenario shifts unknown variable '{name}'")
        shifted[name] = shifted[name].astype(float) * shift.multiply + shift.add
    rng = np.random.default_rng(seed)
    for name, override in scenario.categorical_overrides.items():
        if name not in shifted.columns:
            raise DriftError(f"scenario overrides unknown variable '{name}'")
        n_rows = round(override.share * len(shifted))
        chosen = rng.choice(len(shifted), size=n_rows, replace=False)
        column = shifted.columns.get_loc(name)
        shifted.iloc[chosen, column] = float(override.code)
    return shifted


@dataclass(frozen=True)
class ScenarioWindows:
    """Reference and current windows of one scenario."""

    reference: ReferenceWindow
    current: pd.DataFrame
    reference_origin: str


def prepare_windows(
    cfg: Settings, scenario_name: str, *, use_fixture: bool, materialize: bool
) -> ScenarioWindows:
    """Load the persisted reference (or rebuild it) and the shifted holdout.

    With ``use_fixture`` the synthetic generator is run with
    ``monitoring.scenario_fixture_rows`` rows: the default fixture is too
    small for PSI (a sample-size-sensitive statistic) to stay below the
    thresholds on identical distributions.
    """
    scenario = cfg.monitoring.scenarios[scenario_name]
    if use_fixture:
        raw = make_synthetic_frame(seed=cfg.project.seed, rows=cfg.monitoring.scenario_fixture_rows)
        development, _ = build_development_frame(raw, cfg, enforce_counts=False)
    else:
        development = load_development_frame(cfg, materialize=materialize)
    train, test = split_holdout(development, cfg.split, cfg.project.seed)
    x_train = train.drop(columns=[TARGET_COLUMN])
    x_test = test.drop(columns=[TARGET_COLUMN]).reset_index(drop=True)
    reference_path = resolve(cfg.api.local_run_dir) / cfg.monitoring.reference_file
    if reference_path.is_file() and not use_fixture:
        reference, origin = load_reference(reference_path), str(reference_path)
    else:
        reference, origin = build_reference(x_train, get_mapping(), cfg.monitoring.drift), "rebuilt"
    return ScenarioWindows(reference, apply_scenario(x_test, scenario, cfg.project.seed), origin)


def run_offline(cfg: Settings, windows: ScenarioWindows) -> DriftReport:
    """PSI and KS computed in-process."""
    return compute_drift(windows.reference, windows.current, cfg.monitoring.drift)


def run_online(cfg: Settings, windows: ScenarioWindows, api_url: str) -> dict[str, Any]:
    """Send the window to the API and read the drift gauges back from ``/metrics``."""
    import httpx  # noqa: PLC0415  (client dependency, not needed offline)

    mapping = get_mapping()
    janela, recortes = clip_to_domain(windows.current, mapping)
    if recortes:
        logger.warning(
            "valores recortados para o dominio declarado antes do envio a API",
            extra={"clipped": recortes, "rows": len(janela)},
        )
    records = _api_records(janela, mapping)
    chunk = cfg.app.batch_chunk_size
    with httpx.Client(base_url=api_url, timeout=cfg.app.request_timeout_seconds) as client:
        for start in range(0, len(records), chunk):
            response = client.post(
                "/predict/batch", json={"records": records[start : start + chunk]}
            )
            response.raise_for_status()
        samples = parse_exposition(client.get("/metrics").raise_for_status().text)
    psi = {s.labels["variable"]: s.value for s in select(samples, "data_drift_psi")}
    ks = {s.labels["variable"]: s.value for s in select(samples, "data_drift_ks_statistic")}
    alerts = sorted(
        s.labels["variable"] for s in select(samples, "data_drift_alert") if s.value >= 1
    )
    return {
        "psi": psi,
        "ks_statistic": ks,
        "alerts": alerts,
        "rows_sent": len(records),
        "clipped_to_domain": recortes,
    }


def clip_to_domain(
    frame: pd.DataFrame, mapping: VariableMapping
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip numeric columns to the domain declared in the mapping.

    The API validates every field against ``domain.min`` and ``domain.max`` (see
    ``src/data/contract.py``), and rejects the whole batch when one record falls
    outside. The simulated scenario shifts ``age`` by fifteen years, which pushes the
    oldest respondents past the declared maximum of 104, so the online injection would
    always fail with 422.

    Clipping is the honest reconciliation: the window sent online is the shifted window
    restricted to what the contract admits, and the number of values clipped is reported
    so the difference from the offline window is visible rather than silent.

    Parameters
    ----------
    frame : pandas.DataFrame
        Current window, already shifted by the scenario.
    mapping : VariableMapping
        Source of the declared domains.

    Returns
    -------
    clipped : pandas.DataFrame
        Copy with numeric columns inside their declared range.
    counts : dict of str to int
        Values clipped per column, omitting columns with none.
    """
    clipped = frame.copy()
    counts: dict[str, int] = {}
    for variable in mapping.raw_variables:
        coluna = variable.en
        domain = variable.domain
        if coluna not in clipped.columns or domain.type == "categorical":
            continue
        minimo, maximo = domain.min, domain.max
        if minimo is None and maximo is None:
            continue
        valores = pd.to_numeric(clipped[coluna], errors="coerce")
        fora = pd.Series(False, index=clipped.index)
        if minimo is not None:
            fora |= valores < minimo
        if maximo is not None:
            fora |= valores > maximo
        total = int(fora.fillna(False).sum())
        if total:
            counts[coluna] = total
            clipped[coluna] = valores.clip(lower=minimo, upper=maximo)
    return clipped, counts


def _api_records(frame: pd.DataFrame, mapping: VariableMapping) -> list[dict[str, Any]]:
    """Rows as JSON-ready dictionaries, typed as the API contract declares them.

    Casting every value to ``float`` makes the API reject the whole batch: the
    categorical fields are declared as ``Literal[1, 2, ...]`` in
    ``src/data/contract.py``, and Pydantic 2.9.2, the version pinned in the locks,
    refuses ``1.0`` where ``1`` is required. Newer Pydantic coerces it silently, so the
    defect is invisible outside the locked environment and must not be "simplified"
    back to a blanket ``float``. The cast follows the mapping instead: categorical and
    integer variables go as ``int``, the remaining ones as ``float``, and a missing
    value as ``null``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Window to send.
    mapping : VariableMapping
        Source of the declared type of each raw variable.

    Returns
    -------
    list of dict
        One dictionary per row, ready to serialise as JSON.
    """
    inteiros = {
        variable.en
        for variable in mapping.raw_variables
        if variable.domain.type == "categorical" or variable.dtype == "int"
    }

    def converter(coluna: str, valor: Any) -> Any:
        if pd.isna(valor):
            return None
        return int(round(float(valor))) if coluna in inteiros else float(valor)

    return [
        {k: converter(k, v) for k, v in row.items()}
        for row in frame.to_dict("records")
    ]


def write_evidence(
    cfg: Settings,
    name: str,
    report: DriftReport,
    online: dict[str, Any] | None,
    origin: str,
    *,
    suffix: str = "",
) -> tuple[Path, Path]:
    """Persist JSON and Markdown under ``evidencias/`` (``suffix`` keeps fixture runs apart)."""
    evidence_dir = cfg.paths.absolute("evidence_dir")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    scenario = cfg.monitoring.scenarios[name]
    payload = {
        "scenario": name,
        "description": scenario.description,
        "expect_alerts": scenario.expect_alerts,
        "reference_origin": origin,
        "thresholds": cfg.monitoring.drift.model_dump(),
        "offline": report.to_dict(),
        "online": online,
    }
    json_path = evidence_dir / f"drift_{name}{suffix}.json"
    json_path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    md_path = evidence_dir / f"drift_{name}{suffix}.md"
    title = f"Cenario de drift: {name} ({scenario.description})"
    md_path.write_text(report_markdown(report, title), encoding="utf-8")
    return json_path, md_path


def main(argv: list[str] | None = None, *, default_scenario: str | None = None) -> int:
    """CLI shared by ``scripts/baseline_scenario.py`` and ``scripts/simulate_drift.py``."""
    parser = argparse.ArgumentParser(description="Run a reproducible drift scenario.")
    parser.add_argument("--scenario", default=default_scenario, required=default_scenario is None)
    parser.add_argument("--fixture", action="store_true", help="use the synthetic fixture")
    parser.add_argument("--no-materialize", action="store_true")
    parser.add_argument("--api-url", default=None, help="also run online against this API")
    args = parser.parse_args(argv)
    cfg = get_config()
    configure_logging(cfg.logging)
    if args.scenario not in cfg.monitoring.scenarios:
        parser.error(f"unknown scenario '{args.scenario}'")
    windows = prepare_windows(
        cfg, args.scenario, use_fixture=args.fixture, materialize=not args.no_materialize
    )
    report = run_offline(cfg, windows)
    online = run_online(cfg, windows, args.api_url) if args.api_url else None
    json_path, md_path = write_evidence(
        cfg,
        args.scenario,
        report,
        online,
        windows.reference_origin,
        suffix="_fixture" if args.fixture else "",  # never overwrite the real-data evidence
    )
    scenario = cfg.monitoring.scenarios[args.scenario]
    detected = bool(report.alerts)
    expected_variables = set(scenario.numeric_shifts) | set(scenario.categorical_overrides)
    matches = detected == scenario.expect_alerts and expected_variables <= set(report.alerts)
    if online is not None:
        matches = matches and (bool(online["alerts"]) == scenario.expect_alerts)
    logger.info(
        "scenario finished",
        extra={
            "scenario": args.scenario,
            "alerts": report.alerts,
            "warnings": report.warnings,
            "online_alerts": None if online is None else online["alerts"],
            "gate": "pass" if matches else "fail",
            "json": str(json_path),
            "markdown": str(md_path),
        },
    )
    return 0 if matches else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
