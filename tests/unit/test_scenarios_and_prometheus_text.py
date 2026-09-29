"""Drift scenarios (shifts, windows, evidence) and the Prometheus text parser."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CategoricalOverride, NumericShift, ScenarioConfig, Settings
from src.monitoring.drift import DriftError
from src.monitoring.prometheus_text import (
    histogram_quantile,
    parse_exposition,
    select,
    total,
)
from src.data.mapping import get_mapping
from src.monitoring.scenarios import (
    _api_records,
    apply_scenario,
    clip_to_domain,
    prepare_windows,
    run_offline,
    write_evidence,
)

EXPOSITION = """
# HELP api_requests_total Requests by endpoint, method and HTTP status
# TYPE api_requests_total counter
api_requests_total{endpoint="/predict",method="POST",status="200"} 90.0
api_requests_total{endpoint="/predict",method="POST",status="422"} 10.0
api_requests_total{endpoint="/health",method="GET",status="200"} 5.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.05",method="POST"} 50.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.1",method="POST"} 90.0
api_request_latency_seconds_bucket{endpoint="/predict",le="0.2",method="POST"} 100.0
api_request_latency_seconds_bucket{endpoint="/predict",le="+Inf",method="POST"} 100.0
api_request_latency_seconds_count{endpoint="/predict",method="POST"} 100.0
data_drift_psi{variable="age"} 0.0123
data_drift_alert{variable="age"} 0.0
data_drift_reference_loaded 1.0
weird_line_without_value
"""


def test_parse_exposition_and_helpers() -> None:
    samples = parse_exposition(EXPOSITION)
    assert len(samples) == 11
    assert total(samples, "api_requests_total") == 105.0
    assert total(samples, "api_requests_total", status="422") == 10.0
    assert total(samples, "missing_metric") == 0.0
    assert select(samples, "data_drift_psi")[0].labels == {"variable": "age"}
    assert select(samples, "data_drift_reference_loaded")[0].value == 1.0
    assert math.isinf(parse_exposition('x{le="+Inf"} 1')[0].value) is False  # value, not label


def test_histogram_quantile_interpolates_like_prometheus() -> None:
    samples = parse_exposition(EXPOSITION)
    # p50: rank 50 sits exactly at the 0.05 bucket bound
    assert histogram_quantile(samples, "api_request_latency_seconds", 0.5) == pytest.approx(0.05)
    # p95: rank 95 inside (0.1, 0.2] with 10 observations -> 0.1 + 0.1 * 5/10
    assert histogram_quantile(samples, "api_request_latency_seconds", 0.95) == pytest.approx(0.15)
    assert histogram_quantile(samples, "api_request_latency_seconds", 0.999) == pytest.approx(0.199)
    assert histogram_quantile(samples, "no_histogram", 0.95) is None
    with pytest.raises(ValueError):
        histogram_quantile(samples, "api_request_latency_seconds", 1.0)
    empty = parse_exposition('h_bucket{le="+Inf"} 0\n')
    assert histogram_quantile(empty, "h", 0.5) is None
    only_inf = parse_exposition('h_bucket{le="0.1"} 0\nh_bucket{le="+Inf"} 3\n')
    assert histogram_quantile(only_inf, "h", 0.5) == pytest.approx(0.1)


def test_apply_scenario_is_pure_and_deterministic() -> None:
    frame = pd.DataFrame({"age": [20.0, 30.0, 40.0, 50.0], "region": [1.0, 2.0, 4.0, 5.0]})
    scenario = ScenarioConfig(
        description="t",
        expect_alerts=True,
        numeric_shifts={"age": NumericShift(add=5, multiply=2)},
        categorical_overrides={"region": CategoricalOverride(code=3, share=0.5)},
    )
    first = apply_scenario(frame, scenario, seed=1)
    second = apply_scenario(frame, scenario, seed=1)
    pd.testing.assert_frame_equal(first, second)
    assert frame["age"].tolist() == [20.0, 30.0, 40.0, 50.0]  # untouched input
    assert first["age"].tolist() == [45.0, 65.0, 85.0, 105.0]
    assert (first["region"] == 3.0).sum() == 2
    with pytest.raises(DriftError, match="unknown variable"):
        apply_scenario(
            frame, scenario.model_copy(update={"numeric_shifts": {"x": NumericShift()}}), 1
        )
    with pytest.raises(DriftError, match="unknown variable"):
        apply_scenario(
            frame,
            scenario.model_copy(
                update={"categorical_overrides": {"x": CategoricalOverride(code=1, share=1)}}
            ),
            1,
        )


def test_prepare_windows_and_evidence_on_fixture(cfg: Settings, tmp_path: Path) -> None:
    local_cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"evidence_dir": tmp_path})}
    )
    baseline = prepare_windows(local_cfg, "baseline", use_fixture=True, materialize=False)
    assert baseline.reference_origin == "rebuilt"
    simulated = prepare_windows(local_cfg, "simulated", use_fixture=True, materialize=False)
    assert len(simulated.current) == len(baseline.current)
    shift = cfg.monitoring.scenarios["simulated"].numeric_shifts["age"]
    np.testing.assert_allclose(
        simulated.current["age"], baseline.current["age"] * shift.multiply + shift.add
    )
    report = run_offline(local_cfg, simulated)
    assert {"age", "glucose", "weight_kg", "region"} <= set(report.alerts)
    json_path, md_path = write_evidence(local_cfg, "simulated", report, None, "rebuilt")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["scenario"] == "simulated" and payload["online"] is None
    assert payload["thresholds"]["psi_alert"] == cfg.monitoring.drift.psi_alert
    assert md_path.read_text(encoding="utf-8").startswith("# Cenario de drift: simulated")


def test_main_on_fixture_writes_suffixed_evidence_and_gates(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.monitoring import scenarios  # noqa: PLC0415

    local_cfg = cfg.model_copy(
        update={"paths": cfg.paths.model_copy(update={"evidence_dir": tmp_path})}
    )
    monkeypatch.setattr(scenarios, "get_config", lambda: local_cfg)
    assert scenarios.main(["--fixture"], default_scenario="baseline") == 0
    assert scenarios.main(["--fixture", "--scenario", "simulated"]) == 0
    assert (tmp_path / "drift_baseline_fixture.json").exists()
    assert (tmp_path / "drift_simulated_fixture.md").exists()
    assert not (tmp_path / "drift_baseline.json").exists()
    with pytest.raises(SystemExit):
        scenarios.main(["--fixture", "--scenario", "unknown"])


# ------------------------------------------------- recorte ao dominio declarado
def test_clip_to_domain_respeita_o_contrato_da_api() -> None:
    # O cenario simulado soma 15 anos a idade; o dominio declarado termina em 104 e a
    # API rejeita o lote inteiro quando um unico registro sai da faixa (422).
    quadro = pd.DataFrame({"age": [33.0, 104.0, 119.0], "weight_kg": [60.0, 70.0, 80.0]})
    recortado, contagem = clip_to_domain(quadro, get_mapping())
    assert contagem == {"age": 1}
    assert recortado["age"].tolist() == [33.0, 104.0, 104.0]
    assert recortado["weight_kg"].tolist() == [60.0, 70.0, 80.0]


def test_clip_to_domain_nao_toca_em_variavel_categorica() -> None:
    quadro = pd.DataFrame({"region": [1, 3, 5]})
    recortado, contagem = clip_to_domain(quadro, get_mapping())
    assert contagem == {}
    assert recortado["region"].tolist() == [1, 3, 5]


def test_clip_to_domain_nao_relata_recorte_quando_tudo_esta_na_faixa() -> None:
    quadro = pd.DataFrame({"age": [40.0, 50.0], "glucose": [90.0, 120.0]})
    _, contagem = clip_to_domain(quadro, get_mapping())
    assert contagem == {}


# --------------------------------------------- tipos exigidos pelo contrato da API
def test_api_records_envia_categorica_como_inteiro() -> None:
    # Os campos categoricos sao Literal[1, 2, ...]; o Pydantic travado (2.9.2) rejeita
    # 1.0 onde exige 1. Versoes mais novas coagem em silencio, entao a assercao e de
    # tipo, e nao de aceitacao pelo modelo, para valer nas duas.
    mapping = get_mapping()
    quadro = pd.DataFrame({"sex": [2.0], "region": [3.0], "age": [55.0],
                           "weight_kg": [80.5], "glucose": [110.25]})
    registro = _api_records(quadro, mapping)[0]
    assert isinstance(registro["sex"], int)
    assert isinstance(registro["region"], int)
    assert isinstance(registro["age"], int)
    assert isinstance(registro["weight_kg"], float)
    assert isinstance(registro["glucose"], float)
    assert registro == {"sex": 2, "region": 3, "age": 55,
                        "weight_kg": 80.5, "glucose": 110.25}


def test_api_records_preserva_ausente_como_nulo() -> None:
    mapping = get_mapping()
    quadro = pd.DataFrame({"glucose": [np.nan], "sex": [1.0]})
    registro = _api_records(quadro, mapping)[0]
    assert registro["glucose"] is None
    assert registro["sex"] == 1
