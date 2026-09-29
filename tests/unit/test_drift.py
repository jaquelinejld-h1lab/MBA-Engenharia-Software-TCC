"""PSI and KS statistics, reference persistence and the observation window."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import DriftConfig, MonitoringConfig, RetrainingConfig, Settings
from src.data.filters import TARGET_COLUMN
from src.data.mapping import VariableMapping
from src.monitoring.drift import (
    MISSING_BIN,
    DriftError,
    ReferenceWindow,
    build_reference,
    compute_drift,
    ks_test,
    load_reference,
    psi,
    psi_level,
    report_markdown,
    save_reference,
)
from src.monitoring.window import DriftMonitor, ObservationWindow

RETRAINING = RetrainingConfig(min_variables_in_alert=3, flag_file="retraining_flag.json")
DRIFT = DriftConfig(
    psi_warning=0.10,
    psi_alert=0.25,
    ks_statistic_alert=0.10,
    ks_pvalue_alert=0.05,
    psi_bins=10,
    epsilon=1e-4,
    ks_min_rows=30,
)


# ------------------------------------------------------------------ statistics
def test_psi_is_zero_for_identical_shares_and_symmetric() -> None:
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5], 1e-4) == 0.0
    a, b = [0.1, 0.9], [0.3, 0.7]
    assert psi(a, b, 1e-4) == pytest.approx(psi(b, a, 1e-4))


def test_psi_known_value_and_epsilon_for_empty_bins() -> None:
    # hand computation: (0.3-0.1)*ln(3) + (0.7-0.9)*ln(7/9)
    expected = 0.2 * math.log(3) + (-0.2) * math.log(7 / 9)
    assert psi([0.1, 0.9], [0.3, 0.7], 1e-4) == pytest.approx(expected)
    # a bin that vanishes is smoothed, not infinite
    value = psi([0.5, 0.5], [1.0, 0.0], 1e-4)
    assert math.isfinite(value) and value > 0.25


def test_psi_rejects_length_mismatch_and_handles_all_zero() -> None:
    with pytest.raises(DriftError):
        psi([0.5, 0.5], [1.0], 1e-4)
    assert psi([0.0, 0.0], [0.0, 0.0], 1e-4) == 0.0


def test_psi_level_uses_config_thresholds() -> None:
    assert psi_level(0.05, DRIFT) == "ok"
    assert psi_level(0.10, DRIFT) == "warning"
    assert psi_level(0.25, DRIFT) == "alert"


def test_ks_detects_shift_and_respects_minimum_rows() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500)
    stat, p = ks_test(a, a + 1.0, 30)
    assert stat > 0.3 and p < 1e-6
    stat_same, p_same = ks_test(a, rng.normal(0, 1, 500), 30)
    assert stat_same < 0.1 and p_same > 0.05
    assert all(math.isnan(v) for v in ks_test(a, a[:10], 30))


# ------------------------------------------------------------------- reference
def _split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = frame.drop(columns=[TARGET_COLUMN])
    half = len(features) // 2
    return features.iloc[:half].copy(), features.iloc[half:].copy()


def test_reference_covers_all_raw_variables_with_missing_bin(
    synthetic_dev_large: pd.DataFrame, mapping: VariableMapping
) -> None:
    train, _ = _split(synthetic_dev_large)
    reference = build_reference(train, mapping, DRIFT)
    assert set(reference.variables) == set(mapping.raw_en_names)
    assert reference.rows == len(train)
    for ref in reference.variables.values():
        assert ref.bins[-1] == MISSING_BIN
        assert len(ref.bins) == len(ref.proportions)
        assert sum(ref.proportions) == pytest.approx(1.0)
        if ref.kind == "numeric":
            assert ref.edges[0] == -math.inf and ref.edges[-1] == math.inf
            assert len(ref.bins) == len(ref.edges)  # bins + missing == edges - 1 + 1
            assert ref.sample == sorted(ref.sample)
        else:
            assert not ref.edges and not ref.sample


def test_reference_roundtrip_json(
    synthetic_dev_large: pd.DataFrame, mapping: VariableMapping, tmp_path: Path
) -> None:
    train, _ = _split(synthetic_dev_large)
    reference = build_reference(train, mapping, DRIFT)
    path = save_reference(reference, tmp_path / "ref.json")
    assert "Infinity" not in path.read_text(encoding="utf-8")  # strict JSON
    loaded = load_reference(path)
    assert loaded == reference


def test_reference_errors(tmp_path: Path, mapping: VariableMapping) -> None:
    with pytest.raises(DriftError, match="not found"):
        load_reference(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(DriftError, match="valid JSON"):
        load_reference(bad)
    malformed = tmp_path / "malformed.json"
    malformed.write_text(json.dumps({"source": "x", "rows": 1}), encoding="utf-8")
    with pytest.raises(DriftError, match="malformed"):
        load_reference(malformed)
    with pytest.raises(DriftError, match="lacks column"):
        build_reference(pd.DataFrame({"age": [30.0]}), mapping, DRIFT)
    assert ReferenceWindow.from_dict({"source": "x", "rows": 1, "variables": {}}).variables == {}
    edge_file = tmp_path / "edge.json"
    spec = {"name": "age", "kind": "numeric", "bins": ["a"], "proportions": [1.0], "edges": ["nan"]}
    edge_file.write_text(
        json.dumps({"source": "x", "rows": 1, "variables": {"age": spec}}), encoding="utf-8"
    )
    with pytest.raises(DriftError, match="bin edge"):
        load_reference(edge_file)


# ---------------------------------------------------------------------- report
def test_no_alert_on_same_distribution(
    synthetic_dev_large: pd.DataFrame, mapping: VariableMapping
) -> None:
    train, current = _split(synthetic_dev_large)
    report = compute_drift(build_reference(train, mapping, DRIFT), current, DRIFT)
    assert report.current_rows == len(current) and report.reference_rows == len(train)
    assert report.alerts == []
    frame = report.to_frame()
    assert list(frame["psi"]) == sorted(frame["psi"], reverse=True)
    assert set(frame["name"]) == set(mapping.raw_en_names)


def test_alert_on_shifted_variables_and_missing_share(
    synthetic_dev_large: pd.DataFrame, mapping: VariableMapping
) -> None:
    train, current = _split(synthetic_dev_large)
    current["age"] = current["age"] + 25
    current["sex"] = 1  # every respondent male
    current["cholesterol"] = np.nan  # laboratory exam stops being collected
    report = compute_drift(build_reference(train, mapping, DRIFT), current, DRIFT)
    by_name = {v.name: v for v in report.variables}
    assert by_name["age"].alert and by_name["age"].ks_alert
    assert by_name["sex"].psi_level == "alert" and by_name["sex"].ks_statistic is None
    assert by_name["cholesterol"].alert
    assert by_name["cholesterol"].missing_share_current == 1.0
    assert by_name["cholesterol"].ks_statistic is None  # no sample left for KS
    assert {"age", "sex", "cholesterol"} <= set(report.alerts)
    payload = report.to_dict()
    assert payload["alerts"] == report.alerts and len(payload["variables"]) == len(report.variables)
    text = report_markdown(report, "Teste")
    assert "| age |" in text and "alerta" in text.lower()


def test_compute_drift_errors(synthetic_dev_large: pd.DataFrame, mapping: VariableMapping) -> None:
    train, current = _split(synthetic_dev_large)
    reference = build_reference(train, mapping, DRIFT)
    with pytest.raises(DriftError, match="empty"):
        compute_drift(reference, current.iloc[0:0], DRIFT)
    with pytest.raises(DriftError, match="lacks columns"):
        compute_drift(reference, current.drop(columns=["age"]), DRIFT)


# ---------------------------------------------------------------------- window
def test_observation_window_is_bounded_and_counts_total() -> None:
    window = ObservationWindow(maxlen=3)
    window.extend([{"a": i} for i in range(5)])
    assert len(window) == 3 and window.total_received == 5
    assert window.to_frame()["a"].tolist() == [2, 3, 4]
    window.clear()
    assert len(window) == 0 and window.to_frame().empty
    with pytest.raises(ValueError):
        ObservationWindow(0)


def test_drift_monitor_snapshot_states(
    cfg: Settings, synthetic_dev_large: pd.DataFrame, mapping: VariableMapping
) -> None:
    train, current = _split(synthetic_dev_large)
    monitoring = MonitoringConfig(
        drift=DRIFT,
        retraining=RETRAINING,
        reference_window="train",
        reference_file="drift_reference.json",
        window_size=500,
        min_rows_for_drift=50,
        scenario_fixture_rows=100,
        scenarios={},
    )
    no_reference = DriftMonitor(monitoring, None)
    snap = no_reference.snapshot()
    assert not snap.reference_loaded and snap.report is None
    monitor = DriftMonitor(monitoring, build_reference(train, mapping, DRIFT))
    monitor.observe(current.head(10).to_dict("records"))
    snap = monitor.snapshot()
    assert snap.reference_loaded and not snap.enough_rows and snap.report is None
    monitor.observe(current.iloc[10:].to_dict("records"))
    snap = monitor.snapshot()
    assert snap.enough_rows and snap.report is not None
    assert snap.window_rows == min(len(current), monitoring.window_size)
    assert snap.report.alerts == []
    with pytest.raises(ValueError, match="cannot exceed"):
        MonitoringConfig(
            drift=DRIFT,
            retraining=RETRAINING,
            reference_window="train",
            reference_file="x",
            window_size=10,
            min_rows_for_drift=11,
            scenario_fixture_rows=100,
            scenarios={},
        )
