"""Retraining flag: the decision, its reasons, persistence and the insufficient window."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import RetrainingConfig
from src.monitoring.drift import DriftReport, VariableDrift
from src.monitoring.retraining import decide, load, report_markdown, save

CONFIG = RetrainingConfig(min_variables_in_alert=3, flag_file="retraining_flag.json")


def _variable(name: str, *, psi: float, alert: bool, level: str = "alert") -> VariableDrift:
    """One variable of a drift report; ``alert`` is derived from the level and the KS."""
    return VariableDrift(
        name=name,
        kind="numeric",
        rows=500,
        psi=psi,
        psi_level=level,  # type: ignore[arg-type]
        ks_statistic=0.3 if alert else 0.01,
        ks_pvalue=0.001 if alert else 0.9,
        ks_alert=alert,
        missing_share_reference=0.0,
        missing_share_current=0.0,
    )


def _report(alerts: int, warnings: int = 0, rows: int = 500) -> DriftReport:
    variables = [_variable(f"a{i}", psi=0.5, alert=True) for i in range(alerts)]
    variables += [
        _variable(f"w{i}", psi=0.15, alert=False, level="warning") for i in range(warnings)
    ]
    variables.append(_variable("estavel", psi=0.01, alert=False, level="stable"))
    return DriftReport(reference_rows=1000, current_rows=rows, variables=variables)


def test_stable_population_does_not_recommend() -> None:
    decision = decide(_report(alerts=0), CONFIG, min_rows=100)
    assert decision.recommended is False
    assert decision.evaluated is True
    assert decision.variables_in_alert == []
    assert "estavel" in decision.summary


def test_alerts_below_the_minimum_do_not_recommend() -> None:
    """One variable moving is sample noise, and the flag says so without recommending."""
    decision = decide(_report(alerts=2), CONFIG, min_rows=100)
    assert decision.recommended is False
    assert len(decision.variables_in_alert) == 2
    assert len(decision.reasons) == 2


def test_enough_alerts_recommend_and_name_every_reason() -> None:
    decision = decide(_report(alerts=3), CONFIG, min_rows=100)
    assert decision.recommended is True
    assert decision.variables_in_alert == ["a0", "a1", "a2"]
    assert all("PSI" in reason for reason in decision.reasons)
    assert all("KS" in reason for reason in decision.reasons)


def test_warnings_are_reported_but_do_not_recommend() -> None:
    decision = decide(_report(alerts=0, warnings=4), CONFIG, min_rows=100)
    assert decision.recommended is False
    assert decision.variables_in_warning == ["w0", "w1", "w2", "w3"]


def test_small_window_decides_nothing_which_is_not_the_same_as_being_fine() -> None:
    decision = decide(_report(alerts=5, rows=10), CONFIG, min_rows=100)
    assert decision.evaluated is False
    assert decision.recommended is False
    assert "insuficiente" in decision.summary


def test_missing_report_decides_nothing() -> None:
    decision = decide(None, CONFIG, min_rows=100)
    assert decision.evaluated is False
    assert decision.recommended is False
    assert decision.current_rows == 0


def test_decision_round_trips_through_the_file(tmp_path: Path) -> None:
    decision = decide(_report(alerts=3), CONFIG, min_rows=100)
    path = save(decision, tmp_path / "flag.json")
    restored = load(path)
    assert restored == decision


def test_absent_or_corrupt_flag_reads_as_none(tmp_path: Path) -> None:
    assert load(tmp_path / "nao_existe.json") is None
    corrupt = tmp_path / "flag.json"
    corrupt.write_text("{quebrado", encoding="utf-8")
    assert load(corrupt) is None


def test_markdown_states_the_decision_and_that_it_only_recommends() -> None:
    text = report_markdown(decide(_report(alerts=3), CONFIG, min_rows=100))
    assert "retreinar" in text
    assert "a0" in text
    assert "recomenda, nao executa" in text


def test_threshold_is_configurable() -> None:
    strict = RetrainingConfig(min_variables_in_alert=5, flag_file="f.json")
    assert decide(_report(alerts=3), strict, min_rows=100).recommended is False
    assert decide(_report(alerts=5), strict, min_rows=100).recommended is True


def test_minimum_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="min_variables_in_alert"):
        RetrainingConfig(min_variables_in_alert=0, flag_file="f.json")
