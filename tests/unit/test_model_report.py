"""Readers of the global model report: graceful degradation and correct reshaping."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.api import model_report as report

_METRICS = {
    "cv": {
        "summary": {"auc_roc_mean": 0.75, "auc_roc_std": 0.02, "sensitivity_mean": 0.68},
        "folds": [{"auc_roc": 0.7}] * 10,
    },
    "holdout": {
        "auc_roc": 0.74,
        "accuracy": 0.70,
        "true_negatives": 727,
        "false_positives": 303,
        "false_negatives": 68,
        "true_positives": 143,
        "n": 1241,
    },
}
_ROC = {
    "holdout": {
        "fpr": [0.0, 0.25, 0.5, 0.75, 1.0],
        "tpr": [0.0, 0.5, 0.8, 0.9, 1.0],
        "thresholds": [1.9, 0.8, 0.5, 0.3, 0.05],
    }
}
_INTERPRETATION = {
    "method": "coef_j * x_ij",
    "coefficients": [
        {"dummy": "a", "model_variable": "v", "label_pt": "V", "category": "1", "coefficient": 0.2},
        {
            "dummy": "b",
            "model_variable": "v",
            "label_pt": "V",
            "category": "2",
            "coefficient": -2.0,
        },
        {"dummy": "c", "model_variable": "w", "label_pt": "W", "category": "1", "coefficient": 1.0},
    ],
    "importance": [{"model_variable": "v", "label_pt": "V", "mean_abs_contribution": 0.5}],
    "contributions": {
        "dummies": ["a", "b"],
        "labels_pt": ["V = 1", "V = 2"],
        "contributions": [[0.1, -0.2], [0.0, 0.3]],
        "present": [[1, 0], [0, 1]],
        "rows": 2,
        "rows_available": 9,
    },
}


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    champion = tmp_path / "lr_production"
    champion.mkdir()
    (champion / "metrics.json").write_text(json.dumps(_METRICS), encoding="utf-8")
    (champion / "roc.json").write_text(json.dumps(_ROC), encoding="utf-8")
    (champion / "interpretation.json").write_text(json.dumps(_INTERPRETATION), encoding="utf-8")
    (champion / "subgroup_metrics.json").write_text(
        json.dumps([{"dimension": "sex", "group": "feminino", "n": 10, "auc_roc": 0.7}]),
        encoding="utf-8",
    )
    other = tmp_path / "random_forest"
    other.mkdir()
    (other / "metrics.json").write_text(
        json.dumps({"holdout": {"auc_roc": 0.72}, "cv": {"summary": {"auc_roc_mean": 0.71}}}),
        encoding="utf-8",
    )
    (other / "roc.json").write_text(json.dumps(_ROC), encoding="utf-8")
    return champion


# ------------------------------------------------------------- degradation
@pytest.mark.parametrize(
    "call",
    [
        report.summary,
        lambda d: report.roc(d, "x", 10),
        lambda d: report.threshold_sweep(d, 10),
        lambda d: report.importance(d, 5),
        report.contributions,
        report.subgroups,
    ],
)
def test_every_section_reports_absence_instead_of_raising(
    call: Callable[[Path | None], dict[str, Any]], tmp_path: Path
) -> None:
    assert call(None)["available"] is False
    empty = call(tmp_path / "empty")
    assert empty["available"] is False
    assert empty["reason"]


def test_corrupt_file_is_treated_as_absent(run_dir: Path) -> None:
    (run_dir / "metrics.json").write_text("{not json", encoding="utf-8")
    assert report.summary(run_dir)["available"] is False


# ------------------------------------------------------------------ summary
def test_summary_carries_both_evaluations(run_dir: Path) -> None:
    payload = report.summary(run_dir)
    assert payload["available"] is True
    assert payload["n_folds"] == 10
    assert payload["cross_validation"]["auc_roc_mean"] == 0.75
    assert payload["holdout"]["n"] == 1241


# ---------------------------------------------------------------------- roc
def test_roc_lists_every_sibling_run_with_the_champion_first(run_dir: Path) -> None:
    payload = report.roc(run_dir, "lr_production", 100)
    runs = [curve["run"] for curve in payload["curves"]]
    assert runs[0] == "lr_production"
    assert set(runs) == {"lr_production", "random_forest"}
    assert payload["curves"][0]["is_champion"] is True
    assert payload["curves"][0]["auc_holdout"] == 0.74
    assert payload["curves"][1]["auc_cv_mean"] == 0.71


def test_roc_is_thinned_but_keeps_the_last_point(run_dir: Path) -> None:
    payload = report.roc(run_dir, "lr_production", 2)
    curve = payload["curves"][0]
    assert len(curve["fpr"]) <= 3
    assert curve["fpr"][-1] == 1.0
    assert curve["tpr"][-1] == 1.0


def test_threshold_sweep_drops_the_artificial_first_threshold(run_dir: Path) -> None:
    payload = report.threshold_sweep(run_dir, 100)
    assert payload["available"] is True
    thresholds = [point["threshold"] for point in payload["points"]]
    assert 1.9 not in thresholds
    assert thresholds == [0.8, 0.5, 0.3, 0.05]


def test_threshold_sweep_converts_the_false_positive_rate(run_dir: Path) -> None:
    point = report.threshold_sweep(run_dir, 100)["points"][0]
    assert point["sensitivity"] == pytest.approx(0.5)
    assert point["specificity"] == pytest.approx(0.75)


# ------------------------------------------------------------- interpretation
def test_importance_ranks_coefficients_by_magnitude_not_by_sign(run_dir: Path) -> None:
    payload = report.importance(run_dir, top_coefficients=2)
    assert [row["dummy"] for row in payload["coefficients"]] == ["b", "c"]
    assert payload["variables"][0]["model_variable"] == "v"
    assert payload["method"]


def test_contributions_stay_columnar(run_dir: Path) -> None:
    payload = report.contributions(run_dir)
    assert payload["dummies"] == ["a", "b"]
    assert payload["labels_pt"] == ["V = 1", "V = 2"]
    assert payload["contributions"] == [[0.1, -0.2], [0.0, 0.3]]
    assert payload["present"] == [[1, 0], [0, 1]]
    assert payload["rows"] == 2
    assert payload["rows_available"] == 9


def test_subgroups_pass_the_rows_through(run_dir: Path) -> None:
    payload = report.subgroups(run_dir)
    assert payload["rows"][0]["group"] == "feminino"
