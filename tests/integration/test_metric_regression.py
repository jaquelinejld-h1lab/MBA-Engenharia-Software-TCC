"""Metric regression gate (decision P4).

Golden values are read from ``evidencias/golden_metrics.json``, never typed
here. Two checks:

1. the frozen golden values agree with the Phase 0 reference in the config
   (external verification, tolerance from ``fidelity.*_tolerance``);
2. a fresh fidelity run on the real dataset reproduces the golden values
   within the same tolerance (regression against the current code).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.config import Settings
from src.data.mapping import VariableMapping
from src.data.pipeline import load_development_frame
from src.features.transformer import HypertensionTransformer
from src.models.algorithms import build_estimator
from src.models.cross_validation import cross_validate

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def golden(cfg: Settings) -> dict[str, Any]:
    path = cfg.paths.absolute("golden_metrics")
    if not path.is_file():
        pytest.skip("golden metrics not frozen yet; run: python -m src.models.train --all")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


def test_golden_matches_phase0_reference(golden: dict[str, Any], cfg: Settings) -> None:
    tol = cfg.fidelity
    assert abs(golden["cv"]["auc_roc_mean"] - tol.reference_auc_cv) <= tol.auc_tolerance
    assert (
        abs(golden["cv"]["sensitivity_mean"] - tol.reference_sensitivity_cv)
        <= tol.sensitivity_tolerance
    )
    assert golden["n_splits"] == cfg.split.n_splits
    assert golden["raw_dataset_sha256"] == cfg.data.raw_sha256


def test_fidelity_run_reproduces_golden(
    golden: dict[str, Any], cfg: Settings, mapping: VariableMapping
) -> None:
    if not cfg.paths.absolute("raw_dataset").is_file():
        pytest.skip("real PNS 2013 extract not available")
    run = cfg.model.run(golden["run"])
    development = load_development_frame(cfg, materialize=False)
    transformer = HypertensionTransformer(cfg.features, mapping, mode=run.transformer_mode)
    estimator = build_estimator(run.algorithm, run.params, cfg.project.seed, development["target"])
    result = cross_validate(
        development,
        transformer,
        estimator,
        cfg.split,
        cfg.project.seed,
        cfg.model.decision_threshold,
    )
    tol = cfg.fidelity
    assert abs(result.summary["auc_roc_mean"] - golden["cv"]["auc_roc_mean"]) <= tol.auc_tolerance
    assert (
        abs(result.summary["sensitivity_mean"] - golden["cv"]["sensitivity_mean"])
        <= tol.sensitivity_tolerance
    )
    # Fold-level agreement is stricter than the tolerance and catches silent reordering.
    for fold, frozen in zip(result.fold_metrics, golden["cv_folds"], strict=True):
        assert abs(fold.auc_roc - frozen["auc_roc"]) <= tol.auc_tolerance


def test_acceptance_criteria_oe5(golden: dict[str, Any], cfg: Settings) -> None:
    assert golden["cv"]["auc_roc_mean"] >= cfg.acceptance.min_auc
    assert golden["cv"]["sensitivity_mean"] >= cfg.acceptance.min_sensitivity
