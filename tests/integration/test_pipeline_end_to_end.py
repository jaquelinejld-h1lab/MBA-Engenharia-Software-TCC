"""End-to-end pipeline on the synthetic fixture and reproducibility (OE6).

The six stages implemented so far run back to back on the fixture: raw
frame, quality gate, population filters and target, split, transformer,
training, cross-validation, holdout evaluation, artifact persistence and
tracking. The whole run must finish in under 60 seconds, and two runs from
the same seed must produce byte-identical metrics.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config import Settings
from src.data.mapping import get_mapping
from src.data.pipeline import build_development_frame, load_raw_frame
from src.data.quality import enforce_quality
from src.models import train as train_module

pytestmark = pytest.mark.integration

_RUNS = ["lr_fidelity", "lr_production"]
_MAX_SECONDS = 60.0


def _fast_config(cfg: Settings, root: Path) -> Settings:
    return cfg.model_copy(
        update={
            "split": cfg.split.model_copy(update={"n_splits": 5}),
            "paths": cfg.paths.model_copy(
                update={
                    "artifacts_dir": root / "artifacts",
                    "evidence_dir": root / "evidence",
                    "golden_metrics": root / "evidence" / "golden.json",
                }
            ),
            "tracking": cfg.tracking.model_copy(update={"json_runs_dir": root / "runs"}),
        }
    )


def _run_pipeline(cfg: Settings, root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    (root / "evidence").mkdir(exist_ok=True)
    raw = load_raw_frame(cfg, use_fixture=True)
    development, report = build_development_frame(raw, cfg, enforce_counts=False)
    enforce_quality(development, get_mapping())
    monkeypatch.setattr(train_module, "load_development_frame", lambda *a, **k: development)
    results, skipped = train_module.run_experiments(
        _fast_config(cfg, root),
        only=_RUNS,
        backend="json",
        materialize=False,
        freeze_golden=True,
        promote=True,
    )
    assert skipped == {}
    metrics = {
        r.name: json.loads((r.artifact_dir / "metrics.json").read_text(encoding="utf-8"))
        for r in results
    }
    return {"rows_out": report.rows_out, "prevalence": report.prevalence, "metrics": metrics}


def test_end_to_end_on_fixture_under_sixty_seconds(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = time.perf_counter()
    outcome = _run_pipeline(cfg, tmp_path / "run", monkeypatch)
    elapsed = time.perf_counter() - started
    assert elapsed < _MAX_SECONDS, f"pipeline took {elapsed:.1f}s"
    assert outcome["rows_out"] > 0 and 0.0 < outcome["prevalence"] < 1.0
    for name in _RUNS:
        cv = outcome["metrics"][name]["cv"]["summary"]
        assert cv["n_folds"] == 5 and 0.0 <= cv["auc_roc_mean"] <= 1.0
    registry = json.loads((tmp_path / "run" / "runs" / "registry.json").read_text())
    assert registry[cfg.tracking.registered_model_name]["versions"][0]["stage"] == "Production"


def test_two_executions_are_identical(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _run_pipeline(cfg, tmp_path / "a", monkeypatch)
    second = _run_pipeline(cfg, tmp_path / "b", monkeypatch)
    assert first == second  # every metric, fold by fold, unrounded
    golden_a = json.loads((tmp_path / "a" / "evidence" / "golden.json").read_text())
    golden_b = json.loads((tmp_path / "b" / "evidence" / "golden.json").read_text())
    assert golden_a["cv"] == golden_b["cv"] and golden_a["cv_folds"] == golden_b["cv_folds"]
    assert golden_a["development_frame_sha256"] == golden_b["development_frame_sha256"]
    frame_a = pd.read_csv(tmp_path / "a" / "evidence" / "subgroup_metrics_lr_production.csv")
    frame_b = pd.read_csv(tmp_path / "b" / "evidence" / "subgroup_metrics_lr_production.csv")
    pd.testing.assert_frame_equal(frame_a, frame_b)
