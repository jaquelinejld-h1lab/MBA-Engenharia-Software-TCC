"""Unit tests of evaluate, cross-validation, algorithms, trackers and the runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.config import Settings
from src.data.mapping import VariableMapping
from src.exceptions import ModelNotFoundError, PredictionError
from src.features.transformer import HypertensionTransformer
from src.models import train as train_module
from src.models.algorithms import (
    SUPPORTED_ALGORITHMS,
    build_estimator,
    library_available,
    positive_class_weight,
)
from src.models.cross_validation import cross_validate
from src.models.evaluate import compute_metrics, roc_points, subgroup_metrics, summarize_folds
from src.models.registry import JsonTracker, frame_sha256, git_revision, make_tracker


# ---------------------------------------------------------------- evaluate
def test_compute_metrics_known_values() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.2, 0.9, 0.3, 0.8, 0.7])
    m = compute_metrics(y, p, 0.5)
    assert (m.true_negatives, m.false_positives, m.false_negatives, m.true_positives) == (
        3,
        1,
        1,
        3,
    )
    assert m.sensitivity == 0.75 and m.specificity == 0.75 and m.accuracy == 0.75
    assert m.precision == 0.75 and m.f1 == 0.75
    assert m.auc_roc == pytest.approx(14 / 16)
    assert m.n == 8 and m.positives == 4 and m.threshold == 0.5


def test_compute_metrics_uses_explicit_predictions_when_given() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4])  # all below 0.5
    default = compute_metrics(y, p, 0.5)
    explicit = compute_metrics(y, p, 0.5, y_pred=np.array([0, 0, 1, 1]))
    assert default.sensitivity == 0.0 and explicit.sensitivity == 1.0
    assert default.auc_roc == explicit.auc_roc == 1.0


def test_compute_metrics_rejects_bad_inputs() -> None:
    with pytest.raises(PredictionError):
        compute_metrics(np.array([1, 1]), np.array([0.5, 0.6]), 0.5)
    with pytest.raises(PredictionError):
        compute_metrics(np.array([0, 1]), np.array([0.5]), 0.5)
    with pytest.raises(PredictionError):
        compute_metrics(np.array([0, 1]), np.array([0.5, 0.6]), 0.5, y_pred=np.array([1]))


def test_summarize_folds_population_std() -> None:
    a = compute_metrics(np.array([0, 1]), np.array([0.2, 0.8]), 0.5)
    b = compute_metrics(np.array([0, 1]), np.array([0.8, 0.2]), 0.5)
    s = summarize_folds([a, b])
    assert s["auc_roc_mean"] == 0.5 and s["auc_roc_std"] == 0.5 and s["n_folds"] == 2
    with pytest.raises(PredictionError):
        summarize_folds([])


def test_subgroup_metrics_reports_small_groups_as_nan() -> None:
    y = np.array([0, 1, 0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7, 0.9, 0.1])
    groups = pd.Series(["a", "a", "a", "a", "b", "b"])
    table = subgroup_metrics(y, p, groups, 0.5, min_size=3)
    assert table.set_index("group").loc["a", "auc_roc"] == 1.0
    assert np.isnan(table.set_index("group").loc["b", "auc_roc"])
    assert table["n"].tolist() == [4, 2]


def test_roc_points_shapes() -> None:
    pts = roc_points(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.6, 0.4]))
    assert len(pts.fpr) == len(pts.tpr) == len(pts.thresholds)
    assert pts.fpr[0] == 0.0 and pts.tpr[-1] == 1.0


# -------------------------------------------------------------- algorithms
def test_build_estimator_resolves_tokens(cfg: Settings) -> None:
    y = pd.Series([0] * 8 + [1] * 2)
    assert positive_class_weight(y) == 4.0
    lr = build_estimator(
        "LogisticRegression", cfg.model.run("lr_baseline_unweighted").params, 42, y
    )
    assert isinstance(lr, LogisticRegression) and lr.class_weight is None and lr.random_state == 42
    mlp = build_estimator("MLPClassifier", cfg.model.run("mlp").params, 42, y)
    assert mlp.hidden_layer_sizes == (100,)
    with pytest.raises(ModelNotFoundError, match="unknown algorithm"):
        build_estimator("Nope", {}, 42, y)
    with pytest.raises(ModelNotFoundError):
        positive_class_weight(pd.Series([0, 0]))


def test_every_configured_algorithm_is_supported(cfg: Settings) -> None:
    for run in cfg.model.runs:
        assert run.algorithm in SUPPORTED_ALGORITHMS
    assert library_available("LogisticRegression")
    assert not library_available("Nope")


# --------------------------------------------------------- cross-validation
@pytest.mark.parametrize("mode", ["fidelity", "production"])
def test_cross_validate_covers_every_row(
    mode: str, cfg: Settings, mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> None:
    split = cfg.split.model_copy(update={"n_splits": 3})
    transformer = HypertensionTransformer(cfg.features, mapping, mode=mode)  # type: ignore[arg-type]
    estimator = build_estimator(
        "LogisticRegression", cfg.model.champion.params.model_dump(), 42, synthetic_dev["target"]
    )
    result = cross_validate(synthetic_dev, transformer, estimator, split, 42, 0.5)
    assert len(result.fold_metrics) == 3
    assert not np.isnan(result.oof_probabilities).any()
    assert set(result.fold_of_row) == {0, 1, 2}
    assert 0.0 <= result.summary["auc_roc_mean"] <= 1.0


def test_cross_validate_is_deterministic(
    cfg: Settings, mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> None:
    split = cfg.split.model_copy(update={"n_splits": 3})
    args = (
        synthetic_dev,
        HypertensionTransformer(cfg.features, mapping, mode="production"),
        LogisticRegression(solver="liblinear", random_state=42),
        split,
        42,
        0.5,
    )
    a, b = cross_validate(*args), cross_validate(*args)
    assert a.summary == b.summary
    np.testing.assert_array_equal(a.oof_probabilities, b.oof_probabilities)


# ---------------------------------------------------------------- registry
def test_json_tracker_round_trip(cfg: Settings, tmp_path: Path) -> None:
    tracking = cfg.tracking.model_copy(update={"json_runs_dir": tmp_path / "runs"})
    tracker = JsonTracker(tracking)
    run = tracker.start_run("demo", {"role": "candidate"})
    run.log_params({"C": 1.0, "layers": [100]})
    run.log_metrics({"auc": 0.7})
    run.log_dict({"x": 1}, "extra.json")
    artifact = tmp_path / "runs" / "a.txt"
    artifact.write_text("x")
    run.log_artifact(artifact)
    tracker.end_run(run)
    runs = tracker.list_runs()
    assert runs[0]["name"] == "demo" and runs[0]["metrics"] == {"auc": 0.7}
    assert runs[0]["params"]["layers"] == [100] and runs[0]["extra.json"] == {"x": 1}

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    v1 = tracker.register_model(run, model_dir, "m")
    v2 = tracker.register_model(run, model_dir, "m")
    assert (v1, v2) == ("1", "2")
    tracker.promote("m", "1", "Production", "champion")
    tracker.promote("m", "2", "Production", "champion")
    registry = json.loads(tracker.registry_file.read_text())
    stages = {e["version"]: e["stage"] for e in registry["m"]["versions"]}
    assert stages == {"1": "Archived", "2": "Production"}
    assert registry["m"]["versions"][1]["aliases"] == ["champion"]
    with pytest.raises(ModelNotFoundError):
        tracker.promote("m", "9", "Production", "champion")


def test_make_tracker_backends(cfg: Settings, tmp_path: Path) -> None:
    tracking = cfg.tracking.model_copy(update={"json_runs_dir": tmp_path})
    assert isinstance(make_tracker(tracking, "json"), JsonTracker)
    with pytest.raises(ModelNotFoundError):
        make_tracker(tracking, "bogus")
    pytest.importorskip("mlflow")
    from src.models.registry import MlflowTracker  # noqa: PLC0415

    mlflow_tracking = tracking.model_copy(
        update={"sqlite_path": tmp_path / "mlflow.db", "artifact_root": tmp_path / "artifacts"}
    )
    tracker = MlflowTracker(mlflow_tracking)
    run = tracker.start_run("demo", {"role": "candidate"})
    run.log_params({"C": 1.0})
    run.log_metrics({"auc": 0.7})
    tracker.end_run(run)
    assert any(r["tags.mlflow.runName"] == "demo" for r in tracker.list_runs())


def test_provenance_helpers(synthetic_dev: pd.DataFrame) -> None:
    assert frame_sha256(synthetic_dev) == frame_sha256(synthetic_dev.copy())
    assert frame_sha256(synthetic_dev) != frame_sha256(synthetic_dev.head(10))
    assert isinstance(git_revision(), str) and git_revision()


# ------------------------------------------------------------------ runner
def test_execute_run_end_to_end_on_fixture(
    cfg: Settings, synthetic_dev: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = cfg.model_copy(
        update={
            "split": cfg.split.model_copy(update={"n_splits": 3}),
            "paths": cfg.paths.model_copy(
                update={
                    "artifacts_dir": tmp_path / "artifacts",
                    "evidence_dir": tmp_path / "ev",
                    "golden_metrics": tmp_path / "ev" / "golden.json",
                }
            ),
            "tracking": cfg.tracking.model_copy(update={"json_runs_dir": tmp_path / "runs"}),
        }
    )
    (tmp_path / "ev").mkdir()
    monkeypatch.setattr(train_module, "load_development_frame", lambda *a, **k: synthetic_dev)
    results, skipped = train_module.run_experiments(
        fast,
        only=["lr_fidelity", "lr_production"],
        backend="json",
        materialize=False,
        freeze_golden=True,
        promote=True,
    )
    assert [r.name for r in results] == ["lr_fidelity", "lr_production"] and skipped == {}
    for r in results:
        for name in ("transformer", "knn_imputers", "onehot_encoder", "model", "pipeline"):
            assert (r.artifact_dir / f"{name}.joblib").is_file()
        metrics = json.loads((r.artifact_dir / "metrics.json").read_text())
        assert set(metrics) == {"cv", "holdout"}
    golden = json.loads((tmp_path / "ev" / "golden.json").read_text())
    assert golden["run"] == "lr_fidelity" and "auc_roc_mean" in golden["cv"]
    assert (tmp_path / "ev" / "experiments.md").is_file()
    assert (tmp_path / "ev" / "subgroup_metrics_lr_production.csv").is_file()
    assert (tmp_path / "ev" / "roc_holdout.png").is_file()
    registry = json.loads((tmp_path / "runs" / "registry.json").read_text())
    assert registry[cfg.tracking.registered_model_name]["versions"][0]["stage"] == "Production"
    run_files = {p.name for p in (tmp_path / "runs").glob("*.json")}
    assert {"lr_fidelity.json", "lr_production.json", "registry.json"} <= run_files
    logged = json.loads((tmp_path / "runs" / "lr_production.json").read_text())
    assert logged["params"]["raw_dataset_sha256"] == cfg.data.raw_sha256
    assert "development_frame_sha256" in logged["params"]


def test_skipped_run_when_library_missing(
    cfg: Settings, synthetic_dev: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={
                    "artifacts_dir": tmp_path,
                    "evidence_dir": tmp_path,
                    "golden_metrics": tmp_path / "golden.json",
                }
            ),
            "tracking": cfg.tracking.model_copy(update={"json_runs_dir": tmp_path / "runs"}),
        }
    )
    monkeypatch.setattr(train_module, "load_development_frame", lambda *a, **k: synthetic_dev)
    monkeypatch.setattr(train_module, "library_available", lambda name: False)
    results, skipped = train_module.run_experiments(
        fast,
        only=["xgboost"],
        backend="json",
        materialize=False,
        freeze_golden=False,
        promote=False,
    )
    assert results == [] and "xgboost" in skipped
