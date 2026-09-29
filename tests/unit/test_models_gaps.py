"""Model layer branches: optional libraries, MLflow tracker (faked), runner CLI and failures."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.config import Settings
from src.exceptions import ModelNotFoundError, PredictionError
from src.models import algorithms, registry
from src.models import train as train_module
from src.models.registry import JsonTracker, MlflowTracker


# -------------------------------------------------------------- algorithms
@pytest.mark.parametrize("name", ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"])
def test_optional_library_factories(name: str, cfg: Settings) -> None:
    run = next(r for r in cfg.model.runs if r.algorithm == name)
    y = pd.Series([0] * 8 + [1] * 2)
    if algorithms.library_available(name):
        estimator = algorithms.build_estimator(name, run.params, cfg.project.seed, y)
        assert hasattr(estimator, "fit")
    else:
        with pytest.raises(ModelNotFoundError, match="not installed"):
            algorithms.build_estimator(name, run.params, cfg.project.seed, y)


def test_auto_class_weights_resolution() -> None:
    y = pd.Series([0] * 6 + [1] * 2)
    resolved = algorithms._resolve({"class_weights": "auto", "scale_pos_weight": "auto"}, y)
    assert resolved == {"class_weights": [1.0, 3.0], "scale_pos_weight": 3.0}


# ------------------------------------------------------- MLflow (faked)
class _FakeClient:
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def transition_model_version_stage(self, **kwargs: Any) -> None:
        self.calls.append(("transition", kwargs))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.calls.append(("alias", (name, alias, version)))


def _fake_mlflow(calls: list[tuple[str, Any]]) -> types.ModuleType:
    module = types.ModuleType("mlflow")
    module.set_tracking_uri = lambda uri: calls.append(("uri", uri))  # type: ignore[attr-defined]
    module.get_experiment_by_name = lambda name: None  # type: ignore[attr-defined]
    module.create_experiment = lambda name, artifact_location: calls.append(("create", name))  # type: ignore[attr-defined]
    module.set_experiment = lambda name: calls.append(("set_experiment", name))  # type: ignore[attr-defined]
    module.start_run = lambda run_name, tags: types.SimpleNamespace(  # type: ignore[attr-defined]
        info=types.SimpleNamespace(run_id="abc123")
    )
    module.end_run = lambda status: calls.append(("end", status))  # type: ignore[attr-defined]
    module.log_params = lambda p: calls.append(("params", p))  # type: ignore[attr-defined]
    module.log_metrics = lambda m: calls.append(("metrics", m))  # type: ignore[attr-defined]
    module.log_dict = lambda d, f: calls.append(("dict", f))  # type: ignore[attr-defined]
    module.log_artifact = lambda p: calls.append(("artifact", p))  # type: ignore[attr-defined]
    module.set_tags = lambda t: calls.append(("tags", t))  # type: ignore[attr-defined]
    module.sklearn = types.SimpleNamespace(  # type: ignore[attr-defined]
        log_model=lambda model, artifact_path: calls.append(("log_model", artifact_path))
    )
    module.register_model = lambda uri, name: types.SimpleNamespace(version=7)  # type: ignore[attr-defined]
    module.tracking = types.SimpleNamespace(MlflowClient=lambda: _FakeClient(calls))  # type: ignore[attr-defined]
    module.search_runs = lambda experiment_names: pd.DataFrame(  # type: ignore[attr-defined]
        [{"tags.mlflow.runName": "demo"}]
    )
    return module


def test_mlflow_tracker_call_sequence(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Any]] = []
    monkeypatch.setitem(sys.modules, "mlflow", _fake_mlflow(calls))
    tracking = cfg.tracking.model_copy(
        update={"sqlite_path": tmp_path / "mlflow.db", "artifact_root": tmp_path / "artifacts"}
    )
    tracker = MlflowTracker(tracking)
    assert calls[0][0] == "uri" and calls[0][1].startswith("sqlite:///")
    run = tracker.start_run("demo", {"role": "x"})
    assert run.run_id == "abc123"
    run.log_params({"C": 1.0, "layers": (100,)})
    run.log_metrics({"auc": 0.7})
    run.log_dict({"a": 1}, "a.json")
    run.log_artifact(tmp_path / "x")
    run.set_tags({"k": "v"})
    tracker.end_run(run)

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    import joblib  # noqa: PLC0415

    joblib.dump({"pipeline": True}, model_dir / "pipeline.joblib")
    version = tracker.register_model(run, model_dir, "hypertension-lr")
    assert version == "7"
    tracker.promote("hypertension-lr", version, "Production", "champion")
    kinds = [c[0] for c in calls]
    for expected in (
        "params",
        "metrics",
        "dict",
        "artifact",
        "tags",
        "end",
        "log_model",
        "transition",
        "alias",
    ):
        assert expected in kinds
    transition = next(c[1] for c in calls if c[0] == "transition")
    assert transition == {
        "name": "hypertension-lr",
        "version": "7",
        "stage": "Production",
        "archive_existing_versions": True,
    }
    assert tracker.list_runs()[0]["tags.mlflow.runName"] == "demo"


def test_mlflow_tracker_missing_library(cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)  # makes `import mlflow` fail
    with pytest.raises(ModelNotFoundError, match="mlflow is not installed"):
        MlflowTracker(cfg.tracking)


def test_json_tracker_rejects_foreign_run(cfg: Settings, tmp_path: Path) -> None:
    tracker = JsonTracker(cfg.tracking.model_copy(update={"json_runs_dir": tmp_path}))
    with pytest.raises(ModelNotFoundError):
        tracker.end_run(object())  # type: ignore[arg-type]


def test_git_revision_unknown_outside_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert registry.git_revision() == "unknown"


# ------------------------------------------------------------------ runner
def test_execute_run_marks_failure(
    cfg: Settings, synthetic_dev: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fast = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(
                update={"artifacts_dir": tmp_path / "a", "evidence_dir": tmp_path}
            ),
            "tracking": cfg.tracking.model_copy(update={"json_runs_dir": tmp_path / "runs"}),
        }
    )
    monkeypatch.setattr(train_module, "load_development_frame", lambda *a, **k: synthetic_dev)

    def _fail(*args: Any, **kwargs: Any) -> None:
        raise PredictionError("forced")

    monkeypatch.setattr(train_module, "cross_validate", _fail)
    context = train_module.build_context(fast, materialize=False)
    tracker = JsonTracker(fast.tracking)
    with pytest.raises(PredictionError):
        train_module.execute_run(fast, fast.model.run("lr_production"), context, tracker)
    assert tracker.list_runs()[0]["status"] == "FAILED"


def test_train_cli_dispatch(monkeypatch: pytest.MonkeyPatch, cfg: Settings) -> None:
    captured: dict[str, Any] = {}

    def _fake(cfg_: Settings, **kwargs: Any) -> tuple[list[Any], dict[str, str]]:
        captured.update(kwargs)
        return [object()], {}

    monkeypatch.setattr(train_module, "run_experiments", _fake)
    monkeypatch.setattr(train_module, "configure_logging", lambda c: None)
    assert train_module._main(["--run", "lr_fidelity", "--backend", "json", "--no-promote"]) == 0
    assert captured["only"] == ["lr_fidelity"] and captured["backend"] == "json"
    assert captured["promote"] is False and captured["materialize"] is True

    monkeypatch.setattr(train_module, "run_experiments", lambda c, **k: ([], {"x": "missing"}))
    assert train_module._main(["--all", "--no-materialize"]) == 1
