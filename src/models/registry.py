"""Experiment tracking and model registry behind one small interface.

``MlflowTracker`` is the production implementation (MLflow 2.x, SQLite
backend, local artifact root). ``JsonTracker`` writes the same information to
``evidencias/runs/`` and exists for two reasons: unit tests must not depend
on MLflow, and the evidence folder needs a plain-text copy of every run for
the written work anyway. Both implement :class:`ExperimentTracker`.

Model promotion follows decision P2: the run named in
``tracking.promote_run`` is registered and moved to the ``Production`` stage
(plus an alias, which is the non-deprecated mechanism in recent MLflow).
"""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404 - used only to read the git revision
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config import TrackingConfig
from src.exceptions import ModelNotFoundError
from src.logging_setup import get_logger
from src.paths import PROJECT_ROOT, resolve

logger = get_logger(__name__)


# ------------------------------------------------------------------ helpers
def git_revision() -> str:
    """Return the short git commit, or ``"unknown"`` outside a repository."""
    try:
        out = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607  (git resolved on PATH)
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def frame_sha256(frame: pd.DataFrame) -> str:
    """Content hash of a frame (values and column names, index ignored)."""
    digest = hashlib.sha256()
    digest.update(",".join(map(str, frame.columns)).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def source_sha256(*paths: Path) -> str:
    """Hash of one or more source files, used to version the transformer code."""
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


# ---------------------------------------------------------------- interface
class TrackedRun(Protocol):
    """One open run."""

    @property
    def run_id(self) -> str: ...

    def log_params(self, params: dict[str, Any]) -> None: ...

    def log_metrics(self, metrics: dict[str, float]) -> None: ...

    def log_dict(self, payload: dict[str, Any], filename: str) -> None: ...

    def log_artifact(self, path: Path) -> None: ...

    def set_tags(self, tags: dict[str, str]) -> None: ...


class ExperimentTracker(Protocol):
    """Tracking backend."""

    def start_run(self, name: str, tags: dict[str, str]) -> TrackedRun: ...

    def end_run(self, run: TrackedRun, status: str = "FINISHED") -> None: ...

    def register_model(self, run: TrackedRun, model_dir: Path, name: str) -> str: ...

    def promote(self, name: str, version: str, stage: str, alias: str) -> None: ...

    def list_runs(self) -> list[dict[str, Any]]: ...


# ------------------------------------------------------------ JSON backend
@dataclass
class _JsonRun:
    run_id: str
    name: str
    started_at: str
    directory: Path
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, str] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    dicts: dict[str, dict[str, Any]] = field(default_factory=dict)

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update({k: _jsonable(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self.metrics.update({k: float(v) for k, v in metrics.items()})

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        self.dicts[filename] = payload

    def log_artifact(self, path: Path) -> None:
        self.artifacts.append(_display_path(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.tags.update(tags)


def _display_path(path: Path) -> str:
    """Project-relative path when inside the project, absolute otherwise."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _jsonable(value: Any) -> Any:  # noqa: ANN401
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


class JsonTracker:
    """File-based tracker: one JSON per run plus a registry file."""

    def __init__(self, tracking: TrackingConfig) -> None:
        self.directory = resolve(tracking.json_runs_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.directory / "registry.json"

    def start_run(self, name: str, tags: dict[str, str]) -> _JsonRun:
        started = datetime.now(UTC).isoformat(timespec="seconds")
        run_id = hashlib.sha256(f"{name}{started}".encode()).hexdigest()[:12]
        run = _JsonRun(run_id=run_id, name=name, started_at=started, directory=self.directory)
        run.set_tags(tags)
        return run

    def end_run(self, run: TrackedRun, status: str = "FINISHED") -> None:
        if not isinstance(run, _JsonRun):
            raise ModelNotFoundError("JsonTracker received a run from another backend")
        payload = {
            "run_id": run.run_id,
            "name": run.name,
            "status": status,
            "started_at": run.started_at,
            "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "tags": run.tags,
            "params": run.params,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
            **run.dicts,
        }
        (self.directory / f"{run.name}.json").write_text(
            json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
        )

    def register_model(self, run: TrackedRun, model_dir: Path, name: str) -> str:
        registry = self._read_registry()
        versions = registry.setdefault(name, {"versions": []})["versions"]
        version = str(len(versions) + 1)
        versions.append(
            {
                "version": version,
                "run_id": run.run_id,
                "run_name": getattr(run, "name", ""),
                "model_dir": _display_path(model_dir),
                "stage": "None",
                "aliases": [],
            }
        )
        self._write_registry(registry)
        return version

    def promote(self, name: str, version: str, stage: str, alias: str) -> None:
        registry = self._read_registry()
        entries = registry.get(name, {}).get("versions", [])
        target = next((e for e in entries if e["version"] == version), None)
        if target is None:
            raise ModelNotFoundError(f"model '{name}' version {version} is not registered")
        for entry in entries:
            if entry["stage"] == stage:
                entry["stage"] = "Archived"
            if alias in entry["aliases"]:
                entry["aliases"].remove(alias)
        target["stage"] = stage
        target["aliases"].append(alias)
        self._write_registry(registry)

    def list_runs(self) -> list[dict[str, Any]]:
        runs = []
        for file in sorted(self.directory.glob("*.json")):
            if file.name == "registry.json":
                continue
            runs.append(json.loads(file.read_text(encoding="utf-8")))
        return runs

    def _read_registry(self) -> dict[str, Any]:
        if self.registry_file.is_file():
            data: dict[str, Any] = json.loads(self.registry_file.read_text(encoding="utf-8"))
            return data
        return {}

    def _write_registry(self, registry: dict[str, Any]) -> None:
        self.registry_file.write_text(json.dumps(registry, indent=1), encoding="utf-8")


# ---------------------------------------------------------- MLflow backend
class _MlflowRun:
    def __init__(self, mlflow_module: Any, run_id: str) -> None:  # noqa: ANN401
        self._mlflow = mlflow_module
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        return self._run_id

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: _jsonable(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        self._mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_dict(self, payload: dict[str, Any], filename: str) -> None:
        self._mlflow.log_dict(payload, filename)

    def log_artifact(self, path: Path) -> None:
        self._mlflow.log_artifact(str(path))

    def set_tags(self, tags: dict[str, str]) -> None:
        self._mlflow.set_tags(tags)


class MlflowTracker:
    """MLflow with SQLite tracking store and local artifact root.

    Notes
    -----
    Uses ``transition_model_version_stage`` because the project requires the
    ``Staging``/``Production`` stages, and additionally sets a registered model
    alias, the mechanism MLflow recommends since 2.9.
    """

    def __init__(self, tracking: TrackingConfig) -> None:
        try:
            import mlflow  # noqa: PLC0415  (optional dependency)
        except ImportError as error:
            raise ModelNotFoundError(
                "mlflow is not installed; use tracking.backend=json or install "
                "requirements-train.lock"
            ) from error
        self._mlflow = mlflow
        db_path = resolve(tracking.sqlite_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_root = resolve(tracking.artifact_root)
        artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        if mlflow.get_experiment_by_name(tracking.experiment_name) is None:
            mlflow.create_experiment(
                tracking.experiment_name, artifact_location=artifact_root.as_uri()
            )
        mlflow.set_experiment(tracking.experiment_name)
        self.experiment_name = tracking.experiment_name

    def start_run(self, name: str, tags: dict[str, str]) -> _MlflowRun:
        active = self._mlflow.start_run(run_name=name, tags=tags)
        return _MlflowRun(self._mlflow, active.info.run_id)

    def end_run(self, run: TrackedRun, status: str = "FINISHED") -> None:
        self._mlflow.end_run(status=status)

    def register_model(self, run: TrackedRun, model_dir: Path, name: str) -> str:
        import joblib  # noqa: PLC0415

        pipeline = joblib.load(model_dir / "pipeline.joblib")
        self._mlflow.sklearn.log_model(pipeline, artifact_path="model")
        # The drift reference travels with the registered version (read by the API).
        for extra in model_dir.glob("*.json"):
            self._mlflow.log_artifact(str(extra))
        version = self._mlflow.register_model(f"runs:/{run.run_id}/model", name)
        return str(version.version)

    def promote(self, name: str, version: str, stage: str, alias: str) -> None:
        client = self._mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name, version=version, stage=stage, archive_existing_versions=True
        )
        client.set_registered_model_alias(name, alias, version)

    def list_runs(self) -> list[dict[str, Any]]:
        frame = self._mlflow.search_runs(experiment_names=[self.experiment_name])
        records: list[dict[str, Any]] = frame.to_dict(orient="records")
        return records


def make_tracker(tracking: TrackingConfig, backend: str | None = None) -> ExperimentTracker:
    """Instantiate the tracker selected by config (or by the override)."""
    choice = backend or tracking.backend
    if choice == "mlflow":
        return MlflowTracker(tracking)
    if choice == "json":
        return JsonTracker(tracking)
    raise ModelNotFoundError(f"unknown tracking backend '{choice}'")
