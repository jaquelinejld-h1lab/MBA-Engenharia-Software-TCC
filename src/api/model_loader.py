"""Load the promoted pipeline (transformer + estimator) once, with metadata.

Two sources, selected by ``api.model_source``:

* ``mlflow``: the registered model ``tracking.registered_model_name`` at the
  alias ``tracking.promote_alias`` (``models:/<name>@<alias>``), resolved
  through ``MlflowClient``; version and stage come from the registry.
* ``local``: ``<api.local_run_dir>/pipeline.joblib`` written by the training
  runner, with ``registration.json`` next to it for version and stage.

Both fail fast with ``ModelNotFoundError`` naming what is missing. The loaded
object is cached in the process by :class:`ModelCache`.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import joblib
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

from src.config import FeaturesConfig, Settings
from src.exceptions import ModelNotFoundError
from src.features.transformer import HypertensionTransformer
from src.logging_setup import get_logger
from src.monitoring.drift import DriftError, ReferenceWindow, load_reference
from src.paths import resolve

logger = get_logger(__name__)

_PIPELINE_FILE = "pipeline.joblib"
_REGISTRATION_FILE = "registration.json"


@dataclass(frozen=True)
class LoadedModel:
    """Pipeline plus registry metadata."""

    pipeline: Pipeline
    name: str
    version: str
    stage: str
    source: str
    run_dir: Path | None = None
    run_id: str | None = None

    @property
    def transformer(self) -> HypertensionTransformer:
        """The fitted transformer step."""
        step = self.pipeline.named_steps["transformer"]
        if not isinstance(step, HypertensionTransformer):
            raise ModelNotFoundError("pipeline step 'transformer' is not a HypertensionTransformer")
        return step

    @property
    def estimator(self) -> object:
        """The fitted estimator step."""
        return self.pipeline.named_steps["model"]


def _check_pipeline(obj: object, origin: str) -> Pipeline:
    if not isinstance(obj, Pipeline) or set(obj.named_steps) != {"transformer", "model"}:
        raise ModelNotFoundError(
            f"{origin}: expected a Pipeline with steps 'transformer' and 'model'"
        )
    if not hasattr(obj.named_steps["model"], "predict_proba"):
        raise ModelNotFoundError(f"{origin}: estimator has no predict_proba")
    transformer = obj.named_steps["transformer"]
    if not isinstance(transformer, HypertensionTransformer):
        raise ModelNotFoundError(f"{origin}: step 'transformer' is not a HypertensionTransformer")
    # The transformer pickles its FeaturesConfig. An artifact trained under an older
    # config schema would fail at prediction time with an AttributeError; check the
    # schema now and fail fast with an actionable message instead.
    try:
        FeaturesConfig.model_validate(transformer.features.model_dump())
    except (ValidationError, AttributeError) as error:
        raise ModelNotFoundError(
            f"{origin}: artifact was trained with an incompatible configuration schema "
            f"({type(error).__name__}); retrain with python -m src.models.train --all"
        ) from error
    return obj


def load_local(run_dir: Path, default_name: str) -> LoadedModel:
    """Load the pipeline from a training run directory.

    Raises
    ------
    ModelNotFoundError
        If the directory or the pipeline file does not exist.
    """
    pipeline_path = run_dir / _PIPELINE_FILE
    if not pipeline_path.is_file():
        raise ModelNotFoundError(
            f"no pipeline at '{pipeline_path}'. Train and promote first: "
            "python -m src.models.train --all"
        )
    pipeline = _check_pipeline(joblib.load(pipeline_path), str(pipeline_path))
    registration_path = run_dir / _REGISTRATION_FILE
    if registration_path.is_file():
        meta = json.loads(registration_path.read_text(encoding="utf-8"))
        name, version, stage = meta["name"], str(meta["version"]), meta["stage"]
    else:
        name, version, stage = default_name, "unregistered", "None"
    return LoadedModel(
        pipeline=pipeline,
        name=name,
        version=version,
        stage=stage,
        source="local",
        run_dir=run_dir,
        run_id=str(meta.get("run_id")) if registration_path.is_file() else None,
    )


def load_from_mlflow(cfg: Settings) -> LoadedModel:
    """Load the pipeline registered under the promotion alias in the MLflow registry.

    Raises
    ------
    ModelNotFoundError
        If MLflow is not installed, the model or alias does not exist, or the
        artifact is not a valid pipeline.
    """
    try:
        import mlflow  # noqa: PLC0415  (optional at import time, required at runtime)
    except ImportError as error:
        raise ModelNotFoundError("mlflow is not installed; set api.model_source=local") from error
    tracking = cfg.tracking
    db_path = resolve(tracking.sqlite_path)
    mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
    client = mlflow.tracking.MlflowClient()
    try:
        version_info = client.get_model_version_by_alias(
            tracking.registered_model_name, tracking.promote_alias
        )
    except Exception as error:  # mlflow raises MlflowException / RestException
        raise ModelNotFoundError(
            f"registered model '{tracking.registered_model_name}' has no alias "
            f"'{tracking.promote_alias}' in the registry at {db_path}: {error}"
        ) from error
    uri = f"models:/{tracking.registered_model_name}@{tracking.promote_alias}"
    pipeline = _check_pipeline(mlflow.sklearn.load_model(uri), uri)
    stage = getattr(version_info, "current_stage", None) or cfg.api.model_stage
    return LoadedModel(
        pipeline=pipeline,
        name=tracking.registered_model_name,
        version=str(version_info.version),
        stage=str(stage),
        source="mlflow",
        run_id=str(getattr(version_info, "run_id", "") or "") or None,
    )


def load_drift_reference(cfg: Settings, loaded: LoadedModel) -> ReferenceWindow | None:
    """Locate and read the training reference window of the served model.

    Local source: ``<run_dir>/<monitoring.reference_file>``. MLflow source:
    the file logged on the registration run, downloaded to a temporary
    directory. A missing or unreadable reference disables drift gauges
    (logged as a warning) instead of preventing the API from serving.
    """
    filename = cfg.monitoring.reference_file
    try:
        if loaded.source == "mlflow":
            path = _download_mlflow_artifact(loaded, filename)
        else:
            if loaded.run_dir is None:
                raise DriftError("local model has no run directory")
            path = loaded.run_dir / filename
        reference = load_reference(path)
    except DriftError as error:
        logger.warning("drift reference unavailable", extra={"reason": str(error)})
        return None
    logger.info("drift reference loaded", extra={"rows": reference.rows, "path": str(path)})
    return reference


def _download_mlflow_artifact(loaded: LoadedModel, filename: str) -> Path:
    try:
        import mlflow  # noqa: PLC0415
    except ImportError as error:
        raise DriftError("mlflow is not installed") from error
    if not loaded.run_id:
        raise DriftError("registered model version has no run id")
    try:
        local = mlflow.artifacts.download_artifacts(
            run_id=loaded.run_id, artifact_path=filename, dst_path=tempfile.mkdtemp()
        )
    except Exception as error:  # mlflow raises MlflowException on missing artifacts
        raise DriftError(
            f"could not download '{filename}' from run {loaded.run_id}: {error}"
        ) from error
    return Path(local)


def load_model(cfg: Settings) -> LoadedModel:
    """Dispatch on ``api.model_source``."""
    if cfg.api.model_source == "mlflow":
        loaded = load_from_mlflow(cfg)
    else:
        loaded = load_local(resolve(cfg.api.local_run_dir), cfg.tracking.registered_model_name)
    logger.info(
        "model loaded",
        extra={
            "model": loaded.name,
            "version": loaded.version,
            "stage": loaded.stage,
            "source": loaded.source,
        },
    )
    return loaded


class ModelCache:
    """Process-wide holder: loads on first access, then reuses the object."""

    def __init__(self) -> None:
        self._loaded: LoadedModel | None = None

    def get(self, cfg: Settings) -> LoadedModel:
        """Return the cached model, loading it on first call."""
        if self._loaded is None:
            self._loaded = load_model(cfg)
        return self._loaded

    def set(self, loaded: LoadedModel) -> None:
        """Inject an already loaded model (tests, warm start)."""
        self._loaded = loaded

    def reset(self) -> None:
        """Drop the cached model so the next access reloads it."""
        self._loaded = None

    def peek(self) -> LoadedModel | None:
        """Return the cached model without triggering a load."""
        return self._loaded

    @property
    def is_loaded(self) -> bool:
        """Whether a model is currently cached."""
        return self._loaded is not None


MODEL_CACHE = ModelCache()
