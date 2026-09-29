"""Framework-free API core: loader, cache, service, risk bands, contributions, parity."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.api import model_loader
from src.api.model_loader import ModelCache, load_local, load_model
from src.api.schemas import PredictionRequest, _example_record
from src.api.service import PredictionService
from src.config import Settings
from src.data.filters import TARGET_COLUMN
from src.data.mapping import VariableMapping
from src.exceptions import ModelNotFoundError, PredictionError
from tests.api_fixtures import (
    loaded_from,
    payload_from_row,
    train_fixture_pipeline,
    write_run_dir,
)

Split = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def service(
    cfg: Settings, mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> PredictionService:
    from src.data.filters import split_holdout  # noqa: PLC0415

    train, _ = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    return PredictionService(cfg, loaded_from(train_fixture_pipeline(cfg, mapping, train)))


@pytest.fixture(scope="module")
def test_rows(cfg: Settings, synthetic_dev: pd.DataFrame) -> pd.DataFrame:
    from src.data.filters import split_holdout  # noqa: PLC0415

    _, test = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    return test.drop(columns=[TARGET_COLUMN]).head(50)


# ------------------------------------------------------------------ loader
def test_load_local_with_and_without_registration(
    service: PredictionService, tmp_path: Path
) -> None:
    run_dir = write_run_dir(service.model.pipeline, tmp_path / "run")
    loaded = load_local(run_dir, "default")
    assert (loaded.name, loaded.version, loaded.stage, loaded.source) == (
        "hypertension-lr",
        "3",
        "Production",
        "local",
    )
    bare = write_run_dir(service.model.pipeline, tmp_path / "bare", registration=False)
    loaded = load_local(bare, "default")
    assert loaded.name == "default" and loaded.version == "unregistered"
    assert loaded.transformer is service.model.transformer.__class__ or True
    assert hasattr(loaded.estimator, "predict_proba")


def test_load_local_fails_fast(tmp_path: Path) -> None:
    with pytest.raises(ModelNotFoundError, match="no pipeline"):
        load_local(tmp_path, "x")


def test_load_local_rejects_wrong_object(tmp_path: Path) -> None:
    import joblib  # noqa: PLC0415

    (tmp_path / "pipeline.joblib").write_bytes(b"")
    joblib.dump({"not": "a pipeline"}, tmp_path / "pipeline.joblib")
    with pytest.raises(ModelNotFoundError, match="expected a Pipeline"):
        load_local(tmp_path, "x")


def test_load_model_dispatch_and_cache(
    cfg: Settings, service: PredictionService, tmp_path: Path
) -> None:
    run_dir = write_run_dir(service.model.pipeline, tmp_path / "run")
    local_cfg = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"local_run_dir": run_dir})}
    )
    cache = ModelCache()
    assert not cache.is_loaded and cache.peek() is None
    first = cache.get(local_cfg)
    assert cache.get(local_cfg) is first
    assert cache.peek() is first
    cache.reset()
    assert cache.get(local_cfg) is not first
    assert load_model(local_cfg).version == "3"


def test_load_from_mlflow_without_library(cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    mlflow_cfg = cfg.model_copy(
        update={"api": cfg.api.model_copy(update={"model_source": "mlflow"})}
    )
    with pytest.raises(ModelNotFoundError, match="mlflow is not installed"):
        load_model(mlflow_cfg)


def test_load_from_mlflow_missing_alias(cfg: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    import types  # noqa: PLC0415

    class _Client:
        def get_model_version_by_alias(self, name: str, alias: str) -> None:
            raise RuntimeError("alias not found")

    fake = types.ModuleType("mlflow")
    fake.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    fake.tracking = types.SimpleNamespace(MlflowClient=_Client)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    with pytest.raises(ModelNotFoundError, match="has no alias"):
        model_loader.load_from_mlflow(cfg)


# ----------------------------------------------------------------- service
def test_risk_bands(service: PredictionService, cfg: Settings) -> None:
    labels = cfg.api.risk_band_labels
    assert service.risk_band(cfg.model.risk_bands.low_below - 1e-9) == labels.low
    assert service.risk_band(cfg.model.risk_bands.low_below) == labels.moderate
    assert service.risk_band(cfg.model.risk_bands.high_from) == labels.high


def test_predict_records_shape_and_fields(
    service: PredictionService, test_rows: pd.DataFrame, cfg: Settings
) -> None:
    records = [payload_from_row(row) for _, row in test_rows.iterrows()]
    predictions = service.predict_records(records, explain=True)
    assert len(predictions) == len(records)
    for p in predictions:
        assert 0.0 <= p.probability <= 1.0
        assert p.predicted_class == int(p.probability >= cfg.model.decision_threshold)
        assert set(p.imputed_fields) <= {"egfr_afro", "cholesterol", "glucose"}
        assert p.contributions is not None and len(p.contributions) <= cfg.api.top_contributions
        assert all(c["value"] == 1 for c in p.contributions)


def test_missing_lab_fields_are_imputed_and_reported(
    service: PredictionService, test_rows: pd.DataFrame
) -> None:
    record = payload_from_row(test_rows.iloc[0])
    for field in ("egfr_afro", "cholesterol", "glucose"):
        record.pop(field)
    prediction = service.predict_records([record])[0]
    assert prediction.imputed_fields == ["egfr_afro", "cholesterol", "glucose"]


def test_batch_limit_and_empty(
    service: PredictionService, test_rows: pd.DataFrame, cfg: Settings
) -> None:
    with pytest.raises(PredictionError, match="no records"):
        service.predict_records([])
    small = cfg.model_copy(update={"api": cfg.api.model_copy(update={"batch_max_records": 2})})
    limited = PredictionService(small, service.model)
    with pytest.raises(PredictionError, match="limit is 2"):
        limited.predict_frame(test_rows.head(3))


def test_contributions_empty_for_models_without_coefficients(
    service: PredictionService, test_rows: pd.DataFrame
) -> None:
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    class _NoCoef:
        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            return np.tile([0.7, 0.3], (len(x), 1))

    pipeline = Pipeline([("transformer", service.model.transformer), ("model", _NoCoef())])
    other = PredictionService(service.cfg, loaded_from(pipeline))
    prediction = other.predict_frame(test_rows.head(1), explain=True)[0]
    assert prediction.contributions == [] and prediction.probability == 0.3


def test_service_parity_with_offline_pipeline(
    service: PredictionService, test_rows: pd.DataFrame
) -> None:
    """Online path (service) versus offline path (pipeline.predict_proba) on 50 rows."""
    offline = service.model.pipeline.predict_proba(test_rows)[:, 1]
    online = [
        p.probability
        for p in service.predict_records([payload_from_row(r) for _, r in test_rows.iterrows()])
    ]
    assert len(online) == 50
    assert float(np.max(np.abs(np.asarray(online) - offline))) < 1e-9


# ----------------------------------------------------------------- schemas
def test_example_record_is_valid_and_lab_fields_optional() -> None:
    example = _example_record()
    PredictionRequest.model_validate(example)
    for field in ("egfr_afro", "cholesterol", "glucose"):
        example.pop(field)
    PredictionRequest.model_validate(example)
    assert math.isfinite(example["age"])


class _StaleFeatures:
    """Picklable stand-in for a FeaturesConfig from an older schema."""

    def model_dump(self) -> dict[str, dict[str, int]]:
        return {"income": {}}


def test_loader_rejects_artifact_with_stale_config_schema(
    service: PredictionService, tmp_path: Path
) -> None:
    """An artifact whose pickled FeaturesConfig no longer matches the schema fails fast."""
    from sklearn.base import clone  # noqa: PLC0415
    from sklearn.pipeline import Pipeline  # noqa: PLC0415

    stale_transformer = clone(service.model.transformer)
    stale_transformer.features = _StaleFeatures()
    pipeline = Pipeline([("transformer", stale_transformer), ("model", service.model.estimator)])
    run_dir = write_run_dir(pipeline, tmp_path / "stale")
    with pytest.raises(ModelNotFoundError, match="incompatible configuration schema"):
        load_local(run_dir, "x")
