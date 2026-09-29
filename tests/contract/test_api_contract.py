"""HTTP contract of the four endpoints (FastAPI TestClient) and online/offline parity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.api.model_loader import ModelCache
from src.config import Settings
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import VariableMapping
from src.exceptions import ModelNotFoundError
from tests.api_fixtures import loaded_from, payload_from_row, train_fixture_pipeline

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402

from src.api.main import create_app  # noqa: E402

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def split(cfg: Settings, synthetic_dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return split_holdout(synthetic_dev, cfg.split, cfg.project.seed)


@pytest.fixture(scope="module")
def loaded(cfg: Settings, mapping: VariableMapping, split: tuple[pd.DataFrame, pd.DataFrame]):  # type: ignore[no-untyped-def]
    return loaded_from(train_fixture_pipeline(cfg, mapping, split[0]))


@pytest.fixture(scope="module")
def client(cfg: Settings, loaded) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(cfg, model=loaded, cache=ModelCache())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def rows(split: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return split[1].drop(columns=[TARGET_COLUMN]).head(50)


def test_health(client: TestClient, cfg: Settings) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "test" and body["model_source"] == cfg.api.model_source


def test_predict_single(client: TestClient, rows: pd.DataFrame, cfg: Settings) -> None:
    response = client.post(
        "/predict?explain=true",
        json=payload_from_row(rows.iloc[0]),
        headers={"X-Correlation-ID": "abc-123"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_band"] in cfg.api.risk_band_labels.model_dump().values()
    assert body["threshold"] == cfg.model.decision_threshold
    assert body["model_version"] == "test" and body["disclaimer"]
    assert body["correlation_id"] == "abc-123"
    assert response.headers["X-Correlation-ID"] == "abc-123"
    assert isinstance(body["imputed_fields"], list)
    assert (
        body["contributions"] is not None
        and len(body["contributions"]) <= cfg.api.top_contributions
    )


def test_predict_reports_imputed_lab_fields(client: TestClient, rows: pd.DataFrame) -> None:
    payload = payload_from_row(rows.iloc[1])
    payload.pop("glucose")
    payload["cholesterol"] = None
    body = client.post("/predict", json=payload).json()
    assert set(body["imputed_fields"]) >= {"glucose", "cholesterol"}


def test_predict_validation_error_is_422(client: TestClient, rows: pd.DataFrame) -> None:
    payload = payload_from_row(rows.iloc[0])
    payload["sex"] = 3
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "traceback" not in response.text.lower()


def test_predict_batch(client: TestClient, rows: pd.DataFrame) -> None:
    payload = {"records": [payload_from_row(r) for _, r in rows.head(10).iterrows()]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n"] == 10 and len(body["predictions"]) == 10
    assert body["n_imputed_rows"] <= 10 and body["correlation_id"]


def test_metrics_exposition(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    for name in ("api_request_latency_seconds", "api_requests_total", "api_predictions_total"):
        assert name in text
    assert 'endpoint="/predict"' in text


def test_domain_error_envelope_without_stack_trace(
    client: TestClient, rows: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.api import service as service_module  # noqa: PLC0415

    def _boom(self, frame, *, explain=False):  # type: ignore[no-untyped-def]
        raise ModelNotFoundError("registry offline")

    monkeypatch.setattr(service_module.PredictionService, "predict_frame", _boom)
    response = client.post("/predict", json=payload_from_row(rows.iloc[0]))
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "ModelNotFoundError" and body["detail"] == "registry offline"
    assert body["correlation_id"] and "Traceback" not in response.text


def test_online_offline_parity_50_rows(client: TestClient, rows: pd.DataFrame, loaded) -> None:  # type: ignore[no-untyped-def]
    offline = loaded.pipeline.predict_proba(rows)[:, 1]
    online = []
    for _, row in rows.iterrows():
        response = client.post("/predict", json=payload_from_row(row))
        assert response.status_code == 200, response.text
        online.append(response.json()["probability"])
    assert len(online) == 50
    assert float(np.max(np.abs(np.asarray(online) - offline))) < 1e-9
