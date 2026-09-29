"""HTTP contract of the five read-only ``/model`` endpoints.

They feed the model page of the interface, which is a pure HTTP client: if one of these
answers changes shape, that page silently stops drawing. Two situations are covered, and
both must answer 200: a run directory with every artifact, and a model served without
one, where the payload carries ``available=False`` and the reason.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from src.api.model_loader import ModelCache
from src.config import Settings
from src.data.filters import split_holdout
from src.data.mapping import VariableMapping
from tests.api_fixtures import loaded_from, train_fixture_pipeline

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient

from src.api.main import create_app

pytestmark = pytest.mark.contract

SECTIONS = ("summary", "roc", "importance", "contributions", "subgroups")

_METRICS = {
    "cv": {"summary": {"auc_roc_mean": 0.75, "auc_roc_std": 0.02}, "folds": [{}] * 10},
    "holdout": {
        "auc_roc": 0.74,
        "accuracy": 0.7,
        "precision": 0.32,
        "sensitivity": 0.68,
        "f1": 0.43,
        "true_negatives": 7,
        "false_positives": 3,
        "false_negatives": 1,
        "true_positives": 2,
        "n": 13,
    },
}
_ROC = {"holdout": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.9, 1.0], "thresholds": [1.9, 0.5, 0.0]}}
_INTERPRETATION = {
    "method": "coef_j * x_ij",
    "coefficients": [
        {
            "dummy": "sex_2",
            "model_variable": "sex",
            "label_pt": "Sexo",
            "category": "2",
            "coefficient": -0.7,
            "mean_abs_contribution": 0.4,
        }
    ],
    "importance": [
        {
            "model_variable": "sex",
            "label_pt": "Sexo",
            "source_raw": ["sex"],
            "mean_abs_contribution": 0.4,
            "max_abs_coef": 0.7,
        }
    ],
    "contributions": {
        "dummies": ["sex_2"],
        "labels_pt": ["Sexo = 2"],
        "contributions": [[-0.7], [0.0]],
        "present": [[1], [0]],
        "rows": 2,
        "rows_available": 13,
    },
}
_SUBGROUPS = [
    {
        "dimension": "sex",
        "group": "feminino",
        "n": 10,
        "prevalence": 0.2,
        "auc_roc": 0.7,
        "sensitivity": 0.6,
        "specificity": 0.8,
        "precision": 0.3,
        "f1": 0.4,
    }
]


@pytest.fixture(scope="module")
def artifacts(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A run directory with every artifact the report reads."""
    directory = tmp_path_factory.mktemp("runs") / "lr_production"
    directory.mkdir()
    (directory / "metrics.json").write_text(json.dumps(_METRICS), encoding="utf-8")
    (directory / "roc.json").write_text(json.dumps(_ROC), encoding="utf-8")
    (directory / "interpretation.json").write_text(json.dumps(_INTERPRETATION), encoding="utf-8")
    (directory / "subgroup_metrics.json").write_text(json.dumps(_SUBGROUPS), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def pipeline(cfg: Settings, mapping: VariableMapping, synthetic_dev: pd.DataFrame):  # type: ignore[no-untyped-def]
    train, _ = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    return train_fixture_pipeline(cfg, mapping, train)


@pytest.fixture(scope="module")
def client(cfg: Settings, pipeline, artifacts: Path) -> TestClient:  # type: ignore[no-untyped-def]
    loaded = replace(loaded_from(pipeline), run_dir=artifacts)
    app = create_app(cfg, model=loaded, cache=ModelCache())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def client_without_artifacts(cfg: Settings, pipeline) -> TestClient:  # type: ignore[no-untyped-def]
    app = create_app(cfg, model=loaded_from(pipeline), cache=ModelCache())
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_answers_200_with_artifacts(client: TestClient, section: str) -> None:
    response = client.get(f"/model/{section}")
    assert response.status_code == 200
    assert response.json()["available"] is True


@pytest.mark.parametrize("section", SECTIONS)
def test_every_section_answers_200_without_artifacts(
    client_without_artifacts: TestClient, section: str
) -> None:
    response = client_without_artifacts.get(f"/model/{section}")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["reason"]


def test_summary_carries_the_model_identity(client: TestClient, cfg: Settings) -> None:
    body = client.get("/model/summary").json()
    assert body["model_run"] == "lr_production"
    assert body["model_source"] == cfg.api.model_source
    assert body["decision_threshold"] == cfg.model.decision_threshold
    assert body["n_folds"] == 10
    assert body["holdout"]["n"] == 13


def test_roc_marks_the_champion_and_carries_the_sweep(client: TestClient) -> None:
    body = client.get("/model/roc").json()
    assert body["curves"][0]["run"] == "lr_production"
    assert body["curves"][0]["is_champion"] is True
    assert len(body["curves"][0]["fpr"]) == len(body["curves"][0]["tpr"])
    assert body["threshold_available"] is True
    assert {"threshold", "sensitivity", "specificity"} <= set(body["threshold_points"][0])


def test_importance_and_contributions_keep_their_shape(client: TestClient) -> None:
    importance = client.get("/model/importance").json()
    assert importance["variables"][0]["model_variable"] == "sex"
    assert importance["coefficients"][0]["dummy"] == "sex_2"

    contributions = client.get("/model/contributions").json()
    assert contributions["dummies"] == ["sex_2"]
    assert contributions["contributions"] == [[-0.7], [0.0]]
    assert contributions["present"] == [[1], [0]]
    assert contributions["rows_available"] == 13


def test_subgroups_allow_null_metrics(client: TestClient) -> None:
    body = client.get("/model/subgroups").json()
    assert body["rows"][0]["group"] == "feminino"
    assert body["rows"][0]["auc_roc"] == 0.7


def test_endpoints_are_in_the_openapi_document(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    for section in SECTIONS:
        assert f"/model/{section}" in paths


# ------------------------------------------------------------------ authentication
@pytest.fixture(scope="module")
def protected_client(cfg: Settings, pipeline, artifacts: Path) -> TestClient:  # type: ignore[no-untyped-def]
    """App with HTN_API__AUTH_TOKEN set, built without touching the environment."""
    api = cfg.api.model_copy(update={"auth_token": SecretStr("s3gr3d0")})
    settings = cfg.model_copy(update={"api": api})
    loaded = replace(loaded_from(pipeline), run_dir=artifacts)
    app = create_app(settings, model=loaded, cache=ModelCache())
    with TestClient(app) as test_client:
        yield test_client


@pytest.mark.parametrize("path", ["/predict", "/predict/batch"])
def test_prediction_endpoints_require_the_token(protected_client: TestClient, path: str) -> None:
    response = protected_client.post(path, json={})
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.parametrize("section", SECTIONS)
def test_model_endpoints_require_the_token(protected_client: TestClient, section: str) -> None:
    assert protected_client.get(f"/model/{section}").status_code == 401


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_probes_stay_open_so_prometheus_keeps_scraping(
    protected_client: TestClient, path: str
) -> None:
    assert protected_client.get(path).status_code == 200


def test_correct_token_is_accepted(protected_client: TestClient) -> None:
    response = protected_client.get("/model/summary", headers={"Authorization": "Bearer s3gr3d0"})
    assert response.status_code == 200
    assert response.json()["available"] is True


@pytest.mark.parametrize(
    "header",
    ["Bearer errado", "Basic s3gr3d0", "s3gr3d0", "Bearer ", "Bearer s3gr3d", "Bearer s3gr3d0x"],
)
def test_wrong_credential_is_rejected(protected_client: TestClient, header: str) -> None:
    assert (
        protected_client.get("/model/summary", headers={"Authorization": header}).status_code == 401
    )


def test_open_api_needs_no_credential(client: TestClient) -> None:
    """Default of this study: without the variable set, nothing is required."""
    assert client.get("/model/summary").status_code == 200
