"""Helpers shared by API tests: a pipeline trained on the synthetic fixture."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src.api.model_loader import LoadedModel
from src.config import Settings
from src.data.filters import TARGET_COLUMN
from src.data.mapping import VariableMapping
from src.features.transformer import HypertensionTransformer
from src.models.algorithms import build_estimator


def train_fixture_pipeline(
    cfg: Settings, mapping: VariableMapping, train: pd.DataFrame
) -> Pipeline:
    """Fit the champion configuration (production transformer + LR) on ``train``."""
    features = train.drop(columns=[TARGET_COLUMN])
    transformer = HypertensionTransformer(cfg.features, mapping, mode="production").fit(features)
    estimator = build_estimator(
        cfg.model.champion.algorithm,
        cfg.model.champion.params.model_dump(),
        cfg.project.seed,
        train[TARGET_COLUMN],
    )
    estimator.fit(transformer.transform(features), train[TARGET_COLUMN])
    return Pipeline([("transformer", transformer), ("model", estimator)])


def loaded_from(pipeline: Pipeline, name: str = "hypertension-lr") -> LoadedModel:
    """Wrap a pipeline as a locally loaded model with test metadata."""
    return LoadedModel(
        pipeline=pipeline, name=name, version="test", stage="Staging", source="local"
    )


def write_run_dir(pipeline: Pipeline, directory: Path, *, registration: bool = True) -> Path:
    """Persist ``pipeline.joblib`` (and optionally ``registration.json``) like the trainer."""
    directory.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, directory / "pipeline.joblib")
    if registration:
        (directory / "registration.json").write_text(
            json.dumps({"name": "hypertension-lr", "version": "3", "stage": "Production"}),
            encoding="utf-8",
        )
    return directory


def payload_from_row(row: pd.Series) -> dict[str, float | int | None]:
    """Frame row to a JSON-ready request payload (NaN -> None, integral floats -> int)."""
    payload: dict[str, float | int | None] = {}
    for key, value in row.items():
        if pd.isna(value):
            payload[str(key)] = None
        elif float(value).is_integer():
            payload[str(key)] = int(value)
        else:
            payload[str(key)] = float(value)
    return payload
