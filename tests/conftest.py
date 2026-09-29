"""Shared fixtures: configuration, mapping and the synthetic development frame."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import Settings, load_config
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import VariableMapping, load_mapping
from src.data.pipeline import build_development_frame
from src.data.synthetic import make_synthetic_frame


@pytest.fixture(scope="session")
def cfg() -> Settings:
    return load_config()


@pytest.fixture(scope="session")
def mapping() -> VariableMapping:
    return load_mapping()


@pytest.fixture(scope="session")
def synthetic_raw(cfg: Settings) -> pd.DataFrame:
    """Raw synthetic frame with PNS codes (before filters)."""
    return make_synthetic_frame(seed=cfg.project.seed)


@pytest.fixture(scope="session")
def synthetic_dev(synthetic_raw: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    """Synthetic development frame: 61 raw variables in English plus target."""
    frame, _ = build_development_frame(synthetic_raw, cfg, enforce_counts=False)
    return frame


@pytest.fixture(scope="session")
def synthetic_dev_large(cfg: Settings) -> pd.DataFrame:
    """Larger synthetic development frame (drift tests need small-sample noise below thresholds)."""
    frame, _ = build_development_frame(
        make_synthetic_frame(seed=cfg.project.seed, rows=8000), cfg, enforce_counts=False
    )
    return frame


@pytest.fixture(scope="session")
def synthetic_split(
    synthetic_dev: pd.DataFrame, cfg: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    return train.drop(columns=[TARGET_COLUMN]), test.drop(columns=[TARGET_COLUMN])
