"""Stage 1 orchestration: raw workbook (or fixture) to the development frame.

``load_development_frame`` is the single entry point used by the quality
gate, the training stage and the tests. It returns the frame with the 61 raw
inference variables in English names plus ``target``, after the population
filters, and asserts the frozen counts from the config.
"""

from __future__ import annotations

import pandas as pd

from src.config import Settings
from src.data.filters import (
    FilterReport,
    apply_population_filters,
    check_filter_report,
    select_declared_columns,
    to_english_inference_frame,
)
from src.data.ingest import ingest, read_raw_excel, verify_sha256
from src.data.mapping import get_mapping
from src.data.synthetic import make_synthetic_frame
from src.logging_setup import get_logger

logger = get_logger(__name__)


def load_raw_frame(
    cfg: Settings, *, use_fixture: bool = False, materialize: bool = True
) -> pd.DataFrame:
    """Return the raw frame with PNS column codes.

    Parameters
    ----------
    cfg : Settings
        Project configuration.
    use_fixture : bool
        Generate the synthetic fixture instead of reading the real workbook.
    materialize : bool
        Write the Parquet copy under ``data/interim`` (real data only).
    """
    if use_fixture:
        return make_synthetic_frame(seed=cfg.project.seed)
    if materialize:
        report = ingest(cfg.paths, cfg.data)
        return pd.read_parquet(report.interim_path)
    source = cfg.paths.absolute("raw_dataset")
    verify_sha256(source, cfg.data.raw_sha256)
    return read_raw_excel(source, cfg.data.raw_sheet)


def build_development_frame(
    raw: pd.DataFrame, cfg: Settings, *, enforce_counts: bool
) -> tuple[pd.DataFrame, FilterReport]:
    """Apply filters, target and renaming to a raw frame.

    Parameters
    ----------
    raw : pandas.DataFrame
        Frame with PNS codes as columns.
    cfg : Settings
        Project configuration.
    enforce_counts : bool
        Compare the filter counts with ``population_filters.expected_removed``
        (only meaningful on the real dataset).
    """
    mapping = get_mapping()
    selected = select_declared_columns(raw, mapping)
    filtered, report = apply_population_filters(selected, cfg.population_filters, cfg.target)
    if enforce_counts:
        check_filter_report(report, cfg.population_filters, cfg.data.expected_rows_after_filters)
    return to_english_inference_frame(filtered, mapping), report


def load_development_frame(
    cfg: Settings, *, use_fixture: bool = False, materialize: bool = True
) -> pd.DataFrame:
    """Raw source to development frame in one call."""
    raw = load_raw_frame(cfg, use_fixture=use_fixture, materialize=materialize)
    frame, _ = build_development_frame(raw, cfg, enforce_counts=not use_fixture)
    return frame
