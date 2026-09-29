"""Population filters, target rule, split, quality gate and ingestion checks."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.data.filters import (
    TARGET_COLUMN,
    apply_population_filters,
    check_filter_report,
    select_declared_columns,
    split_holdout,
)
from src.data.ingest import check_dimensions, sha256_of_file, verify_sha256, write_interim
from src.data.mapping import VariableMapping
from src.data.quality import check_quality, enforce_quality
from src.data.synthetic import make_synthetic_frame
from src.exceptions import DataContractError


def test_filters_remove_exactly_the_declared_codes(
    synthetic_raw: pd.DataFrame, cfg: Settings, mapping: VariableMapping
) -> None:
    selected = select_declared_columns(synthetic_raw, mapping)
    filtered, report = apply_population_filters(selected, cfg.population_filters, cfg.target)
    pf = cfg.population_filters
    expected = [
        int((selected["Z051"] != pf.consent_keep_value).sum()),
    ]
    assert report.steps[0].removed == expected[0]
    assert report.steps[0].name == "lab_consent"
    assert [s.name for s in report.steps] == [
        "lab_consent",
        "pregnant",
        "hypertension_pregnancy_only",
        "medication",
        "region_missing",
        "empty_questionnaire",
    ]
    assert report.rows_out == len(filtered)
    assert filtered.index.tolist() == list(range(len(filtered)))
    assert (filtered["Z051"] == 1).all()
    assert (~filtered["P005"].isin([1, 3])).all()
    assert (filtered["Q002"] != 2).all()
    assert (filtered["Q006"] != 1).all()


def test_target_rule_and_blood_pressure_exclusion(
    synthetic_raw: pd.DataFrame, cfg: Settings, mapping: VariableMapping
) -> None:
    selected = select_declared_columns(synthetic_raw, mapping)
    selected = selected[
        (selected["Z051"] == 1)
        & (~selected["P005"].isin([1, 3]))
        & (selected["Q002"] != 2)
        & (selected["Q006"] != 1)
    ].copy()
    selected.loc[selected.index[:3], "W00407"] = math.nan  # D1: three rows without SBP
    filtered, report = apply_population_filters(selected, cfg.population_filters, cfg.target)
    assert report.blood_pressure_missing_removed == 3
    expected = ((filtered["W00407"] >= 140) | (filtered["W00408"] >= 90)).astype(int)
    assert filtered[TARGET_COLUMN].tolist() == expected.tolist()
    assert filtered[TARGET_COLUMN].isin([0, 1]).all()


def test_check_filter_report_flags_divergence(
    synthetic_raw: pd.DataFrame, cfg: Settings, mapping: VariableMapping
) -> None:
    selected = select_declared_columns(synthetic_raw, mapping)
    _, report = apply_population_filters(selected, cfg.population_filters, cfg.target)
    with pytest.raises(DataContractError, match="differ"):
        check_filter_report(report, cfg.population_filters, cfg.data.expected_rows_after_filters)


def test_select_declared_columns_requires_all(
    synthetic_raw: pd.DataFrame, mapping: VariableMapping
) -> None:
    with pytest.raises(DataContractError, match="Z002"):
        select_declared_columns(synthetic_raw.drop(columns=["Z002"]), mapping)


def test_split_is_stratified_and_deterministic(synthetic_dev: pd.DataFrame, cfg: Settings) -> None:
    train_a, test_a = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    train_b, test_b = split_holdout(synthetic_dev, cfg.split, cfg.project.seed)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)
    assert len(test_a) == math.ceil(len(synthetic_dev) * cfg.split.test_size)  # sklearn rounds up
    assert abs(train_a[TARGET_COLUMN].mean() - test_a[TARGET_COLUMN].mean()) < 0.05
    assert not set(train_a.index) & set(test_a.index)


def test_synthetic_frame_is_deterministic(cfg: Settings) -> None:
    pd.testing.assert_frame_equal(
        make_synthetic_frame(cfg.project.seed), make_synthetic_frame(cfg.project.seed)
    )
    assert not make_synthetic_frame(1, rows=50).equals(make_synthetic_frame(2, rows=50))


# ------------------------------------------------------------ quality gate
def test_quality_gate_passes_on_fixture(
    synthetic_dev: pd.DataFrame, mapping: VariableMapping
) -> None:
    results = enforce_quality(synthetic_dev, mapping)
    assert len(results) == 61 and all(r.passed for r in results)


def test_quality_gate_fails_on_excess_nulls(
    synthetic_dev: pd.DataFrame, mapping: VariableMapping
) -> None:
    broken = synthetic_dev.copy()
    broken.loc[broken.index[: len(broken) // 2], "sex"] = math.nan
    with pytest.raises(DataContractError, match="sex: null_share"):
        enforce_quality(broken, mapping)


def test_quality_gate_fails_on_out_of_domain(
    synthetic_dev: pd.DataFrame, mapping: VariableMapping
) -> None:
    broken = synthetic_dev.copy()
    broken.loc[broken.index[0], "region"] = 9
    failed = [r for r in check_quality(broken, mapping) if not r.passed]
    assert [r.column for r in failed] == ["region"]
    assert failed[0].out_of_domain == 1


# ---------------------------------------------------------------- ingest
def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    file = tmp_path / "x.xlsx"
    file.write_bytes(b"not the frozen dataset")
    digest = sha256_of_file(file)
    assert verify_sha256(file, digest) == digest
    with pytest.raises(DataContractError, match="SHA-256"):
        verify_sha256(file, "0" * 64)
    with pytest.raises(DataContractError, match="not found"):
        verify_sha256(tmp_path / "missing.xlsx", digest)


def test_check_dimensions(cfg: Settings) -> None:
    frame = pd.DataFrame(np.zeros((cfg.data.expected_raw_rows, cfg.data.expected_raw_columns)))
    check_dimensions(frame, cfg.data)
    with pytest.raises(DataContractError, match="shape"):
        check_dimensions(frame.iloc[:10], cfg.data)


def test_write_interim_parquet(tmp_path: Path, synthetic_raw: pd.DataFrame) -> None:
    pytest.importorskip("pyarrow")
    path = write_interim(synthetic_raw.head(20), tmp_path / "interim", "raw_test")
    assert path.name == "raw_test.parquet"
    pd.testing.assert_frame_equal(pd.read_parquet(path), synthetic_raw.head(20))
