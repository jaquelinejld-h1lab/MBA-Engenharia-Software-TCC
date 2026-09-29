"""Remaining branches of the data layer: ingestion, pipeline, quality CLI, mapping, contract."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config import Settings
from src.data import pipeline as pipeline_module
from src.data import quality as quality_module
from src.data.contract import RawRecord, validate_frame
from src.data.filters import FilterReport, FilterStep, check_filter_report
from src.data.ingest import ingest, read_raw_excel, sha256_of_file
from src.data.mapping import VariableMapping, load_mapping
from src.data.pipeline import load_raw_frame
from src.exceptions import ConfigError, DataContractError
from src.logging_setup import JsonFormatter
from src.paths import PROJECT_ROOT, resolve


# ------------------------------------------------------------------ ingest
def _small_workbook(frame: pd.DataFrame, path: Path, sheet: str) -> Path:
    frame.rename(columns=str.lower).to_excel(path, sheet_name=sheet, index=False)
    return path


def test_read_raw_excel_uppercases_columns(
    synthetic_raw: pd.DataFrame, tmp_path: Path, cfg: Settings
) -> None:
    path = _small_workbook(synthetic_raw.head(5), tmp_path / "raw.xlsx", cfg.data.raw_sheet)
    frame = read_raw_excel(path, cfg.data.raw_sheet)
    assert list(frame.columns) == list(synthetic_raw.columns)
    assert len(frame) == 5


def test_ingest_end_to_end_on_small_workbook(
    synthetic_raw: pd.DataFrame, tmp_path: Path, cfg: Settings
) -> None:
    pytest.importorskip("pyarrow")
    workbook = _small_workbook(synthetic_raw.head(5), tmp_path / "raw.xlsx", cfg.data.raw_sheet)
    data = cfg.data.model_copy(
        update={
            "raw_sha256": sha256_of_file(workbook),
            "expected_raw_rows": 5,
            "expected_raw_columns": synthetic_raw.shape[1],
        }
    )
    paths = cfg.paths.model_copy(
        update={"raw_dataset": workbook, "interim_dir": tmp_path / "interim"}
    )
    report = ingest(paths, data)
    assert report.rows == 5 and report.interim_path.is_file()
    assert report.sha256 == data.raw_sha256


def test_ingest_refuses_wrong_hash(
    synthetic_raw: pd.DataFrame, tmp_path: Path, cfg: Settings
) -> None:
    workbook = _small_workbook(synthetic_raw.head(2), tmp_path / "raw.xlsx", cfg.data.raw_sheet)
    paths = cfg.paths.model_copy(update={"raw_dataset": workbook, "interim_dir": tmp_path})
    with pytest.raises(DataContractError, match="SHA-256"):
        ingest(paths, cfg.data)


# ---------------------------------------------------------------- pipeline
def test_load_raw_frame_without_materialization(
    cfg: Settings, synthetic_raw: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workbook = _small_workbook(synthetic_raw.head(3), tmp_path / "raw.xlsx", cfg.data.raw_sheet)
    local = cfg.model_copy(
        update={
            "paths": cfg.paths.model_copy(update={"raw_dataset": workbook}),
            "data": cfg.data.model_copy(update={"raw_sha256": sha256_of_file(workbook)}),
        }
    )
    frame = load_raw_frame(local, materialize=False)
    assert len(frame) == 3 and "Z051" in frame.columns


def test_load_raw_frame_materialized_uses_ingest(
    cfg: Settings, synthetic_raw: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("pyarrow")
    interim = tmp_path / "raw.parquet"
    synthetic_raw.head(4).to_parquet(interim, index=False)

    class _Report:
        interim_path = interim

    monkeypatch.setattr(pipeline_module, "ingest", lambda paths, data: _Report())
    assert len(load_raw_frame(cfg, materialize=True)) == 4


# ------------------------------------------------------------ quality CLI
def test_quality_cli_success_and_report(
    cfg: Settings, synthetic_dev: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(pipeline_module, "load_development_frame", lambda *a, **k: synthetic_dev)
    report = tmp_path / "quality.json"
    assert quality_module._main(["--fixture", "--report", str(report)]) == 0
    payload = json.loads(report.read_text())
    assert len(payload) == 61 and all(row["passed"] for row in payload)


def test_quality_cli_failure_returns_one(
    cfg: Settings, synthetic_dev: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    broken = synthetic_dev.copy()
    broken["sex"] = 7
    monkeypatch.setattr(pipeline_module, "load_development_frame", lambda *a, **k: broken)
    assert quality_module._main(["--fixture"]) == 1


# ----------------------------------------------------------------- mapping
def test_mapping_lookup_errors(mapping: VariableMapping) -> None:
    with pytest.raises(ConfigError, match="raw variable 'nope'"):
        mapping.raw("nope")
    with pytest.raises(ConfigError, match="model variable 'nope'"):
        mapping.model_variable("nope")
    assert mapping.raw("age").pns == "Z002"
    assert mapping.model_variable("sex").reference_category == "1"


def test_mapping_file_validation(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_mapping(tmp_path / "missing.yaml")
    source = PROJECT_ROOT / "configs" / "variable_mapping.yaml"
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["model_variables"][0]["reference_category"] = "zzz"
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="reference category"):
        load_mapping(broken)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["raw_variables"][0]["domain"] = {"type": "categorical"}
    broken.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="values"):
        load_mapping(broken)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["model_variables"][1]["dummies"] = []
    broken.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ConfigError, match="dummies"):
        load_mapping(broken)


# ---------------------------------------------------------------- contract
def test_validate_frame_stops_at_max_errors(synthetic_dev: pd.DataFrame) -> None:
    frame = synthetic_dev.drop(columns=["target"]).head(10).copy()
    frame["sex"] = 9
    failures = validate_frame(frame, RawRecord, max_errors=3)
    assert len(failures) == 3


# ----------------------------------------------------------------- filters
def test_check_filter_report_rows_out_mismatch(cfg: Settings) -> None:
    steps = [
        FilterStep(name=str(i), removed=r, remaining=0)
        for i, r in enumerate(cfg.population_filters.expected_removed)
    ]
    report = FilterReport(rows_in=1, steps=steps, rows_out=1)
    with pytest.raises(DataContractError, match="rows after filters"):
        check_filter_report(report, cfg.population_filters, cfg.data.expected_rows_after_filters)


# ---------------------------------------------------------- misc helpers
def test_resolve_keeps_absolute_paths(tmp_path: Path) -> None:
    assert resolve(tmp_path) == tmp_path
    assert resolve("configs") == PROJECT_ROOT / "configs"


def test_json_formatter_includes_exception() -> None:
    formatter = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    payload = json.loads(formatter.format(record))
    assert "boom" in payload["exception"] and payload["level"] == "ERROR"
