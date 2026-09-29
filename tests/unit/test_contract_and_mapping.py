"""Variable mapping integrity and the generated data contracts."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from pydantic import ValidationError

from src.data.contract import ModelRecord, RawRecord, enforce_contract, validate_frame
from src.data.mapping import VariableMapping
from src.exceptions import DataContractError


def test_mapping_counts(mapping: VariableMapping) -> None:
    assert len(mapping.raw_variables) == 61
    assert len(mapping.model_variables) == 33
    assert len(mapping.dummy_names) == 85
    assert len(mapping.auxiliary_variables) == 6


def test_mapping_names_are_unique(mapping: VariableMapping) -> None:
    for attribute in ("pns", "pt", "en"):
        values = [getattr(v, attribute) for v in mapping.raw_variables]
        assert len(values) == len(set(values)), attribute
    assert len(set(mapping.dummy_names)) == 85
    notebook = list(mapping.dummy_en_to_notebook.values())
    assert len(set(notebook)) == 85


def test_model_sources_are_declared_raw_variables(mapping: VariableMapping) -> None:
    raw = set(mapping.raw_en_names)
    for variable in mapping.model_variables:
        assert set(variable.source_raw) <= raw, variable.en


def test_raw_record_accepts_fixture_rows(synthetic_dev: pd.DataFrame) -> None:
    frame = synthetic_dev.drop(columns=["target"]).head(50)
    assert validate_frame(frame, RawRecord) == []


def test_raw_record_rejects_out_of_domain_and_reports_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        RawRecord.model_validate({"sex": 3})
    fields = {e["loc"][0] for e in excinfo.value.errors()}
    assert "sex" in fields
    assert "age" in fields  # required, missing


def test_raw_record_allows_none_only_where_nullable(mapping: VariableMapping) -> None:
    nullable = mapping.raw("glucose")
    strict = mapping.raw("sex")
    assert nullable.nullable and not strict.nullable
    assert RawRecord.model_fields["glucose"].default is None
    assert RawRecord.model_fields["sex"].is_required()


def test_enforce_contract_reports_row_and_column(synthetic_dev: pd.DataFrame) -> None:
    frame = synthetic_dev.drop(columns=["target"]).head(5).copy()
    frame.loc[frame.index[2], "region"] = 7
    with pytest.raises(DataContractError, match="row 2") as excinfo:
        enforce_contract(frame, RawRecord, "raw")
    assert "region" in str(excinfo.value)


def test_enforce_contract_detects_column_drift(synthetic_dev: pd.DataFrame) -> None:
    frame = synthetic_dev.drop(columns=["target", "sex"]).head(2)
    with pytest.raises(DataContractError, match="missing=\\['sex'\\]"):
        enforce_contract(frame, RawRecord, "raw")


def test_model_record_only_accepts_declared_categories() -> None:
    payload = {name: "1" for name in ModelRecord.model_fields}
    payload["age_group"] = "18-29"
    ModelRecord.model_validate(payload)
    payload["bmi_class"] = "7"
    with pytest.raises(ValidationError):
        ModelRecord.model_validate(payload)


def test_nan_is_treated_as_missing_in_frames(synthetic_dev: pd.DataFrame) -> None:
    frame = synthetic_dev.drop(columns=["target"]).head(1).copy()
    frame["sex"] = math.nan
    failures = validate_frame(frame, RawRecord)
    assert failures and failures[0]["errors"][0]["field"] == "sex"
