"""Data contracts generated from ``configs/variable_mapping.yaml``.

Two Pydantic models are built at import time from the mapping, so a change in
the YAML changes the contract without touching code:

* ``RawRecord``: one observation of the 61 raw inference variables in English
  names, with categorical codes restricted to their declared values, numeric
  values restricted to their declared range and ``None`` allowed only where
  the variable is nullable.
* ``ModelRecord``: one observation of the 33 model variables after feature
  derivation, each restricted to its explicit category list.

Frame-level helpers validate every row and report the offending rows and
fields instead of stopping at the first error.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from src.data.mapping import RawVariable, VariableMapping, get_mapping
from src.exceptions import DataContractError


class _StrictRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


def _raw_field(variable: RawVariable) -> tuple[Any, Any]:
    domain = variable.domain
    annotation: Any
    constraints: dict[str, Any] = {"description": variable.label_pt}
    if domain.type == "categorical":
        annotation = Literal[tuple(domain.values or [])]
    else:
        annotation = int if variable.dtype == "int" else float
        constraints.update(ge=domain.min, le=domain.max)
    if variable.nullable:
        # Optional with an explicit None default: the field may be omitted or null.
        return annotation | None, Field(default=None, **constraints)
    return annotation, Field(**constraints)


def build_raw_record_model(mapping: VariableMapping) -> type[BaseModel]:
    """Create the ``RawRecord`` model from the mapping."""
    fields: dict[str, Any] = {v.en: _raw_field(v) for v in mapping.raw_variables}
    model: type[BaseModel] = create_model("RawRecord", __base__=_StrictRecord, **fields)
    return model


def build_model_record_model(mapping: VariableMapping) -> type[BaseModel]:
    """Create the ``ModelRecord`` model (33 categorical variables) from the mapping."""
    fields: dict[str, Any] = {
        v.en: (Literal[tuple(v.categories)], Field(description=v.label_pt))
        for v in mapping.model_variables
    }
    model: type[BaseModel] = create_model("ModelRecord", __base__=_StrictRecord, **fields)
    return model


_MAPPING = get_mapping()
RawRecord: type[BaseModel] = build_raw_record_model(_MAPPING)
ModelRecord: type[BaseModel] = build_model_record_model(_MAPPING)


def _row_to_payload(row: pd.Series) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            payload[str(key)] = None
        elif isinstance(value, float) and value.is_integer():
            payload[str(key)] = int(value)
        else:
            payload[str(key)] = value
    return payload


def validate_frame(
    frame: pd.DataFrame, model: type[BaseModel], *, max_errors: int = 20
) -> list[dict[str, Any]]:
    """Validate every row of ``frame`` against ``model``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Rows to validate; columns must match the model fields exactly.
    model : type[BaseModel]
        ``RawRecord`` or ``ModelRecord``.
    max_errors : int
        Stop collecting after this many failing rows.

    Returns
    -------
    list of dict
        One entry per failing row: ``{"row": index, "errors": [{"field", "message"}]}``.
        Empty when the frame is valid.
    """
    failures: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        try:
            model.model_validate(_row_to_payload(row))
        except ValidationError as error:
            failures.append(
                {
                    "row": index,
                    "errors": [
                        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
                        for e in error.errors()
                    ],
                }
            )
            if len(failures) >= max_errors:
                break
    return failures


def enforce_contract(frame: pd.DataFrame, model: type[BaseModel], name: str) -> None:
    """Raise ``DataContractError`` describing the first failing rows, if any."""
    expected = list(model.model_fields)
    extra = [c for c in frame.columns if c not in expected]
    missing = [c for c in expected if c not in frame.columns]
    if extra or missing:
        raise DataContractError(
            f"{name}: column set differs from contract (missing={missing}, extra={extra})"
        )
    failures = validate_frame(frame, model)
    if failures:
        preview = "; ".join(
            f"row {f['row']}: " + ", ".join(f"{e['field']} {e['message']}" for e in f["errors"])
            for f in failures[:5]
        )
        raise DataContractError(
            f"{name}: {len(failures)} row(s) violate the contract (showing up to 5): {preview}"
        )
