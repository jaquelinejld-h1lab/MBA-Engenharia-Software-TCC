"""Client-side validation and CSV helpers of the Streamlit app.

The API is the authority (its Pydantic contract rejects invalid payloads
with 422); this module reproduces the same domain rules from the mapping
so the user gets specific Portuguese messages before anything is sent,
and so a batch CSV is reported row by row instead of failing as a whole.
"""

from __future__ import annotations

import io
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.data.mapping import RawVariable, VariableMapping


@dataclass(frozen=True)
class FieldError:
    """One validation failure."""

    row: int | None
    variable: str
    message: str


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(float(value)))
    except (TypeError, ValueError):
        return False


def _as_number(value: Any) -> float | None:
    """Parse numbers (accepting decimal commas); ``None`` when not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    text = str(value).strip().replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def check_value(variable: RawVariable, value: Any) -> str | None:  # noqa: PLR0911
    """Portuguese message describing why ``value`` is invalid, or ``None``."""
    label = variable.label_pt
    if _is_missing(value):
        return None if variable.nullable else f"{label}: campo obrigatório."
    number = _as_number(value)
    if number is None:
        return f"{label}: valor '{value}' não é numérico."
    domain = variable.domain
    if domain.type == "categorical":
        codes = domain.values or []
        if number != int(number) or int(number) not in codes:
            return f"{label}: código {value} inválido; válidos: {', '.join(map(str, codes))}."
        return None
    minimum, maximum = float(domain.min or 0), float(domain.max or 0)
    if not minimum <= number <= maximum:
        unit = f" {variable.unit}" if variable.unit else ""
        return f"{label}: valor {number:g} fora da faixa {minimum:g} a {maximum:g}{unit}."
    if variable.dtype == "int" and number != int(number):
        return f"{label}: deve ser um número inteiro."
    return None


def validate_record(record: Mapping[str, Any], mapping: VariableMapping) -> list[FieldError]:
    """Validate one record (missing keys count as missing values)."""
    errors = [
        FieldError(None, variable.en, message)
        for variable in mapping.raw_variables
        if (message := check_value(variable, record.get(variable.en))) is not None
    ]
    unknown = sorted(set(record) - set(mapping.raw_en_names))
    errors.extend(FieldError(None, name, f"{name}: variável desconhecida.") for name in unknown)
    return errors


def clean_record(record: Mapping[str, Any], mapping: VariableMapping) -> dict[str, Any]:
    """JSON-ready payload: numbers typed by ``dtype``, missing values as ``None``."""
    payload: dict[str, Any] = {}
    for variable in mapping.raw_variables:
        value = record.get(variable.en)
        if _is_missing(value):
            payload[variable.en] = None
            continue
        number = _as_number(value)
        if number is None:
            payload[variable.en] = value  # left for the API to reject
        else:
            payload[variable.en] = int(number) if variable.dtype == "int" else number
    return payload


# --------------------------------------------------------------------------- CSV


@dataclass(frozen=True)
class CsvValidation:
    """Outcome of validating an uploaded CSV."""

    total_rows: int
    valid_rows: list[int]
    errors: list[FieldError]
    missing_columns: list[str] = field(default_factory=list)
    unknown_columns: list[str] = field(default_factory=list)

    @property
    def invalid_rows(self) -> list[int]:
        """Row numbers (1-based, header excluded) with at least one error."""
        return sorted({e.row for e in self.errors if e.row is not None})

    @property
    def ok(self) -> bool:
        """True when every row is valid and the columns are complete."""
        return not self.errors and not self.missing_columns

    def error_frame(self) -> pd.DataFrame:
        """Row-by-row report for display and download."""
        return pd.DataFrame(
            [{"linha": e.row, "variavel": e.variable, "erro": e.message} for e in self.errors],
            columns=["linha", "variavel", "erro"],
        )


def csv_template(mapping: VariableMapping) -> str:
    """CSV with the 61 columns and one example row (median or mode of the training set)."""
    example: dict[str, Any] = {}
    for variable in mapping.raw_variables:
        observed = variable.observed
        value = observed.mode if variable.domain.type == "categorical" else observed.median
        if value is None or variable.nullable:
            example[variable.en] = ""
        else:
            example[variable.en] = int(value) if variable.dtype == "int" else float(value)
    return str(pd.DataFrame([example], columns=mapping.raw_en_names).to_csv(index=False))


def read_csv(content: bytes | str) -> pd.DataFrame:
    """Parse an uploaded CSV (comma or semicolon separated, UTF-8 or Latin-1)."""
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    if not text.strip():
        return pd.DataFrame()
    separator = ";" if text.splitlines()[0].count(";") > text.splitlines()[0].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=separator, dtype=str, keep_default_na=True)


def validate_frame(frame: pd.DataFrame, mapping: VariableMapping) -> CsvValidation:
    """Validate every row of ``frame`` against the mapping."""
    columns = [str(c).strip() for c in frame.columns]
    missing = [name for name in mapping.raw_en_names if name not in columns]
    unknown = [name for name in columns if name not in mapping.raw_en_names]
    errors: list[FieldError] = []
    valid: list[int] = []
    renamed = frame.copy()
    renamed.columns = columns
    for position, (_, row) in enumerate(renamed.iterrows(), start=1):
        row_errors = [
            FieldError(position, variable.en, message)
            for variable in mapping.raw_variables
            if (message := check_value(variable, row.get(variable.en))) is not None
        ]
        errors.extend(row_errors)
        if not row_errors:
            valid.append(position)
    return CsvValidation(len(renamed), valid, errors, missing, unknown)


def frame_to_records(frame: pd.DataFrame, mapping: VariableMapping) -> list[dict[str, Any]]:
    """Rows of a validated frame as API payloads, in file order."""
    columns = [str(c).strip() for c in frame.columns]
    renamed = frame.copy()
    renamed.columns = columns
    return [clean_record(row.to_dict(), mapping) for _, row in renamed.iterrows()]


def merge_results(frame: pd.DataFrame, predictions: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    """Input rows side by side with the API answers (same order)."""
    answers = pd.DataFrame(
        [
            {
                "probabilidade": p["probability"],
                "faixa_risco": p["risk_band"],
                "classe_prevista": p["predicted_class"],
                "campos_imputados": ";".join(p.get("imputed_fields", [])),
                "versao_modelo": p["model_version"],
            }
            for p in predictions
        ]
    )
    if len(answers) != len(frame):
        raise ValueError(f"{len(answers)} predictions for {len(frame)} rows")
    return pd.concat([frame.reset_index(drop=True), answers], axis=1)
