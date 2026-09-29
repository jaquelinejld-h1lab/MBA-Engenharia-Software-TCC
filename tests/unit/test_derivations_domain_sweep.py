"""Exhaustive sweep of the declared domains (no Hypothesis needed).

Every categorical raw variable is swept over all its declared codes and every
numeric one over a grid including both bounds; the derivations must always
land in the declared model categories, and every out-of-domain value must be
rejected explicitly by the contract and by the transformer (inventory D10).
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.data.contract import ModelRecord, RawRecord
from src.data.mapping import RawVariable, VariableMapping
from src.exceptions import TransformError
from src.features import derivations as d
from src.features.derivations import derive_laboratory_classes, derive_stateless
from src.features.transformer import HypertensionTransformer


def _grid(variable: RawVariable) -> list[float]:
    domain = variable.domain
    if domain.type == "categorical":
        return [float(v) for v in domain.values or []]
    lo, hi = float(domain.min or 0), float(domain.max or 0)
    return sorted({lo, hi, (lo + hi) / 2, math.floor((lo + hi) / 2) + 0.5})


def _to_record(derived: pd.DataFrame, cfg: Settings) -> dict[str, str]:
    completed = derived.copy()
    for column in ("egfr_afro", "cholesterol", "glucose"):
        completed[column] = completed[column].fillna(100.0)
    labs = derive_laboratory_classes(completed, cfg.features)
    record = {**completed.iloc[0].to_dict(), **labs.iloc[0].to_dict()}
    return {name: record[name] for name in ModelRecord.model_fields}


def test_every_declared_value_of_every_raw_variable_derives_to_declared_categories(
    cfg: Settings, mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> None:
    base = synthetic_dev.drop(columns=["target"]).iloc[[0]].copy()
    checked = 0
    for variable in mapping.raw_variables:
        values = _grid(variable) + ([math.nan] if variable.nullable else [])
        for value in values:
            row = base.copy()
            row[variable.en] = value
            derived = derive_stateless(row, cfg.features)
            ModelRecord.model_validate(_to_record(derived, cfg))
            checked += 1
    assert checked > 200


def test_food_days_exhaustive(cfg: Settings, mapping: VariableMapping) -> None:
    fc = cfg.features
    cats = mapping.model_categories
    for days in range(8):
        assert d.healthy_frequency(days, fc) in cats["beans_intake"]
        assert d.fish_frequency(days, fc) in cats["fish_intake"]
        assert d.juice_frequency(days, fc) in cats["juice_intake"]
        assert d.sweets_frequency(days, fc) in cats["sweets_intake"]
        assert d.meal_replacement_frequency(days, fc) in cats["meal_replacement_freq"]


def test_activity_level_exhaustive_over_types_and_flags(
    cfg: Settings, mapping: VariableMapping
) -> None:
    fc = cfg.features
    cats = set(mapping.model_categories["physical_activity_level"])
    minutes = [0, 1, 74, 75, 149, 150, 299, 300, 6730]
    for exercise, exercise_type, chores, total in itertools.product(
        [1, 2], [*range(1, 18), math.nan], [1, 2], minutes
    ):
        row = pd.Series(
            {"exercise": exercise, "exercise_type": exercise_type, "chores_physical_effort": chores}
        )
        assert d.activity_level(row, total, fc) in cats


def test_smoking_history_exhaustive(cfg: Settings, mapping: VariableMapping) -> None:
    cats = set(mapping.model_categories["smoking_history"])
    for a, b, c in itertools.product([1, 2, 3, math.nan], [1, 2, math.nan], [1, 2, 3, math.nan]):
        assert d.smoking_history(a, b, c, cfg.features) in cats


# --------------------------------------------------------------- D10 tests
def _out_of_domain_values(variable: RawVariable) -> list[float]:
    domain = variable.domain
    if domain.type == "categorical":
        return [max(domain.values or [0]) + 1, -1]
    return [float(domain.min or 0) - 1, float(domain.max or 0) + 1]


def _valid_base(frame: pd.DataFrame, mapping: VariableMapping) -> dict[str, float | None]:
    """Build one valid payload from the first row of ``frame``.

    The development frame keeps every code as float, because a nullable column forces a
    float dtype in pandas. ``RawRecord`` is strict and expects integer codes, which is
    what ``src.app.validation.clean_record`` sends from a CSV. Casting here by the
    mapping dtype makes the test exercise the contract instead of pandas dtype
    inference, which is not identical across pandas versions.
    """
    dtypes = {variable.en: variable.dtype for variable in mapping.raw_variables}
    row = frame.drop(columns=["target"]).iloc[0].to_dict()
    base: dict[str, float | None] = {}
    for column, value in row.items():
        name = str(column)
        if pd.isna(value):
            base[name] = None
        else:
            base[name] = int(value) if dtypes.get(name) == "int" else float(value)
    return base


def test_contract_rejects_every_out_of_domain_value(
    mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> None:
    base = _valid_base(synthetic_dev, mapping)
    RawRecord.model_validate(base)
    for variable in mapping.raw_variables:
        for bad in _out_of_domain_values(variable):
            payload = {**base, variable.en: bad}
            with pytest.raises(ValidationError):
                RawRecord.model_validate(payload)


def test_contract_rejects_missing_where_not_nullable(
    mapping: VariableMapping, synthetic_dev: pd.DataFrame
) -> None:
    base = _valid_base(synthetic_dev, mapping)
    for variable in mapping.raw_variables:
        if variable.nullable:
            continue
        with pytest.raises(ValidationError):
            RawRecord.model_validate({**base, variable.en: None})


@pytest.mark.parametrize(
    "en",
    [
        "sex",
        "region",
        "race_color",
        "lives_with_spouse",
        "health_insurance",
        "dx_chronic_kidney",
        "dx_diabetes",
        "dx_heart_disease",
        "dx_stroke",
        "soda_type",
        "milk_type",
        "salt_intake",
        "alcohol",
        "self_rated_health",
        "sleep_medication",
        "fish_days",
        "juice_days",
        "sweets_days",
        "meal_replacement_days",
    ],
)
def test_transformer_raises_instead_of_silent_fallback(
    en: str,
    cfg: Settings,
    mapping: VariableMapping,
    synthetic_split: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    train, test = synthetic_split
    transformer = HypertensionTransformer(cfg.features, mapping, mode="production").fit(train)
    row = test.iloc[[0]].copy()
    row[en] = max(mapping.raw(en).domain.values or [0]) + 1
    with pytest.raises(TransformError):
        transformer.transform(row)


def test_transformer_rejects_unknown_imputation_base(
    cfg: Settings, mapping: VariableMapping, synthetic_split: tuple[pd.DataFrame, pd.DataFrame]
) -> None:
    train, _ = synthetic_split
    lab = cfg.features.laboratory.model_copy(update={"imputation_base_variables": ["sex", "ghost"]})
    features = cfg.features.model_copy(update={"laboratory": lab})
    with pytest.raises(TransformError, match="ghost"):
        HypertensionTransformer(features, mapping, mode="production").fit(train)


def test_cut_rejects_values_outside_bins(cfg: Settings) -> None:
    with pytest.raises(TransformError, match="outside"):
        d.egfr_class(-1.0, cfg.features)
    with pytest.raises(TransformError):
        d.sweets_frequency(-1, cfg.features)
    with pytest.raises(TransformError):
        d.meal_replacement_frequency(-1, cfg.features)
    assert np.isnan(float("nan"))
