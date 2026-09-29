"""Property-based tests of the derivations over the domain declared in the mapping.

Strategies are built from ``variable_mapping.yaml`` so the tested domain is
the declared one, never a copy. Requires Hypothesis (dev dependency); the
exhaustive sweep in ``test_derivations_domain_sweep.py`` covers the same
properties without it.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import FeaturesConfig, Settings
from src.data.contract import ModelRecord
from src.data.mapping import VariableMapping, load_mapping
from src.features import derivations as d
from src.features.derivations import derive_laboratory_classes, derive_stateless

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

_MAPPING: VariableMapping = load_mapping()
_CATEGORIES: dict[str, list[str]] = _MAPPING.model_categories


def _strategy_for(en: str, *, allow_none: bool = True) -> st.SearchStrategy[float | int | None]:
    """Strategy over the declared domain of one raw variable (with None when nullable)."""
    variable = _MAPPING.raw(en)
    domain = variable.domain
    if domain.type == "categorical":
        base: st.SearchStrategy[float | int | None] = st.sampled_from(list(domain.values or []))
    elif variable.dtype == "int":
        base = st.integers(int(domain.min or 0), int(domain.max or 0))
    else:
        base = st.floats(
            float(domain.min or 0), float(domain.max or 0), allow_nan=False, allow_infinity=False
        )
    if variable.nullable and allow_none:
        return st.one_of(st.none(), base)
    return base


def _raw_row_strategy() -> st.SearchStrategy[dict[str, float | int | None]]:
    return st.fixed_dictionaries({v.en: _strategy_for(v.en) for v in _MAPPING.raw_variables})


@pytest.fixture(scope="module")
def fc(cfg: Settings) -> FeaturesConfig:
    return cfg.features


@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(days=st.integers(0, 7))
def test_food_functions_stay_in_declared_categories(days: int, cfg: Settings) -> None:
    fc = cfg.features
    assert d.healthy_frequency(days, fc) in _CATEGORIES["beans_intake"]
    assert d.fish_frequency(days, fc) in _CATEGORIES["fish_intake"]
    assert d.juice_frequency(days, fc) in _CATEGORIES["juice_intake"]
    assert d.sweets_frequency(days, fc) in _CATEGORIES["sweets_intake"]
    assert d.meal_replacement_frequency(days, fc) in _CATEGORIES["meal_replacement_freq"]


@settings(max_examples=300, deadline=None)
@given(
    weight=st.floats(25, 250, allow_nan=False),
    height=st.floats(120, 220, allow_nan=False),
)
def test_bmi_class_is_monotone_and_in_domain(weight: float, height: float, cfg: Settings) -> None:
    fc = cfg.features
    value = d.bmi(weight, height, fc)
    label = d.bmi_class(value, fc)
    assert label in _CATEGORIES["bmi_class"]
    # heavier at same height never yields a "lower" class except the underweight code "5"
    heavier = d.bmi_class(d.bmi(weight * 1.1, height, fc), fc)
    order = {"5": 0, "1": 1, "2": 2, "3": 3, "4": 4}
    assert order[heavier] >= order[label]


@settings(max_examples=300, deadline=None)
@given(value=st.floats(0, 700, allow_nan=False))
def test_laboratory_classes_in_domain(value: float, cfg: Settings) -> None:
    fc = cfg.features
    assert d.egfr_class(value, fc) in _CATEGORIES["egfr_class"]
    assert d.glucose_class(value, fc) in _CATEGORIES["glucose_class"]
    assert d.cholesterol_desirable(value, fc) in _CATEGORIES["cholesterol_desirable"]


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(row=_raw_row_strategy())
def test_full_row_derivation_satisfies_model_contract(
    row: dict[str, float | int | None], cfg: Settings
) -> None:
    """Any row inside the declared raw domain derives to declared model categories."""
    frame = pd.DataFrame([{k: (math.nan if v is None else v) for k, v in row.items()}])
    derived = derive_stateless(frame, cfg.features)
    # laboratory exams may be None in the domain; the transformer imputes them, so here
    # the categorization is only checked when the value is present
    for column in ("egfr_afro", "cholesterol", "glucose"):
        if math.isnan(float(derived[column].iloc[0])):
            derived[column] = 100.0  # any in-domain placeholder for the categorization step
    labs = derive_laboratory_classes(derived, cfg.features)
    record = {**derived.iloc[0].to_dict(), **labs.iloc[0].to_dict()}
    payload = {name: record[name] for name in ModelRecord.model_fields}
    ModelRecord.model_validate(payload)
