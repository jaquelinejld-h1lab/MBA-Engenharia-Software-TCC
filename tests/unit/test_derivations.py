"""Boundary cases of every pure derivation, against the original notebook rules."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from src.config import FeaturesConfig, Settings
from src.exceptions import TransformError
from src.features import derivations as d


@pytest.fixture(scope="module")
def fc(cfg: Settings) -> FeaturesConfig:
    return cfg.features


# ------------------------------------------------------------------ food
@pytest.mark.parametrize(
    ("days", "expected"),
    [(7, "1"), (6, "2"), (5, "2"), (4, "3"), (3, "3"), (2, "4"), (1, "4"), (0, "4")],
)
def test_healthy_frequency(days: int, expected: str, fc: FeaturesConfig) -> None:
    assert d.healthy_frequency(days, fc) == expected


@pytest.mark.parametrize(("days", "expected"), [(7, "1"), (3, "1"), (2, "2"), (1, "3"), (0, "4")])
def test_fish_frequency(days: int, expected: str, fc: FeaturesConfig) -> None:
    assert d.fish_frequency(days, fc) == expected


@pytest.mark.parametrize(
    ("days", "expected"),
    [(0, "1"), (1, "1"), (2, "2"), (3, "2"), (4, "3"), (5, "3"), (6, "4"), (7, "4")],
)
def test_juice_frequency(days: int, expected: str, fc: FeaturesConfig) -> None:
    assert d.juice_frequency(days, fc) == expected


@pytest.mark.parametrize(
    ("days", "expected"), [(0, "1"), (1, "2"), (2, "3"), (3, "3"), (4, "4"), (7, "4")]
)
def test_sweets_frequency(days: int, expected: str, fc: FeaturesConfig) -> None:
    assert d.sweets_frequency(days, fc) == expected


@pytest.mark.parametrize(("days", "expected"), [(0, "1"), (1, "2"), (2, "3"), (3, "4"), (7, "4")])
def test_meal_replacement_frequency(days: int, expected: str, fc: FeaturesConfig) -> None:
    assert d.meal_replacement_frequency(days, fc) == expected


@pytest.mark.parametrize(
    "func",
    [
        d.healthy_frequency,
        d.fish_frequency,
        d.juice_frequency,
        d.sweets_frequency,
        d.meal_replacement_frequency,
    ],
)
def test_food_functions_reject_missing(func, fc: FeaturesConfig) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(TransformError):
        func(math.nan, fc)


def test_food_functions_reject_out_of_domain(fc: FeaturesConfig) -> None:
    with pytest.raises(TransformError):
        d.fish_frequency(-1, fc)
    with pytest.raises(TransformError):
        d.juice_frequency(8, fc)


# ------------------------------------------------------- physical activity
def _row(**values: float) -> pd.Series:
    base = {
        "exercise": 2,
        "exercise_days": math.nan,
        "exercise_type": math.nan,
        "walk_bike_days": 0,
        "walk_bike_hours": math.nan,
        "walk_bike_minutes": math.nan,
        "work_physical_effort": math.nan,
        "work_effort_days": math.nan,
        "work_effort_hours": math.nan,
        "work_effort_minutes": math.nan,
        "commute_walk_bike": math.nan,
        "commute_hours": math.nan,
        "commute_minutes": math.nan,
        "chores_physical_effort": 2,
        "chores_days": math.nan,
        "chores_hours": math.nan,
        "chores_minutes": math.nan,
    }
    base.update(values)
    return pd.Series(base)


def test_total_minutes_sums_blocks_and_ignores_nan(fc: FeaturesConfig) -> None:
    row = _row(
        exercise=1,
        exercise_days=3,
        walk_bike_days=2,
        walk_bike_hours=1,
        walk_bike_minutes=30,
        commute_walk_bike=1,
        commute_hours=0,
        commute_minutes=20,
    )
    # 3*30 + 2*(60+30) + 20 = 290
    assert d.total_activity_minutes(row, fc) == 290


def test_total_minutes_zero_when_everything_missing(fc: FeaturesConfig) -> None:
    assert d.total_activity_minutes(_row(), fc) == 0


@pytest.mark.parametrize(
    ("row", "minutes", "expected"),
    [
        (_row(), 0, "1"),
        (_row(), 149, "2"),
        (_row(exercise=1, exercise_type=3), 75, "2"),  # <150 wins before type logic
        (_row(exercise=1, exercise_type=3), 150, "4"),  # vigorous type, >=75 and >=150
        (_row(exercise=1, exercise_type=1), 150, "3"),  # moderate type, >=150
        (_row(exercise=1, exercise_type=1), 300, "3"),  # moderate branch matches first
        (_row(exercise=1, exercise_type=17), 400, "2"),  # type 17 in no group: fall-through
        (_row(exercise=1, exercise_type=math.nan), 400, "2"),
        (_row(chores_physical_effort=1), 150, "3"),
        (_row(chores_physical_effort=2), 150, "2"),
    ],
)
def test_activity_level(row: pd.Series, minutes: float, expected: str, fc: FeaturesConfig) -> None:
    assert d.activity_level(row, minutes, fc) == expected


# ---------------------------------------------------------------- smoking
def test_smoking_history_reproduces_bug_by_default(fc: FeaturesConfig) -> None:
    # A daily smoker with no recorded past history is classified as "never" (D4).
    assert fc.smoking.reproduce_string_int_comparison_bug is True
    assert d.smoking_history(1, math.nan, math.nan, fc) == "2"
    assert d.smoking_history(1, 1, math.nan, fc) == "1"
    assert d.smoking_history(3, math.nan, 1, fc) == "1"
    assert d.smoking_history(3, 2, 2, fc) == "2"


def test_smoking_history_corrected_variant(cfg: Settings) -> None:
    fixed = cfg.features.model_copy(
        update={
            "smoking": cfg.features.smoking.model_copy(
                update={"reproduce_string_int_comparison_bug": False}
            )
        }
    )
    assert d.smoking_history(1, math.nan, math.nan, fixed) == "1"
    assert d.smoking_history(2, math.nan, math.nan, fixed) == "1"
    assert d.smoking_history(3, math.nan, math.nan, fixed) == "2"


# --------------------------------------------------------------- body size
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (40.0, "4"),
        (39.99, "3"),
        (30.0, "3"),
        (29.99, "2"),
        (25.0, "2"),
        (24.99, "1"),
        (18.5, "1"),
        (18.49, "5"),
    ],
)
def test_bmi_class(value: float, expected: str, fc: FeaturesConfig) -> None:
    assert d.bmi_class(value, fc) == expected


def test_bmi_formula(fc: FeaturesConfig) -> None:
    assert d.bmi(80, 200, fc) == pytest.approx(20.0)


# -------------------------------------------------------------- laboratory
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "5"),
        (29.9, "5"),
        (30, "4"),
        (44.9, "4"),
        (45, "3"),
        (59.9, "3"),
        (60, "2"),
        (89.9, "2"),
        (90, "1"),
        (557.8, "1"),
    ],
)
def test_egfr_class(value: float, expected: str, fc: FeaturesConfig) -> None:
    assert d.egfr_class(value, fc) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "4"), (69.9, "4"), (70, "1"), (99.9, "1"), (100, "2"), (125.9, "2"), (126, "3")],
)
def test_glucose_class(value: float, expected: str, fc: FeaturesConfig) -> None:
    assert d.glucose_class(value, fc) == expected


@pytest.mark.parametrize(("value", "expected"), [(189.9, "1"), (190, "2"), (433, "2")])
def test_cholesterol_desirable(value: float, expected: str, fc: FeaturesConfig) -> None:
    assert d.cholesterol_desirable(value, fc) == expected


def test_laboratory_rejects_missing(fc: FeaturesConfig) -> None:
    for func in (d.egfr_class, d.glucose_class, d.cholesterol_desirable):
        with pytest.raises(TransformError):
            func(math.nan, fc)


# ----------------------------------------------------------- frame level
def test_derive_stateless_is_pure_and_typed(
    synthetic_dev: pd.DataFrame, fc: FeaturesConfig
) -> None:
    before = synthetic_dev.copy()
    out = d.derive_stateless(synthetic_dev, fc)
    pd.testing.assert_frame_equal(before, synthetic_dev)  # no mutation
    assert out.index.equals(synthetic_dev.index)
    for column in ("age_group", "income_bracket", "physical_activity_level", "bmi_class"):
        assert out[column].map(type).eq(str).all()
    assert set(
        (
            "egfr_afro",
            "cholesterol",
            "glucose",
            "race_color",
            "waist_risk_increased",
            "dx_high_cholesterol_rec",
        )
    ) <= set(out.columns)


def test_income_bracket_edges(synthetic_dev: pd.DataFrame, fc: FeaturesConfig) -> None:
    frame = synthetic_dev.head(3).copy()
    income = [
        "income_main_job",
        "income_main_job_in_kind",
        "income_other_jobs",
        "income_other_jobs_in_kind",
        "income_pension",
        "income_alimony",
        "income_rent",
    ]
    frame[income] = math.nan
    frame.loc[frame.index[0], "income_main_job"] = 0.0  # zero -> "1"
    frame.loc[frame.index[1], "income_main_job"] = 678.0  # exactly 1 SM -> "4"
    frame.loc[frame.index[2], "income_main_job"] = 678.0 * 20  # 20 SM -> "9"
    out = d.derive_stateless(frame, fc)
    assert list(out["income_bracket"]) == ["1", "4", "9"]


def test_all_missing_income_becomes_bracket_one(
    synthetic_dev: pd.DataFrame, fc: FeaturesConfig
) -> None:
    frame = synthetic_dev.head(1).copy()
    for column in frame.columns:
        if column.startswith("income_"):
            frame[column] = math.nan
    assert d.derive_stateless(frame, fc)["income_bracket"].iloc[0] == "1"  # D7 reproduced
