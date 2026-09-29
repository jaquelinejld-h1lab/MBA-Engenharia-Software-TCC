"""Pure feature derivations reproducing ``add_variaveis_ajustadas`` (Dias, 2024).

Every function is deterministic, takes its cut points from ``FeaturesConfig``
and returns a new object; nothing is mutated in place and nothing is printed.
Category codes are the strings the original notebook produced (``"1"``,
``"2"``, ...), so the one-hot design matrix is identical.

Fidelity notes
--------------
* Missing-value semantics of the original row-wise functions are kept: a
  ``NaN`` compared with a number is ``False`` in Python, exactly as inside
  ``DataFrame.apply``.
* ``smoking_history`` reproduces the string-versus-integer comparison bug of
  the original (inventory item D4) when
  ``features.smoking.reproduce_string_int_comparison_bug`` is true. See
  ``docs/adr/0001-reproduce-smoking-history-bug.md``.
* Inputs outside the declared domain raise ``TransformError`` instead of the
  original silent fallbacks (``None`` or ``'Frequência Inválida'``); those
  fallbacks never occur on the training data, so the design matrix is
  unchanged (inventory item D10).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import FeaturesConfig
from src.exceptions import TransformError

# Category codes used by the original notebook
VERY_HEALTHY, HEALTHY, MODERATE, UNHEALTHY = "1", "2", "3", "4"
YES, NO = "1", "2"
INACTIVE, LOW, MODERATE_ACTIVITY, VIGOROUS = "1", "2", "3", "4"


def _is_missing(value: float | int | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _require(value: float | int | None, name: str) -> float:
    if _is_missing(value):
        raise TransformError(f"{name}: value is missing and no fallback is defined")
    return float(value)  # type: ignore[arg-type]


def _require_days(value: float | int | None, name: str, cfg: FeaturesConfig) -> float:
    """Days per week inside ``0..max_days_per_week``; anything else is an explicit error (D10)."""
    d = _require(value, name)
    if not 0 <= d <= cfg.food_frequency.max_days_per_week:
        raise TransformError(
            f"{name}: {value} is outside 0..{cfg.food_frequency.max_days_per_week}"
        )
    return d


def _check_codes(series: pd.Series, allowed: set[int], name: str) -> pd.Series:
    """Reject survey codes outside ``allowed`` (missing values pass through)."""
    present = series.dropna()
    bad = present[~present.isin(allowed)]
    if not bad.empty:
        raise TransformError(f"{name}: codes {sorted(set(bad.tolist()))} outside {sorted(allowed)}")
    return series


# ----------------------------------------------------------------- food frequency
def healthy_frequency(days: float | int | None, cfg: FeaturesConfig) -> str:
    """Classify weekly frequency of a protective food (beans, salad, vegetables, fruit).

    Parameters
    ----------
    days : float or int
        Days per week, 0 to 7.
    cfg : FeaturesConfig
        ``food_frequency.healthy_very_days`` (7), ``healthy_min_days`` (5) and
        ``moderate_min_days`` (3).

    Returns
    -------
    str
        ``"1"`` every day, ``"2"`` 5 to 6 days, ``"3"`` 3 to 4 days, ``"4"`` otherwise.
    """
    d = _require_days(days, "food frequency", cfg)
    ff = cfg.food_frequency
    if d == ff.healthy_very_days:
        return VERY_HEALTHY
    if ff.healthy_min_days <= d < ff.healthy_very_days:
        return HEALTHY
    if ff.moderate_min_days <= d < ff.healthy_min_days:
        return MODERATE
    return UNHEALTHY


def fish_frequency(days: float | int | None, cfg: FeaturesConfig) -> str:
    """Classify weekly fish consumption: ``>=3`` "1", 2 "2", 1 "3", 0 "4"."""
    d = _require_days(days, "fish_days", cfg)
    if d >= cfg.food_frequency.fish_very_healthy_min_days:
        return VERY_HEALTHY
    if d == 2:  # noqa: PLR2004  (codes fixed by the original categorization)
        return HEALTHY
    if d == 1:
        return MODERATE
    if d == 0:
        return UNHEALTHY
    raise TransformError(f"fish_days: {days} is outside 0..7")


def juice_frequency(days: float | int | None, cfg: FeaturesConfig) -> str:
    """Classify natural juice frequency; more juice is worse (0..1 "1" up to 6..7 "4")."""
    d = _require_days(days, "juice_days", cfg)
    e1, e2, e3, e4 = cfg.food_frequency.juice_edges
    if 0 <= d <= e1:
        return VERY_HEALTHY
    if e1 < d <= e2:
        return HEALTHY
    if e2 < d <= e3:
        return MODERATE
    if e3 < d <= e4:
        return UNHEALTHY
    raise TransformError(f"juice_days: {days} is outside 0..{e4}")


def sweets_frequency(days: float | int | None, cfg: FeaturesConfig) -> str:
    """Classify sweets frequency: 0 "1", 1 "2", 2..3 "3", ``>=4`` "4"."""
    d = _require_days(days, "sweets_days", cfg)
    if d == 0:
        return VERY_HEALTHY
    if d == 1:
        return HEALTHY
    if 2 <= d < cfg.food_frequency.sweets_unhealthy_from_days:  # noqa: PLR2004
        return MODERATE
    if d >= cfg.food_frequency.sweets_unhealthy_from_days:
        return UNHEALTHY
    raise TransformError(f"sweets_days: {days} is outside 0..7")


def meal_replacement_frequency(days: float | int | None, cfg: FeaturesConfig) -> str:
    """Classify replacing lunch or dinner with snacks: 0 "1", 1 "2", 2 "3", ``>=3`` "4"."""
    d = _require_days(days, "meal_replacement_days", cfg)
    if d == 0:
        return VERY_HEALTHY
    if d == 1:
        return HEALTHY
    if d == 2:  # noqa: PLR2004
        return MODERATE
    if d >= cfg.food_frequency.meal_replacement_unhealthy_from_days:
        return UNHEALTHY
    raise TransformError(f"meal_replacement_days: {days} is outside 0..7")


# ------------------------------------------------------------ physical activity
def total_activity_minutes(row: pd.Series, cfg: FeaturesConfig) -> float:
    """Weekly minutes of physical activity summed over five blocks.

    Reproduces ``calcular_tempo_total_atividade_fisica``: exercise days count
    ``minutes_per_exercise_day`` each; the other four blocks multiply days by
    hours and minutes. Conditions on ``NaN`` evaluate to ``False``, as in the
    original ``DataFrame.apply``.
    """
    pa = cfg.physical_activity
    total = 0.0
    if row["exercise"] == 1:
        total += row["exercise_days"] * pa.minutes_per_exercise_day
    if row["walk_bike_days"] > 0:
        total += row["walk_bike_days"] * (row["walk_bike_hours"] * 60 + row["walk_bike_minutes"])
    if row["work_physical_effort"] == 1:
        total += row["work_effort_days"] * (
            row["work_effort_hours"] * 60 + row["work_effort_minutes"]
        )
    if row["commute_walk_bike"] in (1, 2):
        total += row["commute_hours"] * 60 + row["commute_minutes"]
    if row["chores_physical_effort"] == 1:
        total += row["chores_days"] * (row["chores_hours"] * 60 + row["chores_minutes"])
    return float(total)


def activity_level(  # noqa: PLR0911  (mirrors the original decision tree one branch per return)
    row: pd.Series, total_minutes: float, cfg: FeaturesConfig
) -> str:
    """Classify the activity level from total minutes and exercise type.

    Reproduces ``classificar_nivel_atividade_fisica`` including its fall-through
    to ``"2"`` when no branch matches.
    """
    pa = cfg.physical_activity
    moderate = set(pa.moderate_exercise_types)
    vigorous = set(pa.vigorous_exercise_types)
    exercise_type = row["exercise_type"]

    if total_minutes == 0:
        return INACTIVE
    if total_minutes < pa.low_activity_below_min:
        return LOW
    if row["exercise"] == 1:
        if exercise_type in vigorous and total_minutes >= pa.vigorous_min_minutes:
            return VIGOROUS
        if exercise_type in moderate and total_minutes >= pa.moderate_min_minutes:
            return MODERATE_ACTIVITY
        if exercise_type in vigorous and total_minutes >= pa.moderate_min_minutes:
            return MODERATE_ACTIVITY
        if exercise_type in moderate and total_minutes >= pa.moderate_to_vigorous_min_minutes:
            return VIGOROUS
    elif row["chores_physical_effort"] == 1 and total_minutes >= pa.moderate_min_minutes:
        return MODERATE_ACTIVITY
    return LOW


# ------------------------------------------------------------------- smoking
def smoking_history(
    smokes_currently: float | int | None,
    smoked_daily_past: float | int | None,
    smoked_past: float | int | None,
    cfg: FeaturesConfig,
) -> str:
    """Smoking history: ``"1"`` ever smoked, ``"2"`` never.

    Fidelity (inventory D4, ADR 0001): the original code stored the current
    smoker flag as the *string* ``'1'`` and then compared it with the
    *integer* ``1``, so current smokers only counted when ``smoked_daily_past``
    or ``smoked_past`` were also ``1``. With
    ``reproduce_string_int_comparison_bug=True`` the current smoker flag is
    therefore ignored on purpose, to keep the champion design matrix
    identical to the dissertation.
    """
    ever = smoked_daily_past == 1 or smoked_past == 1
    if not cfg.smoking.reproduce_string_int_comparison_bug:
        ever = ever or smokes_currently in (1, 2)
    return YES if ever else NO


# ------------------------------------------------------------------ body size
def bmi(weight_kg: float, height_cm: float, cfg: FeaturesConfig) -> float:
    """Body mass index in kg/m^2 from weight in kg and height in cm."""
    height_m = height_cm / cfg.bmi.height_unit_divisor
    return float(weight_kg / (height_m**2))


def bmi_class(value: float | None, cfg: FeaturesConfig) -> str:
    """Classify BMI: ``>=40`` "4", ``>=30`` "3", ``>=25`` "2", ``>=18.5`` "1", else "5"."""
    v = _require(value, "bmi")
    b = cfg.bmi
    if v >= b.severe_obesity_from:
        return "4"
    if v >= b.obesity_from:
        return "3"
    if v >= b.overweight_from:
        return "2"
    if v >= b.underweight_below:
        return "1"
    return "5"


# ------------------------------------------------------------ laboratory exams
def egfr_class(value: float | None, cfg: FeaturesConfig) -> str:
    """Kidney function class from eGFR; labels are ordered worst ("5") to normal ("1")."""
    v = _require(value, "egfr_afro")
    lab = cfg.laboratory
    return _cut(v, lab.egfr_bin_edges, lab.egfr_labels, "egfr_afro")


def glucose_class(value: float | None, cfg: FeaturesConfig) -> str:
    """Glucose class: ``<70`` "4", 70..99 "1", 100..125 "2", ``>=126`` "3"."""
    v = _require(value, "glucose")
    lab = cfg.laboratory
    return _cut(v, lab.glucose_bin_edges, lab.glucose_labels, "glucose")


def cholesterol_desirable(value: float | None, cfg: FeaturesConfig) -> str:
    """``"1"`` when total cholesterol is below the desirable cut point, else ``"2"``."""
    v = _require(value, "cholesterol")
    return YES if v < cfg.laboratory.cholesterol_desirable_below_mg_dl else NO


def _cut(value: float, edges: list[float], labels: list[str], name: str) -> str:
    """Left-closed binning identical to ``pd.cut(..., right=False)``."""
    for lo, hi, label in zip(edges[:-1], edges[1:], labels, strict=True):
        if lo <= value < hi:
            return label
    raise TransformError(f"{name}: {value} is outside the declared bins {edges}")


# ---------------------------------------------------------- frame-level helpers
def _code_to_str(series: pd.Series) -> pd.Series:
    """Numeric survey codes to the string categories used by the encoder; NaN kept."""
    return series.map(lambda v: np.nan if _is_missing(v) else str(int(v)))


def _cut_series(series: pd.Series, edges: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(series, bins=edges, labels=labels, right=False).astype(object)


def derive_stateless(frame: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Derive every model variable that needs no fitted state, plus imputation bases.

    Parameters
    ----------
    frame : pandas.DataFrame
        The 61 raw variables in English names (extra columns are ignored).
    cfg : FeaturesConfig
        Cut points and switches.

    Returns
    -------
    pandas.DataFrame
        Same index; string-coded model variables, the three raw laboratory
        columns untouched (``egfr_afro``, ``cholesterol``, ``glucose``) and the
        intermediate columns ``race_color``, ``waist_risk_increased`` and
        ``dx_high_cholesterol_rec`` used only by the KNN imputation.
    """
    # Columns are collected in a dict and materialised once at the end: assigning
    # 45 columns one by one to a DataFrame costs ~15 ms per call (dominant cost of a
    # single-row prediction) while the result is identical (fidelity hashes pin it).
    out: dict[str, Any] = {}

    # demographic
    out["age_group"] = _cut_series(frame["age"], cfg.age.bin_edges, cfg.age.labels)
    out["sex"] = _code_to_str(frame["sex"])
    out["region"] = _code_to_str(frame["region"])
    out["black_or_brown"] = frame["race_color"].map({1: NO, 2: YES, 3: NO, 4: YES, 5: NO, 9: NO})
    out["household_size_group"] = _cut_series(
        frame["household_size"], cfg.household_size.bin_edges, cfg.household_size.labels
    )
    out["lives_with_spouse"] = _code_to_str(frame["lives_with_spouse"])

    # socioeconomic (D7: missing income counts as zero, as in the original)
    income_columns = [
        "income_main_job",
        "income_main_job_in_kind",
        "income_other_jobs",
        "income_other_jobs_in_kind",
        "income_pension",
        "income_alimony",
        "income_rent",
    ]
    total_income = frame[income_columns].sum(axis=1, skipna=cfg.income.missing_income_as_zero)
    wage = cfg.income.minimum_wage_brl
    edges = [0.0, *[m * wage for m in cfg.income.bracket_multipliers], float("inf")]
    labels = [str(i) for i in range(1, len(edges))]
    out["income_bracket"] = _cut_series(total_income, edges, labels)
    out["health_insurance"] = _code_to_str(frame["health_insurance"])

    # pre-existing conditions
    out["dx_chronic_kidney"] = _code_to_str(frame["dx_chronic_kidney"])
    out["dx_diabetes_rec"] = _code_to_str(frame["dx_diabetes"].replace(3, 2).fillna(3))
    out["dx_high_cholesterol_rec"] = _code_to_str(frame["dx_high_cholesterol"].fillna(3))
    out["dx_heart_disease"] = _code_to_str(frame["dx_heart_disease"].fillna(2))
    out["dx_stroke"] = _code_to_str(frame["dx_stroke"].fillna(2))

    # diet
    out["beans_intake"] = frame["beans_days"].map(lambda d: healthy_frequency(d, cfg))
    out["salad_intake"] = frame["salad_days"].map(lambda d: healthy_frequency(d, cfg))
    out["vegetables_intake"] = frame["vegetables_days"].map(lambda d: healthy_frequency(d, cfg))
    out["fish_intake"] = frame["fish_days"].map(lambda d: fish_frequency(d, cfg))
    out["juice_intake"] = frame["juice_days"].map(lambda d: juice_frequency(d, cfg))
    out["fruit_intake"] = frame["fruit_days"].map(lambda d: healthy_frequency(d, cfg))
    # D10: the recoded "4" means "does not consume" and only arises from a missing answer
    out["soda_type_rec"] = _code_to_str(
        _check_codes(frame["soda_type"], {1, 2, 3}, "soda_type").fillna(4)
    )
    out["milk_type_rec"] = _code_to_str(
        _check_codes(frame["milk_type"], {1, 2, 3}, "milk_type").fillna(4)
    )
    out["sweets_intake"] = frame["sweets_days"].map(lambda d: sweets_frequency(d, cfg))
    out["meal_replacement_freq"] = frame["meal_replacement_days"].map(
        lambda d: meal_replacement_frequency(d, cfg)
    )
    out["salt_intake_rec"] = frame["salt_intake"].map({1: "3", 2: "3", 3: "2", 4: "1", 5: "1"})
    out["alcohol"] = _code_to_str(frame["alcohol"])

    # physical activity
    minutes = frame.apply(lambda row: total_activity_minutes(row, cfg), axis=1)
    out["physical_activity_level"] = [
        activity_level(row, m, cfg) for (_, row), m in zip(frame.iterrows(), minutes, strict=True)
    ]

    # smoking (D4 reproduced, see ADR 0001)
    out["smoking_history"] = [
        smoking_history(a, b, c, cfg)
        for a, b, c in zip(
            frame["smokes_currently"], frame["smoked_daily_past"], frame["smoked_past"], strict=True
        )
    ]
    out["sleep_medication"] = _code_to_str(frame["sleep_medication"])

    # anthropometry
    bmi_values = [
        bmi(w, h, cfg) for w, h in zip(frame["weight_kg"], frame["height_cm"], strict=True)
    ]
    out["bmi_class"] = [bmi_class(v, cfg) for v in bmi_values]
    female, male = frame["sex"] == 2, frame["sex"] == 1  # noqa: PLR2004
    waist = frame["waist_cm"]
    out["waist_risk_increased"] = np.where(
        (female & (waist > cfg.waist.female_increased_cm))
        | (male & (waist > cfg.waist.male_increased_cm)),
        YES,
        NO,
    )

    # health perception
    out["self_rated_health_rec"] = frame["self_rated_health"].map(
        {1: "2", 2: "2", 3: "2", 4: "1", 5: "1"}
    )
    out["angina"] = np.where(
        (frame["angina_walking_fast"] == 1) & (frame["angina_walking_normal"] == 1), YES, NO
    )

    # imputation base and raw laboratory values (categorized after imputation)
    out["race_color"] = _code_to_str(frame["race_color"])
    for column in ("egfr_afro", "cholesterol", "glucose"):
        out[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return pd.DataFrame(out, index=frame.index)


def derive_laboratory_classes(frame: pd.DataFrame, cfg: FeaturesConfig) -> pd.DataFrame:
    """Categorize the (already imputed) laboratory columns.

    Parameters
    ----------
    frame : pandas.DataFrame
        Must contain complete ``egfr_afro``, ``cholesterol`` and ``glucose``.

    Returns
    -------
    pandas.DataFrame
        ``egfr_class``, ``cholesterol_desirable`` and ``glucose_class``.
    """
    out = pd.DataFrame(index=frame.index)
    out["egfr_class"] = [egfr_class(v, cfg) for v in frame["egfr_afro"]]
    out["cholesterol_desirable"] = [cholesterol_desirable(v, cfg) for v in frame["cholesterol"]]
    out["glucose_class"] = [glucose_class(v, cfg) for v in frame["glucose"]]
    return out
