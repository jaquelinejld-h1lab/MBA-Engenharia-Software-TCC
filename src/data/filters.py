"""Population filters, target construction, renaming and the holdout split.

The six filters run in the exact order of ``carregar_dividir_dados`` in the
original ``funcoes_mecai_v3.py`` (inventory, section 3.2), each one logging
how many rows it removed. The only deliberate difference is correction D1:
rows without a blood pressure measurement are excluded explicitly, with a
logged count, instead of silently becoming negatives.

All functions are pure: they return new frames and never mutate their input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import PopulationFiltersConfig, SplitConfig, TargetConfig
from src.data.mapping import VariableMapping
from src.exceptions import DataContractError
from src.logging_setup import get_logger

logger = get_logger(__name__)

TARGET_COLUMN = "target"


@dataclass(frozen=True)
class FilterStep:
    """One filter application."""

    name: str
    removed: int
    remaining: int


@dataclass(frozen=True)
class FilterReport:
    """Row counts of every step, in order."""

    rows_in: int
    steps: list[FilterStep] = field(default_factory=list)
    blood_pressure_missing_removed: int = 0
    rows_out: int = 0
    prevalence: float = 0.0

    @property
    def removed_per_filter(self) -> list[int]:
        """Removed counts of the six population filters, for comparison with config."""
        return [s.removed for s in self.steps]


def select_declared_columns(raw: pd.DataFrame, mapping: VariableMapping) -> pd.DataFrame:
    """Keep only the PNS columns declared in the mapping (auxiliary + raw inference).

    Raises
    ------
    DataContractError
        If any declared column is absent from the raw frame.
    """
    wanted = list(mapping.auxiliary_pns_to_en) + list(mapping.raw_pns_to_en)
    missing = [c for c in wanted if c not in raw.columns]
    if missing:
        raise DataContractError(f"raw dataset is missing declared columns: {missing}")
    return raw.loc[:, wanted].copy()


def _step(frame: pd.DataFrame, mask: pd.Series, name: str, steps: list[FilterStep]) -> pd.DataFrame:
    kept = frame.loc[mask]
    removed = int(len(frame) - len(kept))
    steps.append(FilterStep(name=name, removed=removed, remaining=int(len(kept))))
    logger.info("population filter applied", extra={"filter": name, "removed": removed})
    return kept


def apply_population_filters(
    frame: pd.DataFrame,
    filters: PopulationFiltersConfig,
    target: TargetConfig,
) -> tuple[pd.DataFrame, FilterReport]:
    """Apply the six filters and build the binary target.

    Parameters
    ----------
    frame : pandas.DataFrame
        Raw frame with PNS column codes (upper case).
    filters : PopulationFiltersConfig
        Column names and codes of each filter.
    target : TargetConfig
        Blood pressure columns and thresholds.

    Returns
    -------
    tuple of (pandas.DataFrame, FilterReport)
        Filtered frame with a ``target`` column and index reset, plus the report.
    """
    steps: list[FilterStep] = []
    rows_in = int(len(frame))
    f = frame

    # 1. consent to store laboratory data
    f = _step(f, f[filters.consent_column] == filters.consent_keep_value, "lab_consent", steps)
    # 2. pregnant (or does not know)
    f = _step(
        f, ~f[filters.pregnancy_column].isin(filters.pregnancy_drop_values), "pregnant", steps
    )
    # 3. hypertension diagnosed only during pregnancy
    f = _step(
        f,
        f[filters.hypertension_pregnancy_only_column]
        != filters.hypertension_pregnancy_only_drop_value,
        "hypertension_pregnancy_only",
        steps,
    )
    # 4. took hypertension medication in the last two weeks
    f = _step(f, f[filters.medication_column] != filters.medication_drop_value, "medication", steps)

    # 5. region missing
    f = _step(f, f[filters.region_column].notna(), "region_missing", steps)
    # 6. proxy for an empty questionnaire
    f = _step(f, f[filters.empty_questionnaire_proxy_column].notna(), "empty_questionnaire", steps)

    # D1: blood pressure must exist before the target is defined. Filters 5 and 6
    # do not depend on the target, so they run first; this keeps the per-filter
    # counts of the notebook (which built the target between filters 4 and 5)
    # while making the exclusion explicit and counted.
    sbp = pd.to_numeric(f[target.systolic_column], errors="coerce")
    dbp = pd.to_numeric(f[target.diastolic_column], errors="coerce")
    bp_missing = sbp.isna() | dbp.isna()
    bp_removed = int(bp_missing.sum())
    if target.drop_if_blood_pressure_missing:
        f = f.loc[~bp_missing]
        sbp, dbp = sbp.loc[~bp_missing], dbp.loc[~bp_missing]
    logger.info("blood pressure missing check", extra={"removed": bp_removed})

    f = f.assign(
        **{
            TARGET_COLUMN: (
                (sbp >= target.systolic_threshold_mmhg) | (dbp >= target.diastolic_threshold_mmhg)
            ).astype(int)
        }
    )

    f = f.reset_index(drop=True)
    report = FilterReport(
        rows_in=rows_in,
        steps=steps,
        blood_pressure_missing_removed=bp_removed,
        rows_out=int(len(f)),
        prevalence=float(f[TARGET_COLUMN].mean()),
    )
    logger.info(
        "population defined",
        extra={"rows_out": report.rows_out, "prevalence": round(report.prevalence, 6)},
    )
    return f, report


def check_filter_report(
    report: FilterReport, filters: PopulationFiltersConfig, expected_rows: int
) -> None:
    """Compare the report with the counts frozen in the config.

    Raises
    ------
    DataContractError
        If any count differs. A difference is a refactoring bug, never noise.
    """
    if report.removed_per_filter != filters.expected_removed:
        raise DataContractError(
            f"filter counts {report.removed_per_filter} differ from expected "
            f"{filters.expected_removed}"
        )
    if report.rows_out != expected_rows:
        raise DataContractError(f"{report.rows_out} rows after filters, expected {expected_rows}")


def to_english_inference_frame(frame: pd.DataFrame, mapping: VariableMapping) -> pd.DataFrame:
    """Rename PNS codes to English names and keep the 61 raw inputs plus the target.

    Parameters
    ----------
    frame : pandas.DataFrame
        Output of :func:`apply_population_filters`.
    mapping : VariableMapping
        Name map.

    Returns
    -------
    pandas.DataFrame
        Columns: ``mapping.raw_en_names`` followed by ``target``.
    """
    renamed = frame.rename(columns=mapping.raw_pns_to_en)
    return renamed.loc[:, [*mapping.raw_en_names, TARGET_COLUMN]].copy()


def split_holdout(
    frame: pd.DataFrame, split: SplitConfig, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split identical to the notebook's ``train_test_split``.

    Parameters
    ----------
    frame : pandas.DataFrame
        Development frame with a ``target`` column and a fresh 0..n-1 index.
    split : SplitConfig
        ``test_size`` and stratification flag.
    seed : int
        ``random_state``.

    Returns
    -------
    tuple of (train, test)
        Both keep the original row order of the notebook split.
    """
    features = frame.drop(columns=[TARGET_COLUMN])
    y = frame[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        y,
        test_size=split.test_size,
        stratify=y if split.stratify else None,
        random_state=seed,
    )
    train = pd.concat([x_train, y_train], axis=1)
    test = pd.concat([x_test, y_test], axis=1)
    logger.info("holdout split", extra={"train_rows": len(train), "test_rows": len(test)})
    return train, test
