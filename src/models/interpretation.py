"""Global reading of a linear model: coefficients, importance and contributions.

The champion is a logistic regression, so the contribution of a dummy ``j`` to the
log-odds of an observation is exactly ``coef_j * x_ij``. That is the same quantity the
API returns per prediction (ADR 0003, linear contributions instead of SHAP), aggregated
here over a set of observations:

* **coefficients**: one per dummy, with sign, read straight from the fitted estimator;
* **importance**: ``mean |coef_j * x_ij|`` over the training rows, summed over the
  dummies of each model variable, which puts the 33 variables on one scale;
* **contributions**: the matrix ``coef_j * x_ij`` over holdout rows for the dummies that
  move the log-odds the most, which is what the interface draws as a beeswarm.

Everything is computed once at training time and written next to the model, so the API
can serve it without touching training data and the interface stays a pure HTTP client.
Models without ``coef_`` (random forest, MLP, SVM) produce no interpretation file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.data.mapping import VariableMapping

INTERPRETATION_FILENAME = "interpretation.json"
SUBGROUP_FILENAME = "subgroup_metrics.json"
_BINARY_COEF_NDIM = 2
METHOD = (
    "coef_j * x_ij on the fitted logistic regression: contribution of each dummy to the "
    "log-odds. Importance is mean |coef_j * x_ij| over the training rows, summed over the "
    "dummies of each model variable. Not SHAP (ADR 0003)."
)


@dataclass(frozen=True)
class Coefficient:
    """One dummy of the design matrix and its coefficient."""

    dummy: str
    model_variable: str
    label_pt: str
    category: str
    coefficient: float
    mean_abs_contribution: float


@dataclass(frozen=True)
class VariableImportance:
    """One model variable, with its dummies aggregated."""

    model_variable: str
    label_pt: str
    source_raw: list[str]
    mean_abs_contribution: float
    max_abs_coef: float


@dataclass(frozen=True)
class ContributionSample:
    """Contributions of the strongest dummies over a sample of observations."""

    dummies: list[str]
    labels_pt: list[str]
    contributions: list[list[float]]
    present: list[list[int]]
    rows: int
    rows_available: int


def _dummy_index(mapping: VariableMapping) -> dict[str, tuple[str, str, str]]:
    """Map each dummy name to ``(model variable, label, category)``."""
    index: dict[str, tuple[str, str, str]] = {}
    for variable in mapping.model_variables:
        for dummy in variable.dummies:
            index[dummy.en] = (variable.en, variable.label_pt, dummy.category)
    return index


def _coefficients(estimator: object) -> np.ndarray | None:
    """Coefficient vector of a linear binary classifier, or ``None`` if it has none.

    scikit-learn stores ``coef_`` with one row per class, and a binary classifier has a
    single row; anything else (multiclass, or an estimator without coefficients) has no
    one-dimensional reading and yields ``None``.
    """
    raw = getattr(estimator, "coef_", None)
    if raw is None:
        return None
    array = np.asarray(raw, dtype=float)
    if array.ndim == _BINARY_COEF_NDIM and array.shape[0] == 1:
        return np.asarray(array[0], dtype=float)
    if array.ndim == 1:
        return array
    return None


def build_interpretation(
    estimator: object,
    encoded_train: pd.DataFrame,
    encoded_holdout: pd.DataFrame,
    mapping: VariableMapping,
    *,
    top_dummies: int,
    max_rows: int,
) -> dict[str, Any] | None:
    """Compute the global reading of a fitted linear model.

    Parameters
    ----------
    estimator : object
        Fitted estimator; anything without ``coef_`` yields ``None``.
    encoded_train : pandas.DataFrame
        Design matrix of the training split, columns aligned with the coefficients.
    encoded_holdout : pandas.DataFrame
        Design matrix of the holdout split, used for the contribution sample.
    mapping : VariableMapping
        Project mapping, for the variable each dummy belongs to and its label.
    top_dummies : int
        How many dummies enter the contribution sample, ranked by mean absolute
        contribution on the holdout.
    max_rows : int
        Upper bound on the number of observations kept in the sample. When the holdout
        is larger, rows are taken at a fixed stride so the sample stays deterministic
        and spread over the whole split.

    Returns
    -------
    dict or None
        JSON-ready payload, or ``None`` for a model without coefficients.

    Raises
    ------
    ValueError
        If the coefficient vector does not match the design matrix columns.
    """
    coefficients = _coefficients(estimator)
    if coefficients is None:
        return None
    if coefficients.shape[0] != encoded_train.shape[1]:
        raise ValueError(
            f"{coefficients.shape[0]} coefficients for {encoded_train.shape[1]} columns"
        )

    index = _dummy_index(mapping)
    columns = [str(column) for column in encoded_train.columns]
    train_values = encoded_train.to_numpy(dtype=float)
    train_contributions = np.abs(train_values * coefficients)
    mean_abs_train = train_contributions.mean(axis=0)

    coefficient_rows = []
    for position, column in enumerate(columns):
        variable, label, category = index.get(column, (column, column, ""))
        coefficient_rows.append(
            Coefficient(
                dummy=column,
                model_variable=variable,
                label_pt=label,
                category=category,
                coefficient=float(coefficients[position]),
                mean_abs_contribution=float(mean_abs_train[position]),
            )
        )

    importance = _aggregate(coefficient_rows, mapping)
    sample = _contribution_sample(
        coefficients, encoded_holdout, columns, index, top_dummies=top_dummies, max_rows=max_rows
    )
    intercept = np.asarray(getattr(estimator, "intercept_", [0.0]), dtype=float)
    return {
        "method": METHOD,
        "intercept": float(intercept.reshape(-1)[0]),
        "n_train": int(encoded_train.shape[0]),
        "coefficients": [asdict(row) for row in coefficient_rows],
        "importance": [asdict(row) for row in importance],
        "contributions": asdict(sample),
    }


def _aggregate(
    coefficients: list[Coefficient], mapping: VariableMapping
) -> list[VariableImportance]:
    """Sum the dummy contributions of each model variable, strongest first."""
    by_variable = {variable.en: variable for variable in mapping.model_variables}
    totals: dict[str, list[Coefficient]] = {}
    for entry in coefficients:
        totals.setdefault(entry.model_variable, []).append(entry)

    rows = []
    for name, entries in totals.items():
        variable = by_variable.get(name)
        rows.append(
            VariableImportance(
                model_variable=name,
                label_pt=variable.label_pt if variable else name,
                source_raw=list(variable.source_raw) if variable else [],
                mean_abs_contribution=float(sum(e.mean_abs_contribution for e in entries)),
                max_abs_coef=float(max(abs(e.coefficient) for e in entries)),
            )
        )
    return sorted(rows, key=lambda row: row.mean_abs_contribution, reverse=True)


def _contribution_sample(
    coefficients: np.ndarray,
    encoded_holdout: pd.DataFrame,
    columns: list[str],
    index: dict[str, tuple[str, str, str]],
    *,
    top_dummies: int,
    max_rows: int,
) -> ContributionSample:
    """Contributions of the strongest dummies over a deterministic row sample."""
    values = encoded_holdout.to_numpy(dtype=float)
    contributions = values * coefficients
    ranking = np.argsort(np.abs(contributions).mean(axis=0))[::-1][:top_dummies]

    available = int(values.shape[0])
    stride = max(1, -(-available // max_rows)) if max_rows > 0 else 1
    selected_rows = np.arange(0, available, stride)

    picked = contributions[np.ix_(selected_rows, ranking)]
    present = values[np.ix_(selected_rows, ranking)]
    dummies = [columns[position] for position in ranking]
    return ContributionSample(
        dummies=dummies,
        labels_pt=[_dummy_label(dummy, index) for dummy in dummies],
        # Four decimals: the sample feeds a chart, and the full precision would
        # multiply the payload without moving a single pixel.
        contributions=[[round(float(value), 4) for value in row] for row in picked],
        present=[[int(value != 0.0) for value in row] for row in present],
        rows=len(selected_rows),
        rows_available=available,
    )


def _dummy_label(dummy: str, index: dict[str, tuple[str, str, str]]) -> str:
    """Readable label of a dummy: variable label plus its category."""
    variable, label, category = index.get(dummy, (dummy, dummy, ""))
    del variable
    return f"{label} = {category}" if category else label
