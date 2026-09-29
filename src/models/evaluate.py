"""Classification metrics, computed once and never rounded before persistence.

Plotting lives in ``src/models/plots.py`` (correction of inventory item R8):
this module returns plain numbers and curve coordinates only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.exceptions import PredictionError


@dataclass(frozen=True)
class ClassificationMetrics:
    """Threshold-dependent and threshold-free metrics of a binary classifier."""

    auc_roc: float
    accuracy: float
    precision: float
    sensitivity: float
    specificity: float
    f1: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    n: int
    positives: int
    threshold: float

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping, suitable for MLflow ``log_metrics``."""
        return {k: v for k, v in asdict(self).items()}


@dataclass(frozen=True)
class RocPoints:
    """ROC curve coordinates."""

    fpr: list[float] = field(default_factory=list)
    tpr: list[float] = field(default_factory=list)
    thresholds: list[float] = field(default_factory=list)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    y_pred: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Compute every metric from labels and positive-class probabilities.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    y_prob : array-like of float
        Probability of the positive class (used for AUC and, when ``y_pred`` is
        not given, for the decision rule ``y_prob >= threshold``).
    threshold : float
        Decision threshold, recorded in the result.
    y_pred : array-like of {0, 1}, optional
        Hard predictions from ``estimator.predict``. The original notebook
        computed accuracy, precision, recall and F1 from ``predict``; for
        logistic regression this equals ``y_prob >= 0.5``, but for ``SVC`` the
        decision function and the Platt probabilities disagree, so the
        experiment runner passes ``predict`` output to stay comparable.

    Returns
    -------
    ClassificationMetrics

    Raises
    ------
    PredictionError
        If the inputs are empty, misaligned or contain a single class.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if y.size == 0 or y.shape != p.shape:
        raise PredictionError(f"labels and probabilities misaligned: {y.shape} vs {p.shape}")
    if len(np.unique(y)) < 2:  # noqa: PLR2004
        raise PredictionError("metrics need both classes present in y_true")
    if y_pred is None:
        y_pred = (p >= threshold).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_pred.shape != y.shape:
        raise PredictionError(f"labels and predictions misaligned: {y.shape} vs {y_pred.shape}")
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, y_pred, labels=[0, 1]).ravel())
    return ClassificationMetrics(
        auc_roc=float(roc_auc_score(y, p)),
        accuracy=float(accuracy_score(y, y_pred)),
        precision=float(precision_score(y, y_pred, zero_division=0)),
        sensitivity=float(recall_score(y, y_pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) else 0.0,
        f1=float(f1_score(y, y_pred, zero_division=0)),
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        n=int(y.size),
        positives=int(y.sum()),
        threshold=float(threshold),
    )


def roc_points(y_true: np.ndarray | pd.Series, y_prob: np.ndarray) -> RocPoints:
    """ROC curve coordinates for later plotting."""
    fpr, tpr, thresholds = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))
    return RocPoints(
        fpr=[float(v) for v in fpr],
        tpr=[float(v) for v in tpr],
        thresholds=[float(v) for v in thresholds],
    )


def summarize_folds(fold_metrics: list[ClassificationMetrics]) -> dict[str, float]:
    """Mean and population standard deviation of each metric over folds.

    Uses ``numpy.std`` with ``ddof=0``, as the original notebook did.
    """
    if not fold_metrics:
        raise PredictionError("no folds to summarize")
    keys = ("auc_roc", "accuracy", "precision", "sensitivity", "specificity", "f1")
    summary: dict[str, float] = {}
    for key in keys:
        values = np.asarray([getattr(m, key) for m in fold_metrics], dtype=float)
        summary[f"{key}_mean"] = float(values.mean())
        summary[f"{key}_std"] = float(values.std(ddof=0))
    summary["n_folds"] = float(len(fold_metrics))
    return summary


def subgroup_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    groups: pd.Series,
    threshold: float,
    *,
    min_size: int,
) -> pd.DataFrame:
    """Metrics per subgroup (sex, age group, ...).

    Parameters
    ----------
    groups : pandas.Series
        Group label per observation, aligned with ``y_true``.
    min_size : int
        Groups smaller than this are reported with ``NaN`` metrics.

    Returns
    -------
    pandas.DataFrame
        One row per group with size, prevalence and the metrics; unrounded.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    rows: list[dict[str, Any]] = []
    for label in sorted(pd.unique(groups)):
        mask = np.asarray(groups == label)
        row: dict[str, Any] = {
            "group": str(label),
            "n": int(mask.sum()),
            "prevalence": float(y[mask].mean()),
        }
        if mask.sum() >= min_size and len(np.unique(y[mask])) == 2:  # noqa: PLR2004
            m = compute_metrics(y[mask], p[mask], threshold)
            row.update(
                auc_roc=m.auc_roc,
                sensitivity=m.sensitivity,
                specificity=m.specificity,
                precision=m.precision,
                f1=m.f1,
            )
        else:
            row.update(
                auc_roc=np.nan, sensitivity=np.nan, specificity=np.nan, precision=np.nan, f1=np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
