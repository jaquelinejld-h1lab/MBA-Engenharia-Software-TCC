"""Stratified k-fold evaluation over the full development set (decision P6).

Two protocols, matching the two transformer modes:

* ``fidelity``: the transformer is applied once to the whole development
  frame (imputers fitted on all rows, exactly as ``evaluate_models_cv`` in the
  original notebook) and the folds split the encoded matrix.
* ``production``: the transformer is fitted on each fold's training rows and
  applied to the held-out rows, so no information from the evaluation fold
  reaches the imputers.

Both return out-of-fold probabilities for every development row, which the
subgroup analysis consumes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold

from src.config import SplitConfig
from src.data.filters import TARGET_COLUMN
from src.features.transformer import HypertensionTransformer
from src.logging_setup import get_logger
from src.models.evaluate import ClassificationMetrics, compute_metrics, summarize_folds

logger = get_logger(__name__)


@dataclass(frozen=True)
class CrossValidationResult:
    """Fold metrics, their summary and the out-of-fold probabilities."""

    fold_metrics: list[ClassificationMetrics]
    summary: dict[str, float]
    oof_probabilities: np.ndarray
    fold_of_row: np.ndarray = field(repr=False)


def stratified_folds(split: SplitConfig, seed: int) -> StratifiedKFold:
    """Build the project fold generator: stratified, shuffled, seeded."""
    return StratifiedKFold(n_splits=split.n_splits, shuffle=True, random_state=seed)


def cross_validate(
    development: pd.DataFrame,
    transformer: HypertensionTransformer,
    estimator: ClassifierMixin,
    split: SplitConfig,
    seed: int,
    threshold: float,
) -> CrossValidationResult:
    """Run the k-fold protocol selected by ``transformer.mode``.

    Parameters
    ----------
    development : pandas.DataFrame
        61 raw variables in English names plus ``target``.
    transformer : HypertensionTransformer
        Unfitted transformer; its ``mode`` selects the protocol.
    estimator : ClassifierMixin
        Unfitted estimator; cloned per fold.
    split : SplitConfig
        ``n_splits``.
    seed : int
        Fold shuffling seed.
    threshold : float
        Decision threshold for the threshold metrics.
    """
    features = development.drop(columns=[TARGET_COLUMN])
    y = development[TARGET_COLUMN].to_numpy()
    folds = stratified_folds(split, seed)
    oof = np.full(len(y), np.nan, dtype=float)
    fold_of_row = np.full(len(y), -1, dtype=int)
    fold_metrics: list[ClassificationMetrics] = []

    encoded_all: pd.DataFrame | None = None
    if transformer.mode == "fidelity":
        # Notebook protocol: transform the full development frame once.
        encoded_all = clone(transformer).fit(features).transform(features)

    for k, (train_idx, test_idx) in enumerate(folds.split(features, y)):
        if encoded_all is not None:
            x_train, x_test = encoded_all.iloc[train_idx], encoded_all.iloc[test_idx]
        else:
            fold_transformer = clone(transformer).fit(features.iloc[train_idx])
            x_train = fold_transformer.transform(features.iloc[train_idx])
            x_test = fold_transformer.transform(features.iloc[test_idx])
        model = clone(estimator).fit(x_train, y[train_idx])
        prob = model.predict_proba(x_test)[:, 1]
        oof[test_idx] = prob
        fold_of_row[test_idx] = k
        # Label metrics from ``predict`` (notebook semantics; see compute_metrics).
        metrics = compute_metrics(y[test_idx], prob, threshold, y_pred=model.predict(x_test))
        fold_metrics.append(metrics)
        logger.info(
            "fold evaluated",
            extra={"fold": k, "auc_roc": metrics.auc_roc, "sensitivity": metrics.sensitivity},
        )

    return CrossValidationResult(
        fold_metrics=fold_metrics,
        summary=summarize_folds(fold_metrics),
        oof_probabilities=oof,
        fold_of_row=fold_of_row,
    )
