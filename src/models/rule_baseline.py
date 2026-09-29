"""Clinical rule baseline: how far a count of risk factors gets without learning.

A model is only worth its complexity if it beats something trivial. The trivial thing
here is the rule a clinician could apply with a tape measure and three questions: count
how many classical hypertension risk factors the person has, and treat more factors as
more risk. It learns nothing from the data, so it also cannot overfit it.

The factors are declared in ``configs/config.yaml`` under ``rule_baseline``, each as a
model variable and the categories that count as present, which keeps the clinical
criteria visible and reviewable instead of buried in code. The score is the share of
factors present, which is what allows an AUC to be computed at all, and the predicted
label is "at risk" from ``positive_from`` factors upwards.

The comparison is fair by construction: same development frame, same holdout split and
the same metric function as every trained run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import RuleBaselineConfig
from src.exceptions import TransformError


@dataclass(frozen=True)
class RuleBaselineResult:
    """Score and label of the rule over one set of observations."""

    scores: np.ndarray
    labels: np.ndarray
    factor_counts: pd.Series
    factor_names: list[str]


def apply_rule(model_frame: pd.DataFrame, cfg: RuleBaselineConfig) -> RuleBaselineResult:
    """Count the declared risk factors of every observation.

    Parameters
    ----------
    model_frame : pandas.DataFrame
        The 33 model variables as categorical strings, exactly what the transformer
        produces before one-hot encoding.
    cfg : RuleBaselineConfig
        Declared factors and the number of factors from which the rule predicts risk.

    Returns
    -------
    RuleBaselineResult
        ``scores`` in [0, 1] (share of factors present), ``labels`` in {0, 1} and the
        raw count per observation.

    Raises
    ------
    TransformError
        If a declared factor names a variable the frame does not carry, which means the
        configuration and the mapping drifted apart.
    """
    missing = [factor.variable for factor in cfg.factors if factor.variable not in model_frame]
    if missing:
        raise TransformError(f"rule baseline refers to unknown model variables: {missing}")

    counts = pd.Series(0, index=model_frame.index, dtype=int)
    for factor in cfg.factors:
        present = model_frame[factor.variable].astype(str).isin(factor.categories)
        counts = counts + present.astype(int)

    total = len(cfg.factors)
    scores = counts.to_numpy(dtype=float) / float(total)
    labels = (counts.to_numpy(dtype=int) >= cfg.positive_from).astype(int)
    return RuleBaselineResult(
        scores=scores,
        labels=labels,
        factor_counts=counts,
        factor_names=[factor.variable for factor in cfg.factors],
    )


def describe(cfg: RuleBaselineConfig) -> str:
    """One readable line per declared factor, for the evidence file."""
    lines = [
        f"Risco quando ao menos {cfg.positive_from} dos {len(cfg.factors)} fatores estao presentes:"
    ]
    lines.extend(
        f"- {factor.label_pt} (`{factor.variable}` em {factor.categories})"
        for factor in cfg.factors
    )
    return "\n".join(lines)
