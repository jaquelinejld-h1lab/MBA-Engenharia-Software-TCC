"""Synthetic raw fixture with the PNS 2013 extract schema.

The generator samples every declared variable independently from its domain
in ``variable_mapping.yaml`` (uniformly for categorical codes, uniformly over
the declared range for numeric ones) and injects missing values at the
declared observed share. It never reads the real dataset, so the fixture
carries no information about real respondents and can be committed freely.

Distributions are deliberately not realistic: the fixture exists to exercise
schema, filters, derivations and encoding paths, not to reproduce metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.mapping import VariableMapping, get_mapping

DEFAULT_ROWS = 600

# Probabilities of the filter codes, chosen so that roughly half of the rows
# survive the population filters (values in the order declared in the mapping).
_AUX_PROBABILITIES: dict[str, list[float]] = {
    "Z051": [0.85, 0.05, 0.10],  # consent: 1 kept
    "P005": [0.05, 0.90, 0.05],  # pregnancy: 1 and 3 dropped
    "Q002": [0.20, 0.05, 0.75],  # hypertension only in pregnancy: 2 dropped
    "Q006": [0.15, 0.85],  # medication: 1 dropped
}
# Blood pressure sampled around normotensive values so both target classes occur.
_BLOOD_PRESSURE_MOMENTS: dict[str, tuple[float, float]] = {
    "W00407": (122.0, 16.0),
    "W00408": (78.0, 11.0),
}
# Share of missing values injected into the columns used by filters 5 and 6, so
# that those filters remove something on the fixture as well.
_FILTER_NULL_SHARE = 0.01


def make_synthetic_frame(
    seed: int, rows: int = DEFAULT_ROWS, mapping: VariableMapping | None = None
) -> pd.DataFrame:
    """Build a deterministic raw frame with PNS column codes.

    Parameters
    ----------
    seed : int
        Random seed; the same seed always yields the same frame.
    rows : int
        Number of rows before population filters.
    mapping : VariableMapping, optional
        Defaults to the project mapping.

    Returns
    -------
    pandas.DataFrame
        Columns: the auxiliary (filter/target) codes followed by the 61 raw codes.
    """
    mapping = mapping or get_mapping()
    rng = np.random.default_rng(seed)
    columns: dict[str, np.ndarray] = {}

    for aux in mapping.auxiliary_variables:
        dom = aux.domain
        if dom.type == "categorical":
            values = np.asarray(dom.values, dtype=float)
            probs = _AUX_PROBABILITIES[aux.pns]
            columns[aux.pns] = rng.choice(values, size=rows, p=probs)
        else:
            mean, sd = _BLOOD_PRESSURE_MOMENTS[aux.pns]
            sample = rng.normal(mean, sd, size=rows)
            columns[aux.pns] = np.clip(sample, float(dom.min or 0), float(dom.max or 1))

    for var in mapping.raw_variables:
        dom = var.domain
        if dom.type == "categorical":
            sample = rng.choice(np.asarray(dom.values, dtype=float), size=rows)
        elif var.dtype == "int":
            sample = rng.integers(int(dom.min or 0), int(dom.max or 1) + 1, size=rows).astype(float)
        else:
            sample = np.round(rng.uniform(float(dom.min or 0), float(dom.max or 1), size=rows), 2)
        null_share = var.null_share_observed if var.nullable else 0.0
        if var.pns in ("REGIAO", "P006"):
            null_share = _FILTER_NULL_SHARE
        if null_share > 0:
            mask = rng.random(rows) < null_share
            sample = sample.copy()
            sample[mask] = np.nan
        columns[var.pns] = sample

    frame = pd.DataFrame(columns)
    # Keep blood pressure plausible and integer-like, as in the survey.
    for pns in ("W00407", "W00408"):
        frame[pns] = frame[pns].round(0)
    return frame
