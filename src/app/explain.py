"""Presentation of the local explanation returned by ``/predict?explain=true``.

The API returns the top-k additive log-odds contributions of the active
one-hot columns (exact SHAP values of the linear champion, ADR 0003).
This module only translates dummy names into Portuguese labels and
shapes them for a bar chart; no model is loaded here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from src.data.mapping import VariableMapping

CLINICAL_DISCLAIMER_TITLE = "Apoio à decisão, não diagnóstico"


def dummy_labels(mapping: VariableMapping) -> dict[str, str]:
    """Map every one-hot column to ``"Rótulo da variável = categoria"``."""
    labels: dict[str, str] = {}
    for variable in mapping.model_variables:
        for dummy in variable.dummies:
            labels[dummy.en] = f"{variable.label_pt} = {dummy.category}"
    return labels


def contributions_frame(
    contributions: Iterable[Mapping[str, Any]], labels: Mapping[str, str]
) -> pd.DataFrame:
    """Rows ``fator, contribuicao, direcao`` sorted by absolute contribution.

    ``direcao`` is ``"aumenta o risco"`` for a positive log-odds
    contribution and ``"reduz o risco"`` otherwise.
    """
    rows = [
        {
            "fator": labels.get(str(c["feature"]), str(c["feature"])),
            "contribuicao": float(c["contribution"]),
            "direcao": "aumenta o risco" if float(c["contribution"]) > 0 else "reduz o risco",
        }
        for c in contributions
    ]
    frame = pd.DataFrame(rows, columns=["fator", "contribuicao", "direcao"])
    if frame.empty:
        return frame
    order = frame["contribuicao"].abs().sort_values(ascending=False).index
    return frame.loc[order].reset_index(drop=True)


def risk_band_colour(band: str, labels: Mapping[str, str]) -> str:
    """Streamlit status colour of a band label (``green``, ``orange`` or ``red``)."""
    if band == labels["low"]:
        return "green"
    if band == labels["high"]:
        return "red"
    return "orange"
