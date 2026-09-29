"""Prediction service: framework-free core used by the FastAPI routes.

Everything the API answers is computed here from the loaded pipeline, so the
same code path is testable without HTTP and the online/offline parity test
compares this service with a plain ``pipeline.predict_proba``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.api.model_loader import LoadedModel
from src.config import Settings
from src.data.mapping import VariableMapping, get_mapping
from src.exceptions import PredictionError

_LAB_FIELDS: tuple[str, ...] = ("egfr_afro", "cholesterol", "glucose")


@dataclass(frozen=True)
class Prediction:
    """One scored observation."""

    probability: float
    risk_band: str
    predicted_class: int
    imputed_fields: list[str]
    contributions: list[dict[str, Any]] | None


class PredictionService:
    """Score raw records with the promoted pipeline.

    Parameters
    ----------
    cfg : Settings
        Threshold, risk bands and API limits.
    model : LoadedModel
        Pipeline plus registry metadata.
    mapping : VariableMapping, optional
        Defaults to the project mapping.
    """

    def __init__(
        self, cfg: Settings, model: LoadedModel, mapping: VariableMapping | None = None
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.mapping = mapping or get_mapping()
        self.threshold = cfg.model.decision_threshold

    # ------------------------------------------------------------ helpers
    def risk_band(self, probability: float) -> str:
        """Map a probability to the configured low/moderate/high label."""
        bands, labels = self.cfg.model.risk_bands, self.cfg.api.risk_band_labels
        if probability < bands.low_below:
            return labels.low
        if probability >= bands.high_from:
            return labels.high
        return labels.moderate

    def records_to_frame(self, records: list[dict[str, Any]]) -> pd.DataFrame:
        """Convert validated payloads to a frame with the 61 raw columns in mapping order."""
        if not records:
            raise PredictionError("no records to score")
        frame = pd.DataFrame.from_records(records)
        missing = [c for c in self.mapping.raw_en_names if c not in frame.columns]
        for column in missing:
            frame[column] = np.nan
        frame = frame[self.mapping.raw_en_names].astype(float)
        return frame

    # ---------------------------------------------------------- scoring
    def predict_frame(self, frame: pd.DataFrame, *, explain: bool = False) -> list[Prediction]:
        """Score every row of ``frame`` (raw variables in English names)."""
        if len(frame) > self.cfg.api.batch_max_records:
            raise PredictionError(
                f"batch has {len(frame)} rows; limit is {self.cfg.api.batch_max_records}"
            )
        result = self.model.transformer.transform_detailed(frame)
        probabilities = self.model.pipeline.named_steps["model"].predict_proba(result.encoded)[:, 1]
        predictions: list[Prediction] = []
        for position, probability in enumerate(probabilities):
            p = float(probability)
            predictions.append(
                Prediction(
                    probability=p,
                    risk_band=self.risk_band(p),
                    predicted_class=int(p >= self.threshold),
                    imputed_fields=result.imputed_fields(position),
                    contributions=(
                        self._contributions(result.encoded.iloc[position]) if explain else None
                    ),
                )
            )
        return predictions

    def predict_records(
        self, records: list[dict[str, Any]], *, explain: bool = False
    ) -> list[Prediction]:
        """Score validated dictionaries."""
        return self.predict_frame(self.records_to_frame(records), explain=explain)

    # ---------------------------------------------------- explainability
    def _contributions(self, encoded_row: pd.Series) -> list[dict[str, Any]]:
        """Top-k additive log-odds contributions of the active dummies.

        For the logistic regression champion the log-odds are exactly
        ``intercept + sum(coef_j * x_j)``, so ``coef_j * x_j`` is an exact
        additive attribution (equal to SHAP values of a linear model with a
        zero baseline). Non-linear estimators return an empty list; SHAP for
        them is out of scope (ADR 0003).
        """
        estimator = self.model.pipeline.named_steps["model"]
        coefficients = getattr(estimator, "coef_", None)
        if coefficients is None:
            return []
        weights = np.asarray(coefficients).ravel()
        values = encoded_row.to_numpy(dtype=float)
        contributions = weights * values
        active: list[dict[str, Any]] = [
            {"feature": str(name), "value": int(value), "contribution": float(contribution)}
            for name, value, contribution in zip(
                encoded_row.index, values, contributions, strict=True
            )
            if value != 0
        ]
        active.sort(key=lambda item: abs(float(item["contribution"])), reverse=True)
        return active[: self.cfg.api.top_contributions]
