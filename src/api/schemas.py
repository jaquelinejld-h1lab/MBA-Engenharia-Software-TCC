"""Request and response models of the API, generated from ``variable_mapping.yaml``.

``PredictionRequest`` carries the 61 raw variables in English names. Domain
validation (categorical codes and plausible numeric ranges, inventory item
D8) comes from the mapping through ``RawRecord``; laboratory exams are
optional and are imputed by the persisted KNN imputer (decision P3).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.data.contract import RawRecord
from src.data.mapping import get_mapping


def _example_record() -> dict[str, Any]:
    """Build a plausible in-domain example (mode or median of the training set)."""
    example: dict[str, Any] = {}
    for variable in get_mapping().raw_variables:
        observed = variable.observed
        if variable.domain.type == "categorical":
            value: float | None = observed.mode
        else:
            value = observed.median
        if value is None:
            value = (
                (variable.domain.values or [1])[0]
                if variable.domain.type == "categorical"
                else float(variable.domain.min or 0)
            )
        example[variable.en] = int(value) if variable.dtype == "int" else float(value)
    return example


class PredictionRequest(RawRecord):  # type: ignore[misc, valid-type]
    """One observation: 61 raw variables (laboratory exams optional)."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, json_schema_extra={"examples": [_example_record()]}
    )


# Response models expose model_name, model_version, model_stage and model_source:
# pydantic < 2.10 reserves the model_ namespace and would warn (error under pytest).
_RESPONSE_CONFIG = ConfigDict(protected_namespaces=())


class Contribution(BaseModel):
    """Additive log-odds contribution of one active dummy (linear model)."""

    feature: str
    value: int
    contribution: float


class PredictionResponse(BaseModel):
    """Prediction for one observation."""

    model_config = _RESPONSE_CONFIG

    probability: float = Field(ge=0.0, le=1.0)
    risk_band: str
    predicted_class: int = Field(ge=0, le=1)
    threshold: float
    model_name: str
    model_version: str
    model_stage: str
    imputed_fields: list[str]
    contributions: list[Contribution] | None = None
    disclaimer: str
    correlation_id: str


class BatchPredictionRequest(BaseModel):
    """Several observations scored together."""

    records: list[PredictionRequest] = Field(min_length=1)


class BatchPredictionResponse(BaseModel):
    """Batch result, one entry per input row in the same order."""

    predictions: list[PredictionResponse]
    n: int
    n_imputed_rows: int
    correlation_id: str


class HealthResponse(BaseModel):
    """Liveness and readiness information."""

    model_config = _RESPONSE_CONFIG

    status: str
    model_loaded: bool
    model_name: str | None
    model_version: str | None
    model_stage: str | None
    model_source: str
    package_version: str


class ErrorResponse(BaseModel):
    """Error envelope: never carries a stack trace."""

    error: str
    detail: str
    correlation_id: str


# --------------------------------------------------------------- global model report
class _ReportBase(BaseModel):
    """Every model report carries whether it could be served, and why not."""

    model_config = _RESPONSE_CONFIG

    available: bool
    reason: str | None = None


class ModelSummaryResponse(_ReportBase):
    """Identity of the model in service plus its validation and holdout metrics."""

    model_name: str | None
    model_version: str | None
    model_stage: str | None
    model_source: str
    model_run: str | None
    decision_threshold: float
    n_folds: int = 0
    cross_validation: dict[str, float] = Field(default_factory=dict)
    holdout: dict[str, float] = Field(default_factory=dict)


class RocCurve(BaseModel):
    """One run's ROC on the holdout split, thinned for transport."""

    run: str
    is_champion: bool
    auc_holdout: float | None
    auc_cv_mean: float | None
    fpr: list[float]
    tpr: list[float]


class ModelRocResponse(_ReportBase):
    """ROC of every run trained alongside the champion, plus the threshold sweep."""

    curves: list[RocCurve] = Field(default_factory=list)
    threshold_points: list[dict[str, float]] = Field(default_factory=list)
    threshold_available: bool = False


class VariableImportanceEntry(BaseModel):
    """One model variable with its dummies aggregated."""

    model_config = _RESPONSE_CONFIG

    model_variable: str
    label_pt: str
    source_raw: list[str] = Field(default_factory=list)
    mean_abs_contribution: float
    max_abs_coef: float


class CoefficientEntry(BaseModel):
    """One dummy of the design matrix and its coefficient, with sign."""

    model_config = _RESPONSE_CONFIG

    dummy: str
    model_variable: str
    label_pt: str
    category: str
    coefficient: float
    mean_abs_contribution: float


class ModelImportanceResponse(_ReportBase):
    """Global reading of the champion: importance by variable and coefficients."""

    method: str = ""
    variables: list[VariableImportanceEntry] = Field(default_factory=list)
    coefficients: list[CoefficientEntry] = Field(default_factory=list)


class ModelContributionsResponse(_ReportBase):
    """Linear contributions over the holdout sample (ADR 0003: this is not SHAP).

    Columnar by design: ``contributions[i][j]`` is the contribution of ``dummies[j]``
    in observation ``i``, and ``present[i][j]`` says whether that dummy was active,
    which is what colours the beeswarm.
    """

    method: str = ""
    rows: int = 0
    rows_available: int = 0
    dummies: list[str] = Field(default_factory=list)
    labels_pt: list[str] = Field(default_factory=list)
    contributions: list[list[float]] = Field(default_factory=list)
    present: list[list[int]] = Field(default_factory=list)


class SubgroupEntry(BaseModel):
    """Out-of-fold metrics of one subgroup; metrics are null below the minimum size."""

    dimension: str
    group: str
    n: int
    prevalence: float | None = None
    auc_roc: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    precision: float | None = None
    f1: float | None = None


class ModelSubgroupsResponse(_ReportBase):
    """Performance by sex and age group, for the equity section of the model card."""

    rows: list[SubgroupEntry] = Field(default_factory=list)
