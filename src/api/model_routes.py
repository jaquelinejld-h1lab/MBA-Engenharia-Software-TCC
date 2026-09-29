"""Read-only routes with the global report of the model in service.

They exist so the Streamlit interface can show how the champion behaves without ever
loading the model or reading ``artifacts/``: every number here is read from the files
the training wrote next to the model (see :mod:`src.api.model_report`). None of them
runs inference, and none of them depends on request history, so they are cheap and safe
to call on every page load.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api import model_report
from src.api.model_loader import ModelCache
from src.api.schemas import (
    ModelContributionsResponse,
    ModelImportanceResponse,
    ModelRocResponse,
    ModelSubgroupsResponse,
    ModelSummaryResponse,
)
from src.config import Settings


def build_model_router(settings: Settings, model_cache: ModelCache) -> APIRouter:
    """Build the ``/model`` router bound to one settings object and model cache.

    Parameters
    ----------
    settings : Settings
        Validated configuration; supplies the decision threshold and the transport
        limits (points per ROC curve, number of coefficients).
    model_cache : ModelCache
        Cache holding the model in service; ``peek`` never triggers a load, so a report
        request cannot block on model loading.

    Returns
    -------
    fastapi.APIRouter
        Router with the five read-only endpoints under ``/model``.
    """
    router = APIRouter(prefix="/model", tags=["model"])

    @router.get("/summary", response_model=ModelSummaryResponse)
    async def model_summary() -> ModelSummaryResponse:
        """Return the metrics of the model in service: cross-validation k=10 and holdout."""
        loaded = model_cache.peek()
        run_dir = loaded.run_dir if loaded else None
        report = model_report.summary(run_dir)
        return ModelSummaryResponse(
            available=bool(report["available"]),
            reason=report.get("reason"),
            model_name=loaded.name if loaded else None,
            model_version=loaded.version if loaded else None,
            model_stage=loaded.stage if loaded else None,
            model_source=settings.api.model_source,
            model_run=run_dir.name if run_dir else None,
            decision_threshold=settings.model.decision_threshold,
            n_folds=int(report.get("n_folds", 0)),
            cross_validation=report.get("cross_validation", {}),
            holdout=report.get("holdout", {}),
        )

    @router.get("/roc", response_model=ModelRocResponse)
    async def model_roc() -> ModelRocResponse:
        """Return the ROC of every run trained with the champion, and the threshold sweep."""
        loaded = model_cache.peek()
        run_dir = loaded.run_dir if loaded else None
        champion = run_dir.name if run_dir else ""
        curves = model_report.roc(run_dir, champion, settings.api.roc_max_points)
        sweep = model_report.threshold_sweep(run_dir, settings.api.roc_max_points)
        return ModelRocResponse(
            available=bool(curves["available"]),
            reason=curves.get("reason"),
            curves=curves.get("curves", []),
            threshold_points=sweep.get("points", []),
            threshold_available=bool(sweep["available"]),
        )

    @router.get("/importance", response_model=ModelImportanceResponse)
    async def model_importance() -> ModelImportanceResponse:
        """Return the importance by model variable and the strongest coefficients."""
        loaded = model_cache.peek()
        report = model_report.importance(
            loaded.run_dir if loaded else None, settings.api.top_coefficients
        )
        return ModelImportanceResponse(**report)

    @router.get("/contributions", response_model=ModelContributionsResponse)
    async def model_contributions() -> ModelContributionsResponse:
        """Return the linear contributions over the holdout sample (ADR 0003: not SHAP)."""
        loaded = model_cache.peek()
        return ModelContributionsResponse(
            **model_report.contributions(loaded.run_dir if loaded else None)
        )

    @router.get("/subgroups", response_model=ModelSubgroupsResponse)
    async def model_subgroups() -> ModelSubgroupsResponse:
        """Return the out-of-fold performance by sex and age group."""
        loaded = model_cache.peek()
        return ModelSubgroupsResponse(**model_report.subgroups(loaded.run_dir if loaded else None))

    return router
