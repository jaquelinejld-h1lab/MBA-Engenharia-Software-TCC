"""FastAPI application: ``/predict``, ``/predict/batch``, ``/health``, ``/metrics``.

Run locally::

    uvicorn src.api.main:app --host 0.0.0.0 --port 8000

The model is loaded once at startup (lifespan) from the source selected in
``configs/config.yaml`` (``api.model_source``); a missing model makes the
process fail fast with the reason in the log. ``create_app`` accepts an
already loaded model so tests can inject a pipeline trained on the fixture.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response, status

from src import __version__
from src.api.auth import UNAUTHORIZED_DETAIL, is_authorized, warn_if_open
from src.api.metrics import ApiMetrics
from src.api.middleware import (
    RequestContextMiddleware,
    correlation_id_of,
    register_exception_handlers,
)
from src.api.model_loader import MODEL_CACHE, LoadedModel, ModelCache, load_drift_reference
from src.api.model_routes import build_model_router
from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.api.service import Prediction, PredictionService
from src.config import Settings, get_config
from src.logging_setup import configure_logging, get_logger
from src.monitoring.drift import ReferenceWindow
from src.monitoring.window import DriftMonitor

logger = get_logger(__name__)

_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse},
    503: {"model": ErrorResponse},
    500: {"model": ErrorResponse},
}


def create_app(
    cfg: Settings | None = None,
    *,
    model: LoadedModel | None = None,
    cache: ModelCache | None = None,
    drift_reference: ReferenceWindow | None = None,
) -> FastAPI:
    """Build the application.

    Parameters
    ----------
    cfg : Settings, optional
        Defaults to the project configuration.
    model : LoadedModel, optional
        Pre-loaded model (tests); when omitted the lifespan loads it.
    cache : ModelCache, optional
        Cache instance; defaults to the module-level one.
    drift_reference : ReferenceWindow, optional
        Training reference for drift (tests); when omitted the lifespan
        loads it from the model source.
    """
    settings = cfg or get_config()
    model_cache = cache or MODEL_CACHE
    if model is not None:
        model_cache.set(model)
    monitor = DriftMonitor(settings.monitoring, drift_reference)
    metrics = ApiMetrics(
        drift_provider=monitor.snapshot,
        retraining=settings.monitoring.retraining,
        min_rows_for_drift=settings.monitoring.min_rows_for_drift,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.logging)
        loaded = model_cache.get(settings)  # raises ModelNotFoundError: fail fast
        if monitor.reference is None:
            monitor.reference = load_drift_reference(settings, loaded)
        logger.info(
            "api ready",
            extra={
                "model": loaded.name,
                "version": loaded.version,
                "stage": loaded.stage,
                "drift_reference": monitor.reference is not None,
            },
        )
        yield

    app = FastAPI(
        title="Hypertension risk API",
        version=__version__,
        description=(
            "Produtização do modelo de risco de hipertensão (Dias, 2024). "
            + settings.api.clinical_disclaimer
        ),
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    register_exception_handlers(app, metrics)

    def get_service() -> PredictionService:
        return PredictionService(settings, model_cache.get(settings))

    def to_response(
        prediction: Prediction, loaded: LoadedModel, correlation_id: str
    ) -> PredictionResponse:
        return PredictionResponse(
            probability=prediction.probability,
            risk_band=prediction.risk_band,
            predicted_class=prediction.predicted_class,
            threshold=settings.model.decision_threshold,
            model_name=loaded.name,
            model_version=loaded.version,
            model_stage=loaded.stage,
            imputed_fields=prediction.imputed_fields,
            contributions=prediction.contributions,  # type: ignore[arg-type]
            disclaimer=settings.api.clinical_disclaimer,
            correlation_id=correlation_id,
        )

    # Guard applied to the endpoints that consume the model; /health and /metrics stay
    # open so probes and Prometheus keep working (src/api/auth.py explains the split).
    warn_if_open(settings)

    async def guard(authorization: str | None = Header(default=None)) -> None:
        """Reject a request that lacks the configured bearer token."""
        if not is_authorized(settings, authorization):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=UNAUTHORIZED_DETAIL,
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses=_ERROR_RESPONSES,
        dependencies=[Depends(guard)],
    )
    async def predict(
        payload: PredictionRequest,
        request: Request,
        explain: bool = Query(default=False, description="incluir contribuicoes por variavel"),
        service: PredictionService = Depends(get_service),  # noqa: B008
    ) -> PredictionResponse:
        """Score one observation."""
        record = payload.model_dump()
        prediction = service.predict_records([record], explain=explain)[0]
        metrics.observe_prediction(prediction.risk_band)
        monitor.observe([record])
        return to_response(prediction, service.model, correlation_id_of(request))

    @app.post(
        "/predict/batch",
        response_model=BatchPredictionResponse,
        responses=_ERROR_RESPONSES,
        dependencies=[Depends(guard)],
    )
    async def predict_batch(
        payload: BatchPredictionRequest,
        request: Request,
        service: PredictionService = Depends(get_service),  # noqa: B008
    ) -> BatchPredictionResponse:
        """Score several observations in the order received."""
        records = [r.model_dump() for r in payload.records]
        predictions = service.predict_records(records)
        monitor.observe(records)
        correlation_id = correlation_id_of(request)
        for prediction in predictions:
            metrics.observe_prediction(prediction.risk_band)
        return BatchPredictionResponse(
            predictions=[to_response(p, service.model, correlation_id) for p in predictions],
            n=len(predictions),
            n_imputed_rows=sum(1 for p in predictions if p.imputed_fields),
            correlation_id=correlation_id,
        )

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Liveness (process up) and readiness (model loaded)."""
        loaded = model_cache.peek()
        return HealthResponse(
            status="ok" if loaded is not None else "degraded",
            model_loaded=loaded is not None,
            model_name=loaded.name if loaded else None,
            model_version=loaded.version if loaded else None,
            model_stage=loaded.stage if loaded else None,
            model_source=settings.api.model_source,
            package_version=__version__,
        )

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus text exposition."""
        payload, content_type = metrics.exposition()
        return Response(content=payload, media_type=content_type)

    app.include_router(build_model_router(settings, model_cache), dependencies=[Depends(guard)])

    app.state.metrics = metrics
    app.state.settings = settings
    app.state.drift_monitor = monitor
    return app


app = create_app()
