"""Correlation id, structured request log, metrics and exception envelopes.

Only Starlette symbols are used here (FastAPI re-exports them), so this
module and its tests do not need FastAPI installed.

The correlation id comes from the ``X-Correlation-ID`` request header when
present (so the Streamlit client and the API share one id) and is generated
otherwise. It is stored in ``request.state.correlation_id``, echoed in the
response header and included in every log line and error envelope.

Exception handlers translate domain errors into JSON envelopes with an HTTP
status and never expose a stack trace; unexpected errors are logged with the
traceback server-side and answered with a generic 500.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Protocol

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from src.api.headers import CORRELATION_HEADER
from src.exceptions import (
    DataContractError,
    HypertensionPipelineError,
    ModelNotFoundError,
    PredictionError,
    TransformError,
)
from src.logging_setup import get_logger

logger = get_logger("src.api.request")


class MetricsSink(Protocol):
    """What the middleware needs from the metrics object (``ApiMetrics`` satisfies it)."""

    def observe_request(self, endpoint: str, method: str, status: int, seconds: float) -> None:
        """Record one finished request."""

    def observe_error(self, error: str) -> None:
        """Count one failure."""


_STATUS_BY_ERROR: dict[type[Exception], int] = {
    DataContractError: 422,
    TransformError: 422,
    PredictionError: 422,
    ModelNotFoundError: 503,
}


def correlation_id_of(request: Request) -> str:
    """Correlation id stored by the middleware (or a fresh one outside it)."""
    value = getattr(request.state, "correlation_id", None)
    return str(value) if value else str(uuid.uuid4())


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign the correlation id, time the request, log it and record metrics."""

    def __init__(self, app: ASGIApp, metrics: MetricsSink) -> None:
        super().__init__(app)
        self.metrics = metrics

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Wrap one request."""
        correlation_id = request.headers.get(CORRELATION_HEADER) or str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - started
            endpoint = request.url.path
            self.metrics.observe_request(endpoint, request.method, status, elapsed)
            logger.info(
                "request",
                extra={
                    "correlation_id": correlation_id,
                    "method": request.method,
                    "path": endpoint,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 3),
                },
            )
        response.headers[CORRELATION_HEADER] = correlation_id
        return response


def _envelope(request: Request, status: int, error: str, detail: str) -> JSONResponse:
    correlation_id = correlation_id_of(request)
    return JSONResponse(
        status_code=status,
        content={"error": error, "detail": detail, "correlation_id": correlation_id},
        headers={CORRELATION_HEADER: correlation_id},
    )


def register_exception_handlers(app: Starlette, metrics: MetricsSink) -> None:
    """Install the domain and catch-all handlers on ``app`` (FastAPI or Starlette)."""

    async def _domain_error(request: Request, exc: Exception) -> JSONResponse:
        status = next(
            (code for kind, code in _STATUS_BY_ERROR.items() if isinstance(exc, kind)), 500
        )
        metrics.observe_error(type(exc).__name__)
        logger.warning(
            "domain error",
            extra={
                "correlation_id": correlation_id_of(request),
                "error": type(exc).__name__,
                "detail": str(exc),
            },
        )
        return _envelope(request, status, type(exc).__name__, str(exc))

    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        metrics.observe_error("InternalError")
        logger.error(
            "unhandled error",
            extra={"correlation_id": correlation_id_of(request), "error": type(exc).__name__},
            exc_info=exc,
        )
        return _envelope(request, 500, "InternalError", "internal error; see server logs")

    app.add_exception_handler(HypertensionPipelineError, _domain_error)
    app.add_exception_handler(Exception, _unexpected_error)
