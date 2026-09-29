"""Correlation id, request log, metrics and error envelopes on a bare Starlette app."""

from __future__ import annotations

import json
import logging

import pytest

from src.exceptions import ModelNotFoundError, TransformError

pytest.importorskip("starlette")
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from src.api.headers import CORRELATION_HEADER
from src.api.middleware import RequestContextMiddleware, register_exception_handlers


class _Recorder:
    """Minimal MetricsSink: records calls instead of exporting Prometheus series."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str, int]] = []
        self.errors: list[str] = []

    def observe_request(self, endpoint: str, method: str, status: int, seconds: float) -> None:
        assert seconds >= 0
        self.requests.append((endpoint, method, status))

    def observe_error(self, error: str) -> None:
        self.errors.append(error)


def _app() -> tuple[Starlette, _Recorder]:
    async def ok(request):  # type: ignore[no-untyped-def]
        return JSONResponse({"cid": request.state.correlation_id})

    async def transform_error(request):  # type: ignore[no-untyped-def]
        raise TransformError("bad category")

    async def registry_error(request):  # type: ignore[no-untyped-def]
        raise ModelNotFoundError("no model")

    async def crash(request):  # type: ignore[no-untyped-def]
        raise RuntimeError("secret detail")

    metrics = _Recorder()
    app = Starlette(
        routes=[
            Route("/ok", ok),
            Route("/transform", transform_error),
            Route("/registry", registry_error),
            Route("/crash", crash),
        ]
    )
    app.add_middleware(RequestContextMiddleware, metrics=metrics)
    register_exception_handlers(app, metrics)
    return app, metrics


def test_correlation_id_propagates_and_is_generated() -> None:
    app, _ = _app()
    client = TestClient(app)
    given = client.get("/ok", headers={CORRELATION_HEADER: "abc"})
    assert given.json()["cid"] == "abc" and given.headers[CORRELATION_HEADER] == "abc"
    fresh = client.get("/ok")
    assert (
        fresh.json()["cid"] == fresh.headers[CORRELATION_HEADER] and len(fresh.json()["cid"]) > 10
    )


def test_domain_errors_become_json_envelopes() -> None:
    app, metrics = _app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.get("/transform")
    assert r.status_code == 422 and r.json()["error"] == "TransformError"
    r = client.get("/registry")
    assert r.status_code == 503 and r.json()["detail"] == "no model"
    r = client.get("/crash")
    assert r.status_code == 500 and r.json()["error"] == "InternalError"
    assert "secret detail" not in r.text and "Traceback" not in r.text
    assert all("correlation_id" in client.get(p).json() for p in ("/transform", "/registry"))
    assert {"TransformError", "ModelNotFoundError", "InternalError"} <= set(metrics.errors)


def test_request_log_and_latency_metrics(capsys: pytest.CaptureFixture[str]) -> None:
    from src.config import LoggingConfig  # noqa: PLC0415
    from src.logging_setup import configure_logging  # noqa: PLC0415

    configure_logging(LoggingConfig(level="INFO", json_format=True))
    app, metrics = _app()
    TestClient(app).get("/ok", headers={CORRELATION_HEADER: "log-1"})
    lines = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    request_line = next(line for line in lines if line["message"] == "request")
    assert request_line["correlation_id"] == "log-1" and request_line["status"] == 200
    assert request_line["path"] == "/ok" and request_line["duration_ms"] >= 0
    assert metrics.requests == [("/ok", "GET", 200)]
    logging.getLogger().handlers.clear()
