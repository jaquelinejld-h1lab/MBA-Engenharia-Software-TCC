"""Headless smoke test of the Streamlit client with a mocked API (needs streamlit)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.app import client as client_module
from src.paths import PROJECT_ROOT

APP_FILE = PROJECT_ROOT / "src" / "app" / "main.py"
_TIMEOUT = 30


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/health":
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "model_loaded": True,
                "model_name": "hypertension-lr",
                "model_version": "1",
                "model_stage": "Production",
                "model_source": "local",
                "package_version": "test",
            },
        )
    if path == "/metrics":
        return httpx.Response(
            200,
            text=(
                'api_requests_total{endpoint="/predict",method="POST",status="200"} 10.0\n'
                'api_request_latency_seconds_bucket{endpoint="/predict",le="0.05",method="POST"} '
                "10.0\n"
                'api_request_latency_seconds_bucket{endpoint="/predict",le="+Inf",method="POST"} '
                "10.0\n"
                'api_predictions_total{risk_band="baixo"} 10.0\n'
                "data_drift_reference_loaded 1.0\n"
                "data_drift_evaluated 0.0\n"
                "data_drift_window_rows 10.0\n"
            ),
        )
    if path in {"/predict", "/predict/batch"}:
        body = json.loads(request.content)
        answer = {
            "probability": 0.2,
            "risk_band": "baixo",
            "predicted_class": 0,
            "threshold": 0.5,
            "model_name": "hypertension-lr",
            "model_version": "1",
            "model_stage": "Production",
            "imputed_fields": ["cholesterol"],
            "contributions": [{"feature": "sex_2", "value": 1, "contribution": -0.4}],
            "disclaimer": "apoio",
            "correlation_id": "cid",
        }
        if path == "/predict":
            return httpx.Response(200, json=answer)
        return httpx.Response(
            200,
            json={
                "predictions": [answer] * len(body["records"]),
                "n": len(body["records"]),
                "n_imputed_rows": 0,
                "correlation_id": "cid",
            },
        )
    return httpx.Response(404, json={"detail": "not found"})


class _HttpxProxy:
    """``httpx`` namespace whose ``Client`` injects a mock transport."""

    def __init__(self, factory: Callable[..., httpx.Client]) -> None:
        self.Client = factory

    def __getattr__(self, name: str) -> object:
        return getattr(httpx, name)


@pytest.fixture
def mocked_api(monkeypatch: pytest.MonkeyPatch) -> None:
    real_client = httpx.Client

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_handler)
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client_module, "httpx", _HttpxProxy(factory))


def _run(page: str | None = None) -> AppTest:
    # cache_resource holds the ApiClient across runs in one process: reset per test
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(Path(APP_FILE)), default_timeout=_TIMEOUT)
    app.run()
    if page is not None:
        app.radio(key="page").set_value(page).run()
    return app


@pytest.mark.usefixtures("mocked_api")
def test_individual_page_renders_first_step_with_defaults() -> None:
    app = _run()
    assert not app.exception
    assert app.session_state["step"] == 0
    assert any("Sociodemogr" in h.value for h in app.subheader)
    assert app.number_input(key="f_age").value == 42
    assert any("hypertension-lr" in s.value for s in app.sidebar.success)


@pytest.mark.usefixtures("mocked_api")
def test_batch_and_monitoring_pages_render() -> None:
    batch = _run("Predição em lote")
    assert not batch.exception
    assert any("lote" in h.value for h in batch.subheader)
    monitoring = _run("Monitoramento")
    assert not monitoring.exception
    assert any("monitoramento" in h.value.lower() for h in monitoring.subheader)
    assert any("PSI e KS exigem" in i.value for i in monitoring.info)
    assert any("Modelo ativo" in i.value for i in monitoring.info)


def test_api_down_shows_endpoint_not_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    real_client = httpx.Client

    def factory(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(refuse)
        return real_client(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(client_module, "httpx", _HttpxProxy(factory))
    app = _run()
    assert not app.exception
    messages = [e.value for e in app.error]
    assert messages and "/health" in messages[0] and "Traceback" not in messages[0]
