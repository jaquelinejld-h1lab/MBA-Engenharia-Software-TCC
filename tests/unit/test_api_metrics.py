"""Prometheus metric families (requires prometheus-client)."""

from __future__ import annotations

import pytest

pytest.importorskip("prometheus_client")
from src.api.metrics import ApiMetrics


def test_metrics_exposition_contains_all_families() -> None:
    metrics = ApiMetrics()
    metrics.observe_request("/predict", "POST", 200, 0.012)
    metrics.observe_prediction("alto")
    metrics.observe_error("TransformError")
    text, content_type = metrics.exposition()
    body = text.decode()
    assert content_type.startswith("text/plain")
    assert 'api_requests_total{endpoint="/predict",method="POST",status="200"} 1.0' in body
    assert 'api_predictions_total{risk_band="alto"} 1.0' in body
    assert 'api_prediction_errors_total{error="TransformError"} 1.0' in body
    assert (
        'api_request_latency_seconds_bucket{endpoint="/predict",le="0.025",method="POST"} 1.0'
        in body
    )


def test_two_instances_do_not_collide() -> None:
    ApiMetrics()
    ApiMetrics()
