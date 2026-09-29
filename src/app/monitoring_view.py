"""Data of the monitoring panel, computed from the ``/metrics`` exposition of the API.

Everything here is framework-free (tested without Streamlit). The panel
shows lifetime figures of the API process taken from its own metrics;
when a Prometheus server is reachable the p95 over a moving window is
also queried, which is the figure Grafana shows.

Quality gate status: the API rejects records outside the data contract
with 422 (``DataContractError`` / validation errors). The share of such
rejections over the received prediction requests is the online quality
gate; the offline gate (``python -m src.data.quality``) runs in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

from src.config import DriftConfig
from src.monitoring.prometheus_text import (
    Sample,
    histogram_quantile,
    parse_exposition,
    select,
    total,
)
from src.monitoring.thresholds import psi_level

PREDICTION_ENDPOINTS = ("/predict", "/predict/batch")
_CLIENT_ERROR_MIN = 400
_SERVER_ERROR_MIN = 500
_MS_PER_SECOND = 1000.0


@dataclass(frozen=True)
class PanelSummary:
    """Numbers shown by the monitoring page."""

    total_requests: float
    prediction_requests: float
    error_rate: float  # share of prediction requests answered with 5xx
    rejection_rate: float  # share of prediction requests answered with 4xx (contract)
    p95_ms: float | None
    p50_ms: float | None
    predictions_by_band: dict[str, float]
    drift_reference_loaded: bool
    drift_evaluated: bool
    drift_window_rows: int
    drift_alerts: int
    drift: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def quality_gate(self) -> str:
        """``ok`` (no rejection), ``atencao`` (some) or ``sem dados``."""
        if self.prediction_requests == 0:
            return "sem dados"
        return "ok" if self.rejection_rate == 0 else "atenção"


def _status_share(samples: list[Sample], minimum: int, maximum: int) -> float:
    prediction = [
        s
        for s in select(samples, "api_requests_total")
        if s.labels.get("endpoint") in PREDICTION_ENDPOINTS
    ]
    received = sum(s.value for s in prediction)
    if received == 0:
        return 0.0
    matched = sum(
        s.value for s in prediction if minimum <= int(s.labels.get("status", "0")) < maximum
    )
    return matched / received


def drift_table(samples: list[Sample], drift: DriftConfig) -> pd.DataFrame:
    """PSI, KS and level per variable from the gauges (empty when not evaluated)."""
    psi = {s.labels["variable"]: s.value for s in select(samples, "data_drift_psi")}
    if not psi:
        return pd.DataFrame(columns=["variavel", "psi", "nivel_psi", "ks", "alerta"])
    ks = {s.labels["variable"]: s.value for s in select(samples, "data_drift_ks_statistic")}
    alert = {s.labels["variable"]: s.value >= 1 for s in select(samples, "data_drift_alert")}
    frame = pd.DataFrame(
        {
            "variavel": list(psi),
            "psi": list(psi.values()),
            "nivel_psi": [psi_level(v, drift) for v in psi.values()],
            "ks": [ks.get(name) for name in psi],
            "alerta": [alert.get(name, False) for name in psi],
        }
    )
    return frame.sort_values("psi", ascending=False).reset_index(drop=True)


def summarize(metrics_text: str, drift: DriftConfig) -> PanelSummary:
    """Parse the exposition and compute the panel figures."""
    samples = parse_exposition(metrics_text)
    prediction_requests = sum(
        s.value
        for s in select(samples, "api_requests_total")
        if s.labels.get("endpoint") in PREDICTION_ENDPOINTS
    )
    p95 = histogram_quantile(samples, "api_request_latency_seconds", 0.95, endpoint="/predict")
    p50 = histogram_quantile(samples, "api_request_latency_seconds", 0.50, endpoint="/predict")
    bands = {s.labels["risk_band"]: s.value for s in select(samples, "api_predictions_total")}
    return PanelSummary(
        total_requests=total(samples, "api_requests_total"),
        prediction_requests=prediction_requests,
        error_rate=_status_share(samples, _SERVER_ERROR_MIN, _SERVER_ERROR_MIN + 100),
        rejection_rate=_status_share(samples, _CLIENT_ERROR_MIN, _SERVER_ERROR_MIN),
        p95_ms=None if p95 is None else p95 * _MS_PER_SECOND,
        p50_ms=None if p50 is None else p50 * _MS_PER_SECOND,
        predictions_by_band=bands,
        drift_reference_loaded=total(samples, "data_drift_reference_loaded") >= 1,
        drift_evaluated=total(samples, "data_drift_evaluated") >= 1,
        drift_window_rows=int(total(samples, "data_drift_window_rows")),
        drift_alerts=int(total(samples, "data_drift_alerts")),
        drift=drift_table(samples, drift),
    )


def prometheus_p95_series(
    prometheus_url: str, *, window: str, span_minutes: int, step_seconds: int, timeout: float
) -> pd.DataFrame | None:
    """p95 latency of ``/predict`` over time from Prometheus (``None`` when unreachable).

    Runs ``histogram_quantile(0.95, sum(rate(bucket[window])) by (le))`` on
    the range API; the result has columns ``timestamp`` and ``p95_ms``.
    """
    query = (
        "histogram_quantile(0.95, sum(rate(api_request_latency_seconds_bucket"
        f'{{endpoint="/predict"}}[{window}])) by (le))'
    )
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(minutes=span_minutes)
    try:
        response = httpx.get(
            f"{prometheus_url.rstrip('/')}/api/v1/query_range",
            params={
                "query": query,
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": step_seconds,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    results = payload.get("data", {}).get("result", [])
    if not results:
        return pd.DataFrame(columns=["timestamp", "p95_ms"])
    points = [
        (pd.Timestamp(float(ts), unit="s", tz="UTC"), float(value) * _MS_PER_SECOND)
        for ts, value in results[0].get("values", [])
        if value not in {"NaN", "+Inf", "-Inf"}
    ]
    return pd.DataFrame(points, columns=["timestamp", "p95_ms"])
