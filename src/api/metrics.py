"""Prometheus metrics of the API (RED: rate, errors, duration), predictions by band and drift.

A dedicated ``CollectorRegistry`` is used instead of the global one so that
tests can build several applications in one process without duplicate
registration errors.

Drift gauges are refreshed lazily at every scrape from a
:class:`~src.monitoring.window.DriftSnapshot` provider, so the cost of PSI
and KS is paid once per scrape interval instead of once per request.
"""

from __future__ import annotations

from collections.abc import Callable

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import CONTENT_TYPE_LATEST

from src.config import RetrainingConfig
from src.monitoring.retraining import decide
from src.monitoring.window import DriftSnapshot

# Buckets chosen around the SLO (p95 <= 200 ms local, 300 ms declared)
_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.5, 5.0)


class ApiMetrics:
    """Metric family holder bound to one registry."""

    def __init__(
        self,
        drift_provider: Callable[[], DriftSnapshot] | None = None,
        retraining: RetrainingConfig | None = None,
        min_rows_for_drift: int = 0,
    ) -> None:
        # The retraining decision is derived from the same snapshot that feeds the drift
        # gauges, so the gauge and evidencias/retraining_flag.json cannot disagree on
        # what counts as "enough variables in alert" (src/monitoring/retraining.py).
        self.retraining = retraining
        self.min_rows_for_drift = min_rows_for_drift
        self.registry = CollectorRegistry()
        self.drift_provider = drift_provider
        self.request_latency = Histogram(
            "api_request_latency_seconds",
            "Request latency in seconds",
            labelnames=("endpoint", "method"),
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.requests_total = Counter(
            "api_requests_total",
            "Requests by endpoint, method and HTTP status",
            labelnames=("endpoint", "method", "status"),
            registry=self.registry,
        )
        self.predictions_total = Counter(
            "api_predictions_total",
            "Predictions by risk band",
            labelnames=("risk_band",),
            registry=self.registry,
        )
        self.prediction_errors_total = Counter(
            "api_prediction_errors_total",
            "Prediction failures by error type",
            labelnames=("error",),
            registry=self.registry,
        )
        self.drift_psi = Gauge(
            "data_drift_psi",
            "PSI of each raw variable, current window vs training reference",
            labelnames=("variable",),
            registry=self.registry,
        )
        self.drift_ks = Gauge(
            "data_drift_ks_statistic",
            "Two-sample KS statistic of each numeric raw variable",
            labelnames=("variable",),
            registry=self.registry,
        )
        self.drift_alert = Gauge(
            "data_drift_alert",
            "1 when the variable is in PSI or KS alert",
            labelnames=("variable",),
            registry=self.registry,
        )
        self.drift_alerts_count = Gauge(
            "data_drift_alerts", "Number of variables in alert", registry=self.registry
        )
        self.drift_window_rows = Gauge(
            "data_drift_window_rows", "Rows in the observation window", registry=self.registry
        )
        self.drift_reference_loaded = Gauge(
            "data_drift_reference_loaded",
            "1 when the training reference window is loaded",
            registry=self.registry,
        )
        self.drift_evaluated = Gauge(
            "data_drift_evaluated",
            "1 when the window holds enough rows for PSI and KS",
            registry=self.registry,
        )

        self.retraining_recommended = Gauge(
            "retraining_recommended",
            "1 when the drift in the observation window recommends retraining the model",
            registry=self.registry,
        )

    def observe_request(self, endpoint: str, method: str, status: int, seconds: float) -> None:
        """Record one finished request."""
        self.request_latency.labels(endpoint=endpoint, method=method).observe(seconds)
        self.requests_total.labels(endpoint=endpoint, method=method, status=str(status)).inc()

    def observe_prediction(self, risk_band: str) -> None:
        """Count one prediction in its risk band."""
        self.predictions_total.labels(risk_band=risk_band).inc()

    def observe_error(self, error: str) -> None:
        """Count one prediction failure."""
        self.prediction_errors_total.labels(error=error).inc()

    def apply_drift(self, snapshot: DriftSnapshot) -> None:
        """Publish one drift snapshot on the gauges."""
        self.drift_reference_loaded.set(int(snapshot.reference_loaded))
        self.drift_window_rows.set(snapshot.window_rows)
        self.drift_evaluated.set(int(snapshot.report is not None))
        if self.retraining is not None:
            decision = decide(snapshot.report, self.retraining, min_rows=self.min_rows_for_drift)
            self.retraining_recommended.set(int(decision.recommended))
        if snapshot.report is None:
            self.drift_alerts_count.set(0)
            return
        for variable in snapshot.report.variables:
            self.drift_psi.labels(variable=variable.name).set(variable.psi)
            self.drift_alert.labels(variable=variable.name).set(int(variable.alert))
            if variable.ks_statistic is not None:
                self.drift_ks.labels(variable=variable.name).set(variable.ks_statistic)
        self.drift_alerts_count.set(len(snapshot.report.alerts))

    def exposition(self) -> tuple[bytes, str]:
        """Text exposition payload and its content type (drift gauges refreshed first)."""
        if self.drift_provider is not None:
            self.apply_drift(self.drift_provider())
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
