"""In-memory sliding window of scored observations and the drift monitor on top of it.

The API appends every successfully scored record to :class:`ObservationWindow`
(bounded, thread-safe). :class:`DriftMonitor` compares the window with the
persisted training reference on demand (each ``/metrics`` scrape) and turns
the result into Prometheus gauges through :meth:`DriftMonitor.snapshot`.

The window lives in process memory only: restarting the container resets
it, which is acceptable for the local scope (an external store such as a
database would be the cloud-ready alternative, recorded as an ADR).
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.config import DriftConfig, MonitoringConfig
from src.monitoring.drift import DriftReport, ReferenceWindow, compute_drift


class ObservationWindow:
    """Bounded FIFO of raw records (English variable names)."""

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError("window maxlen must be positive")
        self._rows: deque[dict[str, Any]] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._total = 0

    def extend(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append records, evicting the oldest beyond ``maxlen``."""
        with self._lock:
            for record in records:
                self._rows.append(dict(record))
                self._total += 1

    def to_frame(self) -> pd.DataFrame:
        """Snapshot of the current window as a frame (empty frame when no rows)."""
        with self._lock:
            rows = list(self._rows)
        return pd.DataFrame.from_records(rows) if rows else pd.DataFrame()

    def clear(self) -> None:
        """Drop every row (tests and scenario resets)."""
        with self._lock:
            self._rows.clear()

    def __len__(self) -> int:
        """Rows currently held."""
        with self._lock:
            return len(self._rows)

    @property
    def total_received(self) -> int:
        """Records appended since process start, including evicted ones."""
        with self._lock:
            return self._total


@dataclass(frozen=True)
class DriftSnapshot:
    """What the metrics exporter publishes."""

    reference_loaded: bool
    window_rows: int
    enough_rows: bool
    report: DriftReport | None


class DriftMonitor:
    """Evaluate drift of the observation window against the training reference."""

    def __init__(
        self,
        monitoring: MonitoringConfig,
        reference: ReferenceWindow | None,
        window: ObservationWindow | None = None,
    ) -> None:
        self.monitoring = monitoring
        self.reference = reference
        self.window = window or ObservationWindow(monitoring.window_size)

    @property
    def drift(self) -> DriftConfig:
        """Threshold block of the configuration."""
        return self.monitoring.drift

    def observe(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Add scored records to the window."""
        self.window.extend(records)

    def snapshot(self) -> DriftSnapshot:
        """Compute the current report (``None`` while the window is too small)."""
        rows = len(self.window)
        enough = rows >= self.monitoring.min_rows_for_drift
        if self.reference is None or not enough:
            return DriftSnapshot(self.reference is not None, rows, enough, None)
        report = compute_drift(self.reference, self.window.to_frame(), self.drift)
        return DriftSnapshot(True, rows, True, report)
