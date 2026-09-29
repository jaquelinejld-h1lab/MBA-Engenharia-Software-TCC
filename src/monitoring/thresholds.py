"""PSI interpretation levels (no scientific dependencies; shared with the Streamlit client)."""

from __future__ import annotations

from typing import Literal

from src.config import DriftConfig

DriftLevel = Literal["ok", "warning", "alert"]


def psi_level(value: float, drift: DriftConfig) -> DriftLevel:
    """Map a PSI value to ``ok``, ``warning`` or ``alert`` using config thresholds."""
    if value >= drift.psi_alert:
        return "alert"
    if value >= drift.psi_warning:
        return "warning"
    return "ok"
