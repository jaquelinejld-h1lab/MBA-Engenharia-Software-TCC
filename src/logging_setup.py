"""Structured JSON logging.

Call ``configure_logging`` once at process start (CLI entry point, API
startup, Streamlit startup). Modules obtain loggers with ``get_logger`` and
attach context with the ``extra`` argument, which lands as top-level JSON
keys (for example a correlation id).
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from src.config import LoggingConfig

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialize ``record`` to a single JSON line."""
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Any key passed via ``extra=`` becomes a top-level field.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(config: LoggingConfig) -> None:
    """Configure the root logger according to the logging section of the config.

    Parameters
    ----------
    config : LoggingConfig
        Level and output format. Idempotent: existing handlers are replaced.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    # Windows redirects stdout with the locale code page (cp1252) and would raise
    # UnicodeEncodeError on the first non-ASCII log field; force UTF-8 when possible.
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")
    handler = logging.StreamHandler(sys.stdout)
    if config.json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(config.level)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (thin wrapper kept for a single import point)."""
    return logging.getLogger(name)
