"""Structured logging output."""

from __future__ import annotations

import json
import logging

import pytest

from src.config import LoggingConfig
from src.logging_setup import configure_logging, get_logger


def test_json_lines_with_extra_fields(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(LoggingConfig(level="INFO", json_format=True))
    get_logger("test").info("hello", extra={"correlation_id": "abc-123"})
    line = capsys.readouterr().out.strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["message"] == "hello"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["correlation_id"] == "abc-123"
    assert "timestamp" in payload


def test_level_is_applied(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging(LoggingConfig(level="WARNING", json_format=True))
    get_logger("test").info("suppressed")
    assert capsys.readouterr().out.strip() == ""


def test_reconfigure_is_idempotent() -> None:
    configure_logging(LoggingConfig(level="INFO", json_format=True))
    configure_logging(LoggingConfig(level="INFO", json_format=False))
    assert len(logging.getLogger().handlers) == 1
