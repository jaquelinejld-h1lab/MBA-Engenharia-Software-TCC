"""Optional bearer token: header parsing, comparison and the open default."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from src.api.auth import configured_token, extract_bearer, is_authorized
from src.config import Settings


def _with_token(cfg: Settings, value: str | None) -> Settings:
    """Copy of the settings carrying (or not) a token, without touching the environment."""
    api = cfg.api.model_copy(update={"auth_token": SecretStr(value) if value else None})
    return cfg.model_copy(update={"api": api})


def test_no_token_configured_means_the_api_is_open(cfg: Settings) -> None:
    assert configured_token(_with_token(cfg, None)) is None


def test_blank_token_counts_as_no_token(cfg: Settings) -> None:
    """An empty variable is a deployment mistake, not a credential of length zero."""
    assert configured_token(_with_token(cfg, "   ")) is None


def test_configured_token_is_read_from_the_secret(cfg: Settings) -> None:
    assert configured_token(_with_token(cfg, "s3gr3d0")) == "s3gr3d0"


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("BEARER   abc  ", "abc"),
        (None, None),
        ("", None),
        ("abc", None),
        ("Basic abc", None),
        ("Bearer", None),
        ("Bearer   ", None),
    ],
)
def test_header_parsing(header: str | None, expected: str | None) -> None:
    assert extract_bearer(header) == expected


def test_token_with_spaces_survives_parsing() -> None:
    """Only the scheme is split off: the credential itself is taken verbatim."""
    assert extract_bearer("Bearer a b c") == "a b c"


def test_the_token_never_appears_in_a_repr(cfg: Settings) -> None:
    settings = _with_token(cfg, "s3gr3d0")
    assert "s3gr3d0" not in repr(settings.api)
    assert "s3gr3d0" not in str(settings.api.auth_token)


def test_config_yaml_does_not_carry_a_token(cfg: Settings) -> None:
    """The credential is environment-only; a token committed in the YAML is a leak."""
    assert cfg.api.auth_token is None


def test_authorization_is_granted_only_by_the_exact_token(cfg: Settings) -> None:
    settings = _with_token(cfg, "s3gr3d0")
    assert is_authorized(settings, "Bearer s3gr3d0") is True
    assert is_authorized(settings, "Bearer s3gr3d") is False
    assert is_authorized(settings, "Bearer s3gr3d0x") is False
    assert is_authorized(settings, None) is False


def test_open_api_authorizes_every_request(cfg: Settings) -> None:
    settings = _with_token(cfg, None)
    assert is_authorized(settings, None) is True
    assert is_authorized(settings, "Bearer qualquer") is True
