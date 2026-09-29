"""Optional bearer token for the prediction endpoints.

The token never lives in the repository: it comes from ``HTN_API__AUTH_TOKEN`` alone,
which is an environment variable locally and a GitHub Secret in the pipeline. Leaving it
unset keeps the API open, which is the local default of this study, and the API logs a
warning at startup so an open deployment is never silent.

What the token protects, when set:

* ``/predict`` and ``/predict/batch``: they consume the model and carry health data;
* ``/model/*``: they describe the model in service.

What stays open either way: ``/health`` and ``/metrics``, because Prometheus scrapes them
and a liveness probe that needs a credential is a liveness probe that fails for the wrong
reason. Neither endpoint exposes a prediction or a personal record.

Comparison uses :func:`secrets.compare_digest`, so a wrong token takes the same time to
reject whatever its prefix, and the token is never written to a log or an error message.
The module holds the policy only; the HTTP response that carries a rejection is built in
``src/api/main.py``, which keeps this file importable without FastAPI and therefore unit
testable on its own.
"""

from __future__ import annotations

import secrets

from src.config import Settings
from src.logging_setup import get_logger

logger = get_logger(__name__)

_SCHEME = "bearer"
UNAUTHORIZED_DETAIL = "credencial ausente ou invalida"


def configured_token(settings: Settings) -> str | None:
    """Return the configured token, or ``None`` when the API is open."""
    secret = settings.api.auth_token
    if secret is None:
        return None
    value = secret.get_secret_value().strip()
    return value or None


def warn_if_open(settings: Settings) -> None:
    """Log once, at startup, whether the prediction endpoints require a credential."""
    if configured_token(settings) is None:
        logger.warning(
            "api without authentication",
            extra={"detail": "defina HTN_API__AUTH_TOKEN para exigir Bearer em /predict e /model"},
        )
    else:
        logger.info("api authentication enabled", extra={"scheme": "bearer"})


def extract_bearer(authorization: str | None) -> str | None:
    """Token of an ``Authorization: Bearer <token>`` header, or ``None``."""
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    expected_parts = 2
    if len(parts) != expected_parts or parts[0].lower() != _SCHEME:
        return None
    return parts[1].strip() or None


def is_authorized(settings: Settings, authorization: str | None) -> bool:
    """Whether a request carrying this ``Authorization`` header may proceed.

    Parameters
    ----------
    settings : Settings
        Validated configuration of the running app.
    authorization : str or None
        Raw header value, exactly as received.

    Returns
    -------
    bool
        ``True`` when no token is configured (the open local default) or when the header
        carries exactly the configured one. The comparison is constant time, so a wrong
        credential takes the same time to reject whatever its prefix, and neither the
        expected nor the received value is ever logged.

    Notes
    -----
    This module deliberately does not import FastAPI: the policy is plain Python and is
    unit tested without a web framework. ``src/api/main.py`` turns a ``False`` here into
    the 401 response with the ``WWW-Authenticate`` header.
    """
    expected = configured_token(settings)
    if expected is None:
        return True
    received = extract_bearer(authorization)
    if received is None:
        return False
    return secrets.compare_digest(received, expected)
