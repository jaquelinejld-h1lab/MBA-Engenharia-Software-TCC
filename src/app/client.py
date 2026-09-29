"""HTTP client of the prediction API used by the Streamlit pages.

The interface never imports the model, the transformer or MLflow: every
number it shows comes from this client. Each call carries an
``X-Correlation-ID`` header so the API log and the interface share one
id, and failures are turned into :class:`ApiUnavailableError` with the
endpoint that failed and a message safe to show (no stack trace).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from typing import Any

import httpx

from src.api.headers import CORRELATION_HEADER
from src.exceptions import HypertensionPipelineError


class ApiUnavailableError(HypertensionPipelineError):
    """The API could not be reached or answered with an error."""

    def __init__(self, endpoint: str, reason: str, status: int | None = None) -> None:
        self.endpoint = endpoint
        self.reason = reason
        self.status = status
        super().__init__(f"{endpoint}: {reason}")


class ApiClient:
    """Thin wrapper over ``httpx.Client`` with the project's error handling."""

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float,
        batch_chunk_size: int,
        transport: httpx.BaseTransport | None = None,
        token: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.batch_chunk_size = batch_chunk_size
        # Sent on every request when the API requires a credential; kept out of the
        # instance attributes so it does not show up in a repr or a Streamlit trace.
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_seconds,
            transport=transport,
            headers=headers,
        )

    def close(self) -> None:
        """Release the connection pool."""
        self._client.close()

    # ---------------------------------------------------------------- core
    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        json: dict[str, object] | None = None,
        params: dict[str, str] | None = None,
    ) -> httpx.Response:
        correlation_id = str(uuid.uuid4())
        url = f"{self.base_url}{endpoint}"
        try:
            response = self._client.request(
                method,
                endpoint,
                json=json,
                params=params,
                headers={CORRELATION_HEADER: correlation_id},
            )
        except httpx.TimeoutException as error:
            raise ApiUnavailableError(url, "tempo limite excedido ao aguardar a API") from error
        except httpx.HTTPError as error:
            raise ApiUnavailableError(
                url, f"não foi possível conectar ({type(error).__name__})"
            ) from error
        if response.status_code >= 400:  # noqa: PLR2004
            raise ApiUnavailableError(url, _error_detail(response), response.status_code)
        return response

    # ------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        """``GET /health``."""
        return dict(self._request("GET", "/health").json())

    def predict(self, record: dict[str, Any], *, explain: bool = True) -> dict[str, Any]:
        """``POST /predict`` for one record."""
        params = {"explain": "true"} if explain else None
        return dict(self._request("POST", "/predict", json=record, params=params).json())

    def predict_batch(
        self,
        records: Sequence[dict[str, Any]],
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """``POST /predict/batch`` in chunks of ``batch_chunk_size``; keeps the order."""
        predictions: list[dict[str, Any]] = []
        total = len(records)
        for start in range(0, total, self.batch_chunk_size):
            chunk = list(records[start : start + self.batch_chunk_size])
            payload = self._request("POST", "/predict/batch", json={"records": chunk}).json()
            predictions.extend(payload["predictions"])
            if progress is not None:
                progress(min(start + len(chunk), total), total)
        return predictions

    def model_report(self, section: str) -> dict[str, Any]:
        """``GET /model/<section>``: the global report of the model in service.

        Sections are ``summary``, ``roc``, ``importance``, ``contributions`` and
        ``subgroups``. Each answer carries ``available`` and, when false, ``reason``:
        an artifact missing from an older training is a message on the page, not an
        error, so the caller checks the flag instead of catching an exception.
        """
        return dict(self._request("GET", f"/model/{section}").json())

    def metrics_text(self) -> str:
        """``GET /metrics`` raw Prometheus exposition."""
        return self._request("GET", "/metrics").text


def _error_detail(response: httpx.Response) -> str:
    """Message from the JSON envelope (or a generic one), never the body verbatim."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return f"HTTP {response.status_code}: {detail}"
        if isinstance(detail, list):  # FastAPI validation errors
            items = [
                f"{'.'.join(str(p) for p in item.get('loc', []) if p != 'body')}: {item.get('msg')}"
                for item in detail
                if isinstance(item, dict)
            ]
            return f"HTTP {response.status_code}: " + "; ".join(items)
    return f"HTTP {response.status_code}"
