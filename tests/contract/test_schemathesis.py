"""Schema-driven contract test: schemathesis generates requests from the OpenAPI document."""

from __future__ import annotations

import pytest

from src.api.model_loader import ModelCache
from src.config import load_config
from src.data.filters import split_holdout
from src.data.mapping import load_mapping
from src.data.pipeline import build_development_frame
from src.data.synthetic import make_synthetic_frame
from tests.api_fixtures import loaded_from, train_fixture_pipeline

schemathesis = pytest.importorskip("schemathesis")
pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from src.api.main import create_app  # noqa: E402

try:
    from schemathesis.exceptions import SchemaError
except ImportError:  # module layout changed (schemathesis 4.x)
    SchemaError = Exception  # type: ignore[misc,assignment]

# Generation budget: schemathesis 3.x reuses ``hypothesis.settings`` directly and only
# re-exports it as ``schemathesis.settings`` from 4.x onwards, so resolve whichever
# exists instead of assuming one of the two layouts.
_hypothesis_settings = pytest.importorskip("hypothesis").settings
case_settings = getattr(schemathesis, "settings", _hypothesis_settings)

# Each generated case drives the app through a fresh starlette TestClient, and
# schemathesis 3.x does not close the anyio memory streams that client opens. The
# resulting ResourceWarning is emitted from ``__del__``; ``-p no:unraisableexception``
# in pyproject.toml keeps it from failing whichever test triggers the collector.
pytestmark = pytest.mark.contract


def _build_app():  # type: ignore[no-untyped-def]
    cfg = load_config()
    mapping = load_mapping()
    development, _ = build_development_frame(
        make_synthetic_frame(cfg.project.seed), cfg, enforce_counts=False
    )
    train, _ = split_holdout(development, cfg.split, cfg.project.seed)
    return create_app(
        cfg, model=loaded_from(train_fixture_pipeline(cfg, mapping, train)), cache=ModelCache()
    )


def _load_schema():  # type: ignore[no-untyped-def]
    """Load the OpenAPI document into schemathesis, or skip when the tool cannot read it.

    FastAPI 0.115 emits OpenAPI 3.1; schemathesis 3.x only fully supports 3.0 and
    refuses 3.1 unless told to treat it as 3.0 (``force_schema_version="30"``).
    The contract itself is also covered by ``test_api_contract.py`` with the
    FastAPI TestClient, so a schemathesis limitation must not block the pipeline;
    it is reported as a skip with the reason, never silently.
    """
    app = _build_app()
    try:
        return schemathesis.from_asgi("/openapi.json", app, force_schema_version="30")
    except TypeError:  # older/newer schemathesis without the keyword
        pass
    except SchemaError as error:
        pytest.skip(
            f"schemathesis cannot read this OpenAPI document: {error}", allow_module_level=True
        )
    try:
        return schemathesis.from_asgi("/openapi.json", app)
    except SchemaError as error:
        pytest.skip(
            f"schemathesis cannot read this OpenAPI document: {error}", allow_module_level=True
        )


schema = _load_schema()


@schema.parametrize()
@case_settings(max_examples=25, deadline=None)
def test_openapi_contract(case) -> None:  # type: ignore[no-untyped-def]
    """Every generated request must be answered per the schema (no 5xx, valid bodies)."""
    case.call_and_validate()
