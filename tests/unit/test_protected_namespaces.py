"""Every field named ``model_*`` must come from a model that disables the namespace.

Pydantic reserves the ``model_`` prefix for its own API and warns whenever a field uses
it. Under the project's warning policy that warning is an error, so a single model
declared without ``protected_namespaces=()`` breaks collection of every test that
imports the module, far from where the mistake is.

The check is written against the requirement instead of against the installed pydantic,
because the default changed between releases: in 2.9 the prefix is protected and in
newer versions it is not. Asserting on the configuration keeps the suite honest in the
pinned environment even when the environment running the test is more permissive.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

import pytest
from pydantic import BaseModel

MODULES = (
    "src.api.schemas",
    "src.config",
    "src.data.contract",
    "src.data.mapping",
)


def _models() -> list[tuple[str, type[BaseModel]]]:
    """Every pydantic model declared in the project modules listed above."""
    found: list[tuple[str, type[BaseModel]]] = []
    for name in MODULES:
        module: Any = importlib.import_module(name)
        for attribute, value in vars(module).items():
            if (
                inspect.isclass(value)
                and issubclass(value, BaseModel)
                and value is not BaseModel
                and value.__module__ == name
            ):
                found.append((f"{name}.{attribute}", value))
    return found


def test_the_scan_finds_models() -> None:
    """Guard against the check silently passing because it found nothing."""
    assert len(_models()) > 20


@pytest.mark.parametrize(
    "qualified,model", _models(), ids=lambda value: getattr(value, "__name__", value)
)
def test_model_prefixed_fields_disable_the_protected_namespace(
    qualified: str, model: type[BaseModel]
) -> None:
    reserved = [field for field in model.model_fields if field.startswith("model_")]
    if not reserved:
        return
    assert model.model_config.get("protected_namespaces") == (), (
        f"{qualified} declares {reserved} and must set protected_namespaces=() , "
        "otherwise pydantic 2.9 warns at import time and the warning policy turns it "
        "into a collection error"
    )
