"""Pin the warning policy of pyproject.toml (``filterwarnings``).

Deprecations attributed to third-party modules are ignored because they depend
on the transitive versions resolved in each environment (anyio vs starlette,
pyparsing vs matplotlib); the same category attributed to ``src`` or ``tests``
is an error, and FutureWarning/UserWarning from the numeric stack stay errors
because they can change results.
"""

import sys
import types

import pytest


def _emit(modname: str, category: type[Warning]) -> str:
    module = types.ModuleType(modname)
    sys.modules[modname] = module
    code = compile("import warnings\nwarnings.warn('probe', CAT)", modname, "exec")
    try:
        exec(code, {"__name__": modname, "CAT": category})  # noqa: S102
    except Warning:
        return "ERRO"
    return "ignorado"


@pytest.mark.parametrize(
    ("modname", "category", "expected"),
    [
        ("anyio._lazyimport", DeprecationWarning, "ignorado"),
        ("pyparsing.util", DeprecationWarning, "ignorado"),
        ("src.models.train", DeprecationWarning, "ERRO"),
        ("tests.unit.x", DeprecationWarning, "ERRO"),
        ("sklearn.base", FutureWarning, "ERRO"),
        ("pandas.core.frame", FutureWarning, "ERRO"),
        ("pydantic._internal._fields", UserWarning, "ERRO"),
        ("sklearn.linear_model", UserWarning, "ERRO"),
    ],
)
def test_policy(modname: str, category: type[Warning], expected: str) -> None:
    assert _emit(modname, category) == expected
