r"""Freeze the stage-2 fidelity reference by re-executing the original notebook code.

The gate in ``tests/integration/test_notebook_fidelity.py`` compares the refactored
pre-processing against numbers produced by ``notebooks/original/funcoes_mecai_v3.py``.
Those numbers are **environment dependent**: ``impute_knn`` uses
``sklearn.impute.KNNImputer``, whose neighbour selection changes by a few rows between
scikit-learn and NumPy releases, and rows sitting on a bin edge then fall into a
different class. A reference frozen in one environment is therefore only a valid
contract for that environment.

Run this inside the pinned environment, which is the training image::

    docker compose -f docker/docker-compose.yml --profile tools run --rm train \\
        python scripts/freeze_notebook_reference.py

The resulting JSON records the interpreter and library versions that produced it, so a
later divergence can be told apart from a refactoring bug: same versions means a real
regression, different versions means the reference has to be frozen again here.

Nothing in the project's runtime imports this script; it is a maintenance tool.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Settings, load_config
from src.data.mapping import VariableMapping, load_mapping
from src.exceptions import DataContractError

ORIGINAL_DIR = Path(__file__).resolve().parents[1] / "notebooks" / "original"
TARGET = "target"


USED_FUNCTIONS = ("carregar_dividir_dados", "add_variaveis_ajustadas", "encode_data")
MAX_STUBS = 40


class _AbsentAttribute:
    """Anything read from an absent library: callable, chainable and inert.

    The original file runs ``sns.set(style="whitegrid")`` at import time, so a
    placeholder that is only a module would raise ``TypeError`` there. Every attribute
    of a placeholder is therefore an object that can be called, indexed or read further
    without doing anything.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: Any, **kwargs: Any) -> _AbsentAttribute:
        return _AbsentAttribute(f"{self._name}()")

    def __getattr__(self, name: str) -> _AbsentAttribute:
        return _AbsentAttribute(f"{self._name}.{name}")

    def __getitem__(self, key: Any) -> _AbsentAttribute:
        return _AbsentAttribute(f"{self._name}[...]")

    def __setitem__(self, key: Any, value: Any) -> None:
        return None

    def __repr__(self) -> str:
        return f"<absent {self._name}>"


class _AbsentModule(ModuleType):
    """Placeholder for a library the original file imports but the used functions never call."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        # Declared as a package so that ``from x.y import z`` raises ModuleNotFoundError
        # for ``x.y`` (which the loop then stubs) instead of TypeError in the import machinery.
        self.__path__: list[str] = []

    def __getattr__(self, name: str) -> _AbsentAttribute:
        return _AbsentAttribute(f"{self.__name__}.{name}")


def _import_original() -> tuple[Any, list[str]]:
    """Import ``funcoes_mecai_v3`` from ``notebooks/original``.

    The original file imports the whole modelling stack at module level (SMOTE, Boruta,
    SHAP, plotting). Only three of its functions are needed here, and none of them
    touches those libraries, so a missing one is replaced by a placeholder instead of
    being added back to the pinned environment. Every placeholder is returned and
    recorded in the JSON, and the three functions are checked afterwards.

    Returns
    -------
    tuple
        The imported module and the sorted names of the placeholders that were needed.

    Raises
    ------
    DataContractError
        If the file is missing, if more placeholders than ``MAX_STUBS`` are required, or
        if one of the used functions turns out to depend on a placeholder.
    """
    if not (ORIGINAL_DIR / "funcoes_mecai_v3.py").is_file():
        raise DataContractError(f"original functions not found in {ORIGINAL_DIR}")
    sys.path.insert(0, str(ORIGINAL_DIR))

    stubbed: list[str] = []
    for _ in range(MAX_STUBS):
        try:
            module = importlib.import_module("funcoes_mecai_v3")
        except ModuleNotFoundError as error:
            missing = error.name
            if not missing or missing == "funcoes_mecai_v3":
                raise DataContractError(f"cannot import the original functions: {error}") from error
            sys.modules[missing] = _AbsentModule(missing)
            stubbed.append(missing)
            continue
        _assert_used_functions_are_real(module, stubbed)
        return module, sorted(stubbed)
    raise DataContractError(f"more than {MAX_STUBS} missing libraries; install them instead")


def _assert_used_functions_are_real(module: Any, stubbed: list[str]) -> None:
    """Fail loudly if a needed function is missing or reads a placeholder global."""
    for name in USED_FUNCTIONS:
        function = getattr(module, name, None)
        if not callable(function):
            raise DataContractError(f"the original module has no function '{name}'")
        for global_name in function.__code__.co_names:
            value = getattr(module, global_name, None)
            if isinstance(value, _AbsentModule | _AbsentAttribute):
                raise DataContractError(
                    f"'{name}' uses '{global_name}', which is a placeholder for a missing "
                    f"library (stubbed: {stubbed}); install that library and run again"
                )


def _patch_one_hot_encoder(module: Any) -> list[str]:
    """Accept the original ``OneHotEncoder(sparse=False)`` on scikit-learn 1.2 and later.

    This is the single compatibility change applied to the original code, and it is the
    one already recorded in the ``source`` field of the reference: the keyword was
    renamed to ``sparse_output`` with identical behaviour. The patch lives in the
    module's namespace so the file on disk stays untouched.
    """
    encoder_class = module.OneHotEncoder
    if "sparse" in inspect.signature(encoder_class.__init__).parameters:
        return []

    def factory(*args: Any, **kwargs: Any) -> Any:
        if "sparse" in kwargs:
            kwargs["sparse_output"] = kwargs.pop("sparse")
        return encoder_class(*args, **kwargs)

    module.OneHotEncoder = factory
    return ["OneHotEncoder(sparse=) -> OneHotEncoder(sparse_output=)"]


def _category_key(value: Any) -> str:
    """Category label as the project stores it.

    The original code leaves the untouched codes as floats, so ``value_counts`` keys come
    back as ``2.0`` where the refactored derivations produce the string ``"2"``. The
    integral float is written as an integer so both sides compare on equal terms; the
    counts themselves are untouched.
    """
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _describe(
    frame: pd.DataFrame, original: Any, mapping: VariableMapping, keep_index: bool
) -> dict[str, Any]:
    """Derive, encode and summarise one set exactly as the notebook did.

    Parameters
    ----------
    frame : pandas.DataFrame
        Development, train or test set as returned by ``carregar_dividir_dados``.
    original : module
        The imported original module.
    mapping : VariableMapping
        Project mapping: ``pt`` is the column name the notebook used, ``en`` ours.
    keep_index : bool
        Whether to store the row index (it pins the train/test split).

    Returns
    -------
    dict
        Same shape the fidelity test expects under ``sets[<name>]``.
    """
    derived = original.add_variaveis_ajustadas(frame.copy())
    notebook_names = [variable.pt for variable in mapping.model_variables]
    selected = derived[[*notebook_names, TARGET]]
    encoded = original.encode_data(selected, TARGET)

    matrix = encoded.drop(columns=[TARGET]).to_numpy().astype(int)
    distributions = {
        variable.en: {
            _category_key(key): int(count)
            for key, count in derived[variable.pt].value_counts().items()
        }
        for variable in mapping.model_variables
    }
    return {
        "rows": len(frame),
        "encoded_columns": int(encoded.shape[1]),
        "target_positive": int(frame[TARGET].sum()),
        "row_index": [int(i) for i in frame.index] if keep_index else None,
        "encoded_matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
        "distributions": distributions,
    }


def build_reference(cfg: Settings, mapping: VariableMapping) -> dict[str, Any]:
    """Re-execute the original pipeline over the real extract and collect the numbers."""
    original, stubbed = _import_original()
    patches = _patch_one_hot_encoder(original)
    raw_path = cfg.paths.absolute("raw_dataset")
    if not raw_path.is_file():
        raise DataContractError(f"real PNS 2013 extract not found at {raw_path}")

    development, train, test = original.carregar_dividir_dados(str(raw_path), cfg.split.test_size)
    sets = {
        # The development set keeps the positional index rebuilt by the original code, so
        # storing it adds nothing; train and test carry the shuffled index produced by
        # train_test_split, which is what pins the split itself.
        "development": _describe(development, original, mapping, keep_index=False),
        "train": _describe(train, original, mapping, keep_index=True),
        "test": _describe(test, original, mapping, keep_index=True),
    }
    return {
        "source": "funcoes_mecai_v3.py original functions re-executed "
        "(OneHotEncoder sparse->sparse_output only)",
        "stubbed_libraries": stubbed,
        "compatibility_patches": patches,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "sets": sets,
    }


def main(argv: list[str] | None = None) -> int:
    """Write the reference JSON, or compare it with the committed one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write: report whether the committed reference matches this environment",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    mapping = load_mapping()
    reference = build_reference(cfg, mapping)
    path = cfg.paths.absolute("evidence_dir") / "fase0_referencia_notebook_etapa2.json"

    if args.check:
        committed = json.loads(path.read_text(encoding="utf-8"))
        same = committed.get("sets") == reference["sets"]
        print(f"committed environment: {committed.get('environment')}")
        print(f"this environment:      {reference['environment']}")
        print("sets identical" if same else "sets DIFFER: freeze the reference again here")
        return 0 if same else 1

    path.write_text(json.dumps(reference, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(f"environment: {reference['environment']}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
