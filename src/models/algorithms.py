"""Estimator factories for every algorithm declared in ``configs/config.yaml``.

Parameters come verbatim from the run configuration; this module only
resolves the few symbolic values the YAML cannot express (``none``, ``auto``
class weights) and applies the project seed. Libraries outside scikit-learn
are imported lazily so that an environment without them (the API image, for
instance) can still import the package; a run whose library is missing
raises ``ModelNotFoundError`` with the library name.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from src.exceptions import ModelNotFoundError

_NONE_TOKEN = "none"  # noqa: S105  (YAML token, not a secret)
_AUTO_TOKEN = "auto"  # noqa: S105


def positive_class_weight(y: pd.Series | np.ndarray) -> float:
    """Ratio negatives / positives, as ``scale_pos_weight`` in the original notebook."""
    values = np.asarray(y)
    positives = int((values == 1).sum())
    if positives == 0:
        raise ModelNotFoundError("cannot compute class weight: no positive examples")
    return float((values == 0).sum() / positives)


def _resolve(params: dict[str, Any], y: pd.Series | np.ndarray) -> dict[str, Any]:
    """Replace YAML tokens: ``none`` -> None, ``auto`` -> negatives/positives, lists -> tuples."""
    resolved: dict[str, Any] = {}
    for key, value in params.items():
        if value == _NONE_TOKEN:
            resolved[key] = None
        elif value == _AUTO_TOKEN and key == "scale_pos_weight":
            resolved[key] = positive_class_weight(y)
        elif value == _AUTO_TOKEN and key == "class_weights":
            resolved[key] = [1.0, positive_class_weight(y)]
        elif isinstance(value, list):
            resolved[key] = tuple(value)
        else:
            resolved[key] = value
    return resolved


def _xgboost(params: dict[str, Any], seed: int) -> ClassifierMixin:
    try:
        from xgboost import XGBClassifier  # noqa: PLC0415  (optional dependency)
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ModelNotFoundError("xgboost is not installed in this environment") from error
    return XGBClassifier(**params, random_state=seed)


def _lightgbm(params: dict[str, Any], seed: int) -> ClassifierMixin:
    try:
        from lightgbm import LGBMClassifier  # noqa: PLC0415  (optional dependency)
    except ImportError as error:  # pragma: no cover
        raise ModelNotFoundError("lightgbm is not installed in this environment") from error
    return LGBMClassifier(**params, random_state=seed)


def _catboost(params: dict[str, Any], seed: int) -> ClassifierMixin:
    try:
        from catboost import CatBoostClassifier  # noqa: PLC0415  (optional dependency)
    except ImportError as error:  # pragma: no cover
        raise ModelNotFoundError("catboost is not installed in this environment") from error
    return CatBoostClassifier(**params, random_state=seed)


_FACTORIES: dict[str, Callable[[dict[str, Any], int], ClassifierMixin]] = {
    "LogisticRegression": lambda p, s: LogisticRegression(**p, random_state=s),
    "RandomForestClassifier": lambda p, s: RandomForestClassifier(**p, random_state=s),
    "MLPClassifier": lambda p, s: MLPClassifier(**p, random_state=s),
    "SVC": lambda p, s: SVC(**p, random_state=s),
    "XGBClassifier": _xgboost,
    "LGBMClassifier": _lightgbm,
    "CatBoostClassifier": _catboost,
}

SUPPORTED_ALGORITHMS: tuple[str, ...] = tuple(_FACTORIES)


def build_estimator(
    algorithm: str, params: dict[str, Any], seed: int, y: pd.Series | np.ndarray
) -> ClassifierMixin:
    """Instantiate a classifier from its config name and parameters.

    Parameters
    ----------
    algorithm : str
        One of :data:`SUPPORTED_ALGORITHMS`.
    params : dict
        Parameters as declared in the YAML (tokens are resolved here).
    seed : int
        Project seed passed as ``random_state``.
    y : array-like
        Training labels, needed to resolve ``auto`` class weights.

    Raises
    ------
    ModelNotFoundError
        If the algorithm is unknown or its library is not installed.
    """
    try:
        factory = _FACTORIES[algorithm]
    except KeyError as error:
        raise ModelNotFoundError(
            f"unknown algorithm '{algorithm}'; supported: {list(_FACTORIES)}"
        ) from error
    return factory(_resolve(params, y), seed)


def library_available(algorithm: str) -> bool:
    """Return whether the library backing ``algorithm`` can be imported."""
    module = {
        "XGBClassifier": "xgboost",
        "LGBMClassifier": "lightgbm",
        "CatBoostClassifier": "catboost",
    }
    name = module.get(algorithm)
    if name is None:
        return algorithm in _FACTORIES
    # The probe answers "can this library be imported", nothing else. Third-party
    # wheels warn at import time about matters outside this project (catboost 1.2.7
    # raises a numpy ABI RuntimeWarning), and under the pytest policy that turns every
    # warning into an error those warnings would be read as an unavailable library.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            __import__(name)
        except ImportError:
            return False
    return True
