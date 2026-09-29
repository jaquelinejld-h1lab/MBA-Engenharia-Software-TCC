"""Stage-2 exit gate: the refactored pre-processing reproduces the notebook.

Reference values were frozen in ``evidencias/fase0_referencia_notebook_etapa2.json``
by ``scripts/freeze_notebook_reference.py``, which re-executes the original
``funcoes_mecai_v3.py`` functions. With the environment that produced the reference,
any divergence here is a refactoring bug, never acceptable variation; that condition is
what ``test_reference_was_frozen_in_this_environment`` checks first.
"""

from __future__ import annotations

import hashlib
import json
import platform

import numpy as np
import pandas as pd
import pytest
import sklearn

from src.config import Settings
from src.data.filters import TARGET_COLUMN, split_holdout
from src.data.mapping import VariableMapping
from src.data.pipeline import build_development_frame, load_raw_frame
from src.features.transformer import HypertensionTransformer

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def reference(cfg: Settings) -> dict:  # type: ignore[type-arg]
    path = cfg.paths.absolute("evidence_dir") / "fase0_referencia_notebook_etapa2.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def real_sets(cfg: Settings) -> dict[str, pd.DataFrame]:
    raw_path = cfg.paths.absolute("raw_dataset")
    if not raw_path.is_file():
        pytest.skip("real PNS 2013 extract not available")
    raw = load_raw_frame(cfg, materialize=False)
    dev, report = build_development_frame(raw, cfg, enforce_counts=True)
    assert report.removed_per_filter == cfg.population_filters.expected_removed
    assert report.blood_pressure_missing_removed == 0
    train, test = split_holdout(dev, cfg.split, cfg.project.seed)
    return {"development": dev, "train": train, "test": test}


def test_reference_was_frozen_in_this_environment(reference: dict) -> None:  # type: ignore[type-arg]
    """The reference only binds the environment that produced it.

    ``impute_knn`` in the original code uses ``KNNImputer``; its neighbour selection
    changes by a few rows across scikit-learn and NumPy releases, and rows sitting on a
    bin edge then fall into a different class. Comparing against a reference frozen
    elsewhere would report a refactoring bug that does not exist, so the environment is
    checked first: if this fails, freeze the reference again in the pinned environment,
    which is the training image, and commit the result. Freezing it in a local Python
    of another version is what this check exists to catch.
    """
    frozen = reference.get("environment", {})
    running = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }
    shared = {key: running[key] for key in frozen if key in running}
    assert shared == {key: frozen[key] for key in shared}, (
        f"reference frozen with {frozen}, running with {running}. "
        "Freeze it again in the pinned environment, which is the training image: "
        "docker compose -f docker/docker-compose.yml --profile tools run --rm train "
        "python scripts/freeze_notebook_reference.py . "
        "Running the script outside that image (a local Python, another interpreter "
        "version) produces a reference that binds no environment the project uses."
    )


def test_population_and_split_counts(real_sets: dict[str, pd.DataFrame], cfg: Settings) -> None:
    dev = real_sets["development"]
    assert len(dev) == 6203
    assert len(real_sets["train"]) == 4962
    assert len(real_sets["test"]) == 1241
    assert round(dev[TARGET_COLUMN].mean(), 4) == cfg.target.expected_prevalence


@pytest.mark.parametrize("name", ["development", "train", "test"])
def test_encoded_matrix_is_identical_to_notebook(
    name: str,
    real_sets: dict[str, pd.DataFrame],
    reference: dict,  # type: ignore[type-arg]
    cfg: Settings,
    mapping: VariableMapping,
) -> None:
    frame = real_sets[name]
    expected = reference["sets"][name]
    features = frame.drop(columns=[TARGET_COLUMN])
    transformer = HypertensionTransformer(cfg.features, mapping, mode="fidelity").fit(features)
    result = transformer.transform_detailed(features)

    assert result.encoded.shape == (expected["rows"], expected["encoded_columns"] - 1)
    assert result.encoded.shape[1] + 1 == cfg.features.encoding.expected_encoded_columns
    if expected["row_index"] is not None:
        assert frame.index.tolist() == expected["row_index"]

    for variable in mapping.model_variables:
        observed = {
            str(k): int(v) for k, v in result.model_frame[variable.en].value_counts().items()
        }
        assert observed == expected["distributions"][variable.en], variable.en

    digest = hashlib.sha256(result.encoded.to_numpy().astype(int).tobytes()).hexdigest()
    assert digest == expected["encoded_matrix_sha256"]


def test_production_mode_differs_only_in_laboratory_dummies(
    real_sets: dict[str, pd.DataFrame], cfg: Settings, mapping: VariableMapping
) -> None:
    train = real_sets["train"].drop(columns=[TARGET_COLUMN])
    test = real_sets["test"].drop(columns=[TARGET_COLUMN])
    production = HypertensionTransformer(cfg.features, mapping, mode="production").fit(train)
    fidelity = HypertensionTransformer(cfg.features, mapping, mode="fidelity").fit(test)
    diff = production.transform(test).ne(fidelity.transform(test)).any(axis=0)
    changed = [c for c in diff[diff].index]
    assert all(
        c.startswith(("egfr_class_", "glucose_class_", "cholesterol_desirable_")) for c in changed
    )
