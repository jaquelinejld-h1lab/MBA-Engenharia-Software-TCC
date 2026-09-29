"""HypertensionTransformer: shape, modes, idempotency, single row, serialization."""

from __future__ import annotations

import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.config import Settings
from src.data.contract import ModelRecord, enforce_contract
from src.data.mapping import VariableMapping
from src.exceptions import TransformError
from src.features.transformer import HypertensionTransformer

Split = tuple[pd.DataFrame, pd.DataFrame]


@pytest.fixture(scope="module")
def fitted(
    cfg: Settings, mapping: VariableMapping, synthetic_split: Split
) -> HypertensionTransformer:
    train, _ = synthetic_split
    return HypertensionTransformer(cfg.features, mapping, mode="production").fit(train)


def test_output_has_85_named_dummies(
    fitted: HypertensionTransformer, mapping: VariableMapping, synthetic_split: Split
) -> None:
    _, test = synthetic_split
    out = fitted.transform(test)
    assert out.shape == (len(test), 85)
    assert list(out.columns) == mapping.dummy_names
    assert list(fitted.get_feature_names_out()) == mapping.dummy_names
    assert set(np.unique(out.to_numpy())) <= {0, 1}
    assert out.index.equals(test.index)


def test_transform_is_idempotent(fitted: HypertensionTransformer, synthetic_split: Split) -> None:
    _, test = synthetic_split
    pd.testing.assert_frame_equal(fitted.transform(test), fitted.transform(test))


def test_single_row_equals_batch_row(
    fitted: HypertensionTransformer, synthetic_split: Split
) -> None:
    _, test = synthetic_split
    batch = fitted.transform(test)
    for position in (0, len(test) // 2, len(test) - 1):
        single = fitted.transform(test.iloc[[position]])
        pd.testing.assert_frame_equal(single, batch.iloc[[position]])


def test_model_frame_satisfies_contract(
    fitted: HypertensionTransformer, synthetic_split: Split
) -> None:
    _, test = synthetic_split
    result = fitted.transform_detailed(test)
    enforce_contract(result.model_frame, ModelRecord, "model frame")
    assert result.imputed.columns.tolist() == ["egfr_afro", "cholesterol", "glucose"]
    assert result.imputed.dtypes.eq(bool).all()


def test_imputed_fields_reported(fitted: HypertensionTransformer, synthetic_split: Split) -> None:
    _, test = synthetic_split
    row = test.iloc[[0]].copy()
    row["glucose"] = math.nan
    result = fitted.transform_detailed(row)
    assert result.imputed_fields(0) == ["glucose"]
    assert result.model_frame["glucose_class"].iloc[0] in {"1", "2", "3", "4"}


def test_joblib_roundtrip(
    fitted: HypertensionTransformer, synthetic_split: Split, tmp_path: Path
) -> None:
    _, test = synthetic_split
    path = tmp_path / "transformer.joblib"
    joblib.dump(fitted, path)
    loaded = joblib.load(path)
    pd.testing.assert_frame_equal(loaded.transform(test), fitted.transform(test))


def test_clone_keeps_parameters(fitted: HypertensionTransformer) -> None:
    copy = clone(fitted)
    assert copy.mode == "production"
    assert copy.features == fitted.features
    with pytest.raises(TransformError, match="not fitted"):
        copy.transform(pd.DataFrame())


def test_fidelity_mode_refits_on_each_frame(
    cfg: Settings, mapping: VariableMapping, synthetic_split: Split
) -> None:
    train, test = synthetic_split
    fidelity = HypertensionTransformer(cfg.features, mapping, mode="fidelity").fit(train)
    production = HypertensionTransformer(cfg.features, mapping, mode="production").fit(train)
    # on the training frame both modes coincide (imputers fitted on the same rows)
    pd.testing.assert_frame_equal(fidelity.transform(train), production.transform(train))
    # on another frame, only laboratory dummies may differ
    diff = fidelity.transform(test).ne(production.transform(test)).any(axis=0)
    lab_prefixes = ("egfr_class_", "glucose_class_", "cholesterol_desirable_")
    assert all(col.startswith(lab_prefixes) for col in diff[diff].index)


def test_unknown_category_raises_transform_error(
    fitted: HypertensionTransformer, synthetic_split: Split
) -> None:
    _, test = synthetic_split
    row = test.iloc[[0]].copy()
    row["age"] = 10  # below the first age bin -> NaN age_group -> encoder error
    with pytest.raises(TransformError):
        fitted.transform(row)


def test_invalid_mode_rejected(
    cfg: Settings, mapping: VariableMapping, synthetic_split: Split
) -> None:
    train, _ = synthetic_split
    with pytest.raises(TransformError, match="mode"):
        HypertensionTransformer(cfg.features, mapping, mode="bogus").fit(train)  # type: ignore[arg-type]
