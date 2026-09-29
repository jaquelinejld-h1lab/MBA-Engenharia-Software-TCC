"""Global reading of a linear model: coefficients, importance and contributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data.mapping import VariableMapping
from src.models.interpretation import build_interpretation

_Fitted = tuple[LogisticRegression, pd.DataFrame, pd.DataFrame]


class _NoCoefficients:
    """Stand-in for a tree or network model."""


def _design(mapping: VariableMapping, rows: int = 40) -> pd.DataFrame:
    """Deterministic design matrix with the real dummy names."""
    dummies = [dummy.en for variable in mapping.model_variables for dummy in variable.dummies]
    rng = np.random.default_rng(42)
    values = rng.integers(0, 2, size=(rows, len(dummies)))
    return pd.DataFrame(values, columns=dummies, dtype=float)


@pytest.fixture
def fitted(mapping: VariableMapping) -> _Fitted:
    train = _design(mapping, rows=60)
    holdout = _design(mapping, rows=25)
    y = pd.Series(([0] * 40) + ([1] * 20))
    model = LogisticRegression(max_iter=200, random_state=42).fit(train, y)
    return model, train, holdout


def test_returns_none_without_coefficients(mapping: VariableMapping) -> None:
    train = _design(mapping, rows=5)
    assert (
        build_interpretation(_NoCoefficients(), train, train, mapping, top_dummies=3, max_rows=5)
        is None
    )


def test_one_coefficient_per_column(fitted: _Fitted, mapping: VariableMapping) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=5, max_rows=10)
    assert payload is not None
    assert [row["dummy"] for row in payload["coefficients"]] == list(train.columns)
    assert payload["n_train"] == len(train)


def test_importance_covers_every_model_variable_and_is_sorted(
    fitted: _Fitted, mapping: VariableMapping
) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=5, max_rows=10)
    assert payload is not None
    names = [row["model_variable"] for row in payload["importance"]]
    assert set(names) == {variable.en for variable in mapping.model_variables}
    values = [row["mean_abs_contribution"] for row in payload["importance"]]
    assert values == sorted(values, reverse=True)


def test_importance_of_a_variable_sums_its_dummies(
    fitted: _Fitted, mapping: VariableMapping
) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=5, max_rows=10)
    assert payload is not None
    by_variable = {row["model_variable"]: row for row in payload["importance"]}
    target = mapping.model_variables[0]
    expected = sum(
        row["mean_abs_contribution"]
        for row in payload["coefficients"]
        if row["model_variable"] == target.en
    )
    assert by_variable[target.en]["mean_abs_contribution"] == pytest.approx(expected)


def test_contribution_equals_coefficient_times_value(
    fitted: _Fitted, mapping: VariableMapping
) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=4, max_rows=100)
    assert payload is not None
    sample = payload["contributions"]
    coefficients = {row["dummy"]: row["coefficient"] for row in payload["coefficients"]}
    for column, dummy in enumerate(sample["dummies"]):
        observed = sample["contributions"][0][column]
        expected = coefficients[dummy] * float(holdout.iloc[0][dummy])
        assert observed == pytest.approx(expected, abs=1e-4)


def test_sample_is_capped_and_reports_what_it_left_out(
    fitted: _Fitted, mapping: VariableMapping
) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=3, max_rows=10)
    assert payload is not None
    sample = payload["contributions"]
    assert sample["rows"] <= 10
    assert sample["rows_available"] == len(holdout)
    assert len(sample["dummies"]) == 3
    assert len(sample["contributions"]) == sample["rows"]
    assert all(len(row) == 3 for row in sample["contributions"])


def test_present_flag_matches_the_design_matrix(fitted: _Fitted, mapping: VariableMapping) -> None:
    model, train, holdout = fitted
    payload = build_interpretation(model, train, holdout, mapping, top_dummies=3, max_rows=100)
    assert payload is not None
    sample = payload["contributions"]
    for column, dummy in enumerate(sample["dummies"]):
        assert sample["present"][0][column] == int(holdout.iloc[0][dummy] != 0)


def test_mismatched_matrix_is_rejected(fitted: _Fitted, mapping: VariableMapping) -> None:
    model, train, holdout = fitted
    with pytest.raises(ValueError, match="coefficients"):
        build_interpretation(model, train.iloc[:, :-1], holdout, mapping, top_dummies=2, max_rows=5)
