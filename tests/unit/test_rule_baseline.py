"""Clinical rule baseline: counting, threshold, score and configuration errors."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import RuleBaselineConfig, RuleFactor, Settings
from src.data.mapping import VariableMapping
from src.exceptions import TransformError
from src.models.rule_baseline import apply_rule, describe

_FRAME = pd.DataFrame(
    {
        "age_group": ["18-29", "60-69", "70-79", "30-39"],
        "bmi_class": ["1", "3", "2", "1"],
        "smoking_history": ["2", "1", "2", "2"],
    }
)
_CONFIG = RuleBaselineConfig(
    factors=[
        RuleFactor(variable="age_group", label_pt="Idade", categories=["60-69", "70-79", "80+"]),
        RuleFactor(variable="bmi_class", label_pt="IMC", categories=["2", "3", "4"]),
        RuleFactor(variable="smoking_history", label_pt="Tabagismo", categories=["1"]),
    ],
    positive_from=2,
)


def test_counts_one_point_per_factor_present() -> None:
    result = apply_rule(_FRAME, _CONFIG)
    assert result.factor_counts.tolist() == [0, 3, 2, 0]


def test_score_is_the_share_of_factors_and_stays_in_the_unit_interval() -> None:
    result = apply_rule(_FRAME, _CONFIG)
    assert result.scores.tolist() == [0.0, 1.0, pytest.approx(2 / 3), 0.0]
    assert result.scores.min() >= 0.0
    assert result.scores.max() <= 1.0


def test_label_follows_the_declared_threshold() -> None:
    assert apply_rule(_FRAME, _CONFIG).labels.tolist() == [0, 1, 1, 0]
    strict = _CONFIG.model_copy(update={"positive_from": 3})
    assert apply_rule(_FRAME, strict).labels.tolist() == [0, 1, 0, 0]


def test_categories_compare_as_strings_even_when_the_frame_holds_numbers() -> None:
    numeric = pd.DataFrame({"bmi_class": [2, 3, 1]})
    config = RuleBaselineConfig(
        factors=[RuleFactor(variable="bmi_class", label_pt="IMC", categories=["2", "3"])],
        positive_from=1,
    )
    assert apply_rule(numeric, config).labels.tolist() == [1, 1, 0]


def test_unknown_variable_is_a_domain_error_not_a_key_error() -> None:
    config = RuleBaselineConfig(
        factors=[RuleFactor(variable="nao_existe", label_pt="X", categories=["1"])],
        positive_from=1,
    )
    with pytest.raises(TransformError, match="unknown model variables"):
        apply_rule(_FRAME, config)


def test_threshold_above_the_number_of_factors_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="positive_from"):
        RuleBaselineConfig(
            factors=[RuleFactor(variable="a", label_pt="A", categories=["1"])], positive_from=2
        )


def test_description_names_every_factor(cfg: Settings) -> None:
    text = describe(cfg.model.rule_baseline)
    for factor in cfg.model.rule_baseline.factors:
        assert factor.label_pt in text
        assert factor.variable in text


def test_project_factors_exist_in_the_mapping(cfg: Settings, mapping: VariableMapping) -> None:
    """The configured rule must refer to variables the transformer actually produces."""
    by_name = {variable.en: variable for variable in mapping.model_variables}
    for factor in cfg.model.rule_baseline.factors:
        assert factor.variable in by_name, factor.variable
        assert set(factor.categories) <= set(by_name[factor.variable].categories), factor.variable
