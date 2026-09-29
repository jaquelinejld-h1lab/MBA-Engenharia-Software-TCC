"""Domain exception hierarchy."""

from __future__ import annotations

import pytest

from src.exceptions import (
    ConfigError,
    DataContractError,
    HypertensionPipelineError,
    ModelNotFoundError,
    PredictionError,
    TransformError,
)


@pytest.mark.parametrize(
    "exc",
    [ConfigError, DataContractError, TransformError, ModelNotFoundError, PredictionError],
)
def test_all_domain_errors_share_base(exc: type[Exception]) -> None:
    assert issubclass(exc, HypertensionPipelineError)
    with pytest.raises(HypertensionPipelineError):
        raise exc("message")
