"""Domain exceptions.

Every failure raised by the pipeline, the API or the app derives from
``HypertensionPipelineError`` so callers can catch the family without
swallowing unrelated errors. Messages must say what failed and what was
expected; never raise a bare ``Exception``.
"""

from __future__ import annotations


class HypertensionPipelineError(Exception):
    """Base class for all domain errors of this project."""


class ConfigError(HypertensionPipelineError):
    """Configuration file missing, malformed or failing validation."""


class DataContractError(HypertensionPipelineError):
    """Input data violates the declared schema, hash or quality gate."""


class TransformError(HypertensionPipelineError):
    """A feature derivation or the fitted transformer could not be applied."""


class ModelNotFoundError(HypertensionPipelineError):
    """The requested model name, version or stage does not exist in the registry."""


class PredictionError(HypertensionPipelineError):
    """The model was loaded but could not score the given observation(s)."""
