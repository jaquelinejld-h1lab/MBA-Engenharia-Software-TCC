"""Typed access to ``configs/variable_mapping.yaml``.

The mapping is the single source of variable names (PNS code, original
Portuguese name, English code name), domains and categories. Every module
that needs a column name or a category list reads it from here; nothing is
hardcoded twice.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from src.exceptions import ConfigError
from src.paths import CONFIG_DIR, resolve

DEFAULT_MAPPING_FILE: Path = CONFIG_DIR / "variable_mapping.yaml"


class _Frozen(BaseModel):
    # protected_namespaces=(): VariableMapping.model_variables starts with "model_",
    # which pydantic < 2.10 reserves (warning, error under the pytest filter).
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class Domain(_Frozen):
    """Valid domain of a raw variable: a categorical code list or a numeric range."""

    type: Literal["categorical", "numeric"]
    values: list[int] | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_shape(self) -> Domain:
        if self.type == "categorical" and not self.values:
            raise ValueError("categorical domain requires 'values'")
        if self.type == "numeric" and (self.min is None or self.max is None):
            raise ValueError("numeric domain requires 'min' and 'max'")
        return self


class ObservedStats(_Frozen):
    """Descriptive statistics observed on the development set (UI defaults)."""

    min: float | None = None
    max: float | None = None
    median: float | None = None
    mode: float | None = None


class AuxiliaryVariable(_Frozen):
    """Column used only to filter the population or build the target."""

    pns: str
    pt: str
    en: str
    role: Literal["filter", "target"]
    domain: Domain


class RawVariable(_Frozen):
    """One of the 61 raw inputs of the inference path."""

    pns: str
    pt: str
    en: str
    kind: Literal["questionnaire", "anthropometry", "laboratory"]
    dtype: Literal["int", "float"]
    block: Literal["socio", "anthro", "lifestyle", "health", "lab"]
    label_pt: str
    unit: str | None
    nullable: bool
    null_share_observed: float
    null_share_max: float
    domain: Domain
    observed: ObservedStats
    value_labels_pt: dict[int, str] | None = None

    @model_validator(mode="after")
    def _check_labels(self) -> RawVariable:
        if self.domain.type == "categorical":
            if self.value_labels_pt is None:
                raise ValueError(f"{self.en}: categorical variable requires 'value_labels_pt'")
            if set(self.value_labels_pt) != set(self.domain.values or []):
                raise ValueError(f"{self.en}: value labels must cover exactly the domain codes")
        elif self.value_labels_pt is not None:
            raise ValueError(f"{self.en}: numeric variable cannot have 'value_labels_pt'")
        return self


class IntermediateVariable(_Frozen):
    """Derived variable used only as KNN imputation base."""

    pt: str
    en: str
    categories: list[str]
    note: str


class Dummy(_Frozen):
    """One one-hot column of the encoded design matrix."""

    en: str
    notebook_name: str
    category: str


class ModelVariable(_Frozen):
    """One of the 33 categorical variables consumed by the model."""

    pt: str
    en: str
    label_pt: str
    categories: list[str]
    reference_category: str
    source_raw: list[str]
    dummies: list[Dummy]

    @model_validator(mode="after")
    def _check_dummies(self) -> ModelVariable:
        if self.reference_category != self.categories[0]:
            raise ValueError(f"{self.en}: reference category must be the first category")
        expected = [f"{self.en}_{c}" for c in self.categories[1:]]
        if [d.en for d in self.dummies] != expected:
            raise ValueError(f"{self.en}: dummies must be {expected}")
        return self


class VariableMapping(_Frozen):
    """Whole mapping file."""

    version: int
    description: str
    auxiliary_variables: list[AuxiliaryVariable]
    raw_variables: list[RawVariable]
    intermediate_variables: list[IntermediateVariable]
    model_variables: list[ModelVariable]

    # ---- convenience views -------------------------------------------------
    @property
    def raw_pns_to_en(self) -> dict[str, str]:
        """PNS code to English name for the 61 raw inference variables."""
        return {v.pns: v.en for v in self.raw_variables}

    @property
    def auxiliary_pns_to_en(self) -> dict[str, str]:
        """PNS code to English name for filter and target columns."""
        return {v.pns: v.en for v in self.auxiliary_variables}

    @property
    def raw_en_names(self) -> list[str]:
        """English names of the raw inference variables, in mapping order."""
        return [v.en for v in self.raw_variables]

    @property
    def model_en_names(self) -> list[str]:
        """English names of the 33 model variables, in mapping order."""
        return [v.en for v in self.model_variables]

    @property
    def model_categories(self) -> dict[str, list[str]]:
        """Explicit, ordered category list per model variable."""
        return {v.en: list(v.categories) for v in self.model_variables}

    @property
    def dummy_names(self) -> list[str]:
        """English dummy names in encoder output order."""
        return [d.en for v in self.model_variables for d in v.dummies]

    @property
    def dummy_en_to_notebook(self) -> dict[str, str]:
        """English dummy name to the name the original notebook produced."""
        return {d.en: d.notebook_name for v in self.model_variables for d in v.dummies}

    def raw(self, en: str) -> RawVariable:
        """Return the raw variable named ``en``.

        Raises
        ------
        ConfigError
            If no raw variable has that name.
        """
        for v in self.raw_variables:
            if v.en == en:
                return v
        raise ConfigError(f"raw variable '{en}' is not declared in the variable mapping")

    def model_variable(self, en: str) -> ModelVariable:
        """Return the model variable named ``en``.

        Raises
        ------
        ConfigError
            If no model variable has that name.
        """
        for v in self.model_variables:
            if v.en == en:
                return v
        raise ConfigError(f"model variable '{en}' is not declared in the variable mapping")


def load_mapping(mapping_file: str | Path | None = None) -> VariableMapping:
    """Load and validate the variable mapping.

    Parameters
    ----------
    mapping_file : str or Path, optional
        YAML file; defaults to ``configs/variable_mapping.yaml``.

    Returns
    -------
    VariableMapping

    Raises
    ------
    ConfigError
        If the file is missing or does not satisfy the schema.
    """
    path = resolve(mapping_file) if mapping_file is not None else DEFAULT_MAPPING_FILE
    if not path.is_file():
        raise ConfigError(f"Variable mapping file not found: '{path}'")
    with path.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    try:
        return VariableMapping.model_validate(payload)
    except ValidationError as error:
        issues = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()
        )
        raise ConfigError(f"Invalid variable mapping '{path}': {issues}") from error


@lru_cache(maxsize=1)
def get_mapping() -> VariableMapping:
    """Return the default mapping, loaded once per process."""
    return load_mapping()
