"""Typed, validated configuration loaded from ``configs/config.yaml``.

The YAML file is the single source of every tunable parameter. Values can be
overridden with environment variables prefixed ``HTN_`` and nested with
``__`` (for example ``HTN_LOGGING__LEVEL=DEBUG``). Unknown keys and missing
keys both fail loudly with a ``ConfigError`` that names the offending field.

Examples
--------
>>> from src.config import load_config
>>> cfg = load_config()
>>> cfg.project.seed
42
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from src.exceptions import ConfigError
from src.paths import DEFAULT_CONFIG_FILE, resolve


class _Strict(BaseModel):
    """Base for config sections: unknown keys are rejected, values are frozen."""

    # protected_namespaces=(): pydantic < 2.10 warns on fields named model_* (this
    # project has model_stage and model_source); the warning would be an error under
    # the pytest filter. The API contract keeps these names on purpose.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class ProjectConfig(_Strict):
    """Project identity and the single random seed."""

    name: str
    seed: int = Field(ge=0)


class LoggingConfig(_Strict):
    """Log level and output format."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    json_format: bool


class PathsConfig(_Strict):
    """Project-relative locations of data, artifacts and evidence."""

    raw_dataset: Path
    interim_dir: Path
    processed_dir: Path
    evidence_dir: Path
    golden_metrics: Path
    fase0_reference_metrics: Path
    variable_mapping: Path
    mlflow_tracking_uri: str
    artifacts_dir: Path

    def absolute(self, attribute: str) -> Path:
        """Return one of the path attributes resolved against the project root.

        Parameters
        ----------
        attribute : str
            Name of a ``Path`` attribute of this section.

        Returns
        -------
        Path
            Absolute path.

        Raises
        ------
        ConfigError
            If ``attribute`` is not a path attribute of this section.
        """
        value = getattr(self, attribute, None)
        if not isinstance(value, Path):
            raise ConfigError(f"'{attribute}' is not a path attribute of paths config")
        return resolve(value)


class DataConfig(_Strict):
    """Identity and expected dimensions of the raw dataset."""

    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_sheet: str
    expected_raw_rows: int = Field(gt=0)
    expected_raw_columns: int = Field(gt=0)
    expected_rows_after_filters: int = Field(gt=0)
    source: str
    freeze_date: str


class PopulationFiltersConfig(_Strict):
    """Columns, values and expected counts of the six population filters."""

    consent_column: str
    consent_keep_value: int
    pregnancy_column: str
    pregnancy_drop_values: list[int]
    hypertension_pregnancy_only_column: str
    hypertension_pregnancy_only_drop_value: int
    medication_column: str
    medication_drop_value: int
    region_column: str
    empty_questionnaire_proxy_column: str
    expected_removed: list[int]


class TargetConfig(_Strict):
    """Operational definition of the hypertension target."""

    systolic_column: str
    diastolic_column: str
    systolic_threshold_mmhg: float = Field(gt=0)
    diastolic_threshold_mmhg: float = Field(gt=0)
    drop_if_blood_pressure_missing: bool
    expected_prevalence: float = Field(gt=0, lt=1)


class SplitConfig(_Strict):
    """Holdout split and cross-validation design."""

    test_size: float = Field(gt=0, lt=1)
    stratify: bool
    n_splits: int = Field(ge=2)
    cv_on_full_development_set: bool


class IncomeConfig(_Strict):
    """Income aggregation and minimum-wage brackets."""

    minimum_wage_brl: float = Field(gt=0)
    bracket_multipliers: list[float]
    missing_income_as_zero: bool


class BinnedConfig(_Strict):
    """Generic bin edges and labels for a ``pd.cut`` derivation."""

    bin_edges: list[float]
    labels: list[str]


class BmiConfig(_Strict):
    """BMI class cut points and unit conversion."""

    underweight_below: float
    overweight_from: float
    obesity_from: float
    severe_obesity_from: float
    height_unit_divisor: float = Field(gt=0)


class WaistConfig(_Strict):
    """Waist circumference risk cut points by sex."""

    female_increased_cm: float
    female_greatly_increased_cm: float
    male_increased_cm: float
    male_greatly_increased_cm: float


class PhysicalActivityConfig(_Strict):
    """Weekly minutes thresholds and exercise type groups."""

    minutes_per_exercise_day: int = Field(gt=0)
    low_activity_below_min: int
    vigorous_min_minutes: int
    moderate_min_minutes: int
    moderate_to_vigorous_min_minutes: int
    moderate_exercise_types: list[int]
    vigorous_exercise_types: list[int]


class FoodFrequencyConfig(_Strict):
    """Weekly food frequency cut points."""

    max_days_per_week: int = Field(ge=1)
    healthy_very_days: int
    healthy_min_days: int
    moderate_min_days: int
    fish_very_healthy_min_days: int
    juice_edges: list[int]
    sweets_unhealthy_from_days: int
    meal_replacement_unhealthy_from_days: int


class LaboratoryConfig(_Strict):
    """KNN imputation and laboratory exam categorization."""

    knn_imputer_neighbors: int = Field(ge=1)
    egfr_bin_edges: list[float]
    egfr_labels: list[str]
    cholesterol_desirable_below_mg_dl: float
    glucose_bin_edges: list[float]
    glucose_labels: list[str]
    imputation_base_variables: list[str]


class SmokingConfig(_Strict):
    """Smoking history derivation switches (see P1)."""

    reproduce_string_int_comparison_bug: bool


class EncodingConfig(_Strict):
    """One-hot encoding options and expected width."""

    drop_first: bool
    expected_encoded_columns: int = Field(gt=0)


class FeaturesConfig(_Strict):
    """All feature engineering parameters."""

    income: IncomeConfig
    age: BinnedConfig
    household_size: BinnedConfig
    bmi: BmiConfig
    waist: WaistConfig
    physical_activity: PhysicalActivityConfig
    food_frequency: FoodFrequencyConfig
    laboratory: LaboratoryConfig
    smoking: SmokingConfig
    encoding: EncodingConfig


class ChampionParams(_Strict):
    """Hyperparameters of the champion estimator."""

    C: float = Field(gt=0)
    solver: str
    class_weight: str
    max_iter: int = Field(gt=0)


class ChampionConfig(_Strict):
    """Champion algorithm, hyperparameters and scaling flag."""

    algorithm: str
    params: ChampionParams
    scale_features: bool


class RiskBandsConfig(_Strict):
    """Probability cut points for the risk bands shown in the UI."""

    low_below: float = Field(ge=0, le=1)
    high_from: float = Field(ge=0, le=1)


class RunConfig(_Strict):
    """One experiment run: algorithm, transformer mode, feature variant and params."""

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    algorithm: str
    transformer_mode: Literal["fidelity", "production"]
    smoking_bug_reproduced: bool
    role: Literal["fidelity", "champion", "comparison", "baseline", "candidate"]
    params: dict[str, Any]


class RuleFactor(_Strict):
    """One clinical risk factor of the rule baseline."""

    variable: str
    label_pt: str
    categories: list[str] = Field(min_length=1)


class RuleBaselineConfig(_Strict):
    """Clinical rule used as the floor every trained model has to beat."""

    factors: list[RuleFactor] = Field(min_length=1)
    positive_from: int = Field(ge=1)

    @model_validator(mode="after")
    def _threshold_within_factors(self) -> RuleBaselineConfig:
        if self.positive_from > len(self.factors):
            raise ValueError(
                f"positive_from ({self.positive_from}) exceeds the number of factors "
                f"({len(self.factors)}): the rule could never predict risk"
            )
        return self


class InterpretationConfig(_Strict):
    """Size of the global reading persisted next to each linear model."""

    top_dummies: int = Field(ge=1)
    max_contribution_rows: int = Field(ge=1)


class ModelConfig(_Strict):
    """Champion definition, decision threshold, transformer mode and experiment runs."""

    champion: ChampionConfig
    decision_threshold: float = Field(gt=0, lt=1)
    risk_bands: RiskBandsConfig
    transformer_mode: Literal["fidelity", "production"]
    interpretation: InterpretationConfig
    rule_baseline: RuleBaselineConfig
    runs: list[RunConfig]

    @model_validator(mode="after")
    def _unique_run_names(self) -> ModelConfig:
        names = [r.name for r in self.runs]
        if len(names) != len(set(names)):
            raise ValueError("run names must be unique")
        if sum(r.role == "fidelity" for r in self.runs) != 1:
            raise ValueError("exactly one run must have role 'fidelity'")
        return self

    def run(self, name: str) -> RunConfig:
        """Return the run configuration named ``name``."""
        for r in self.runs:
            if r.name == name:
                return r
        raise ConfigError(f"run '{name}' is not declared in model.runs")


class TrackingConfig(_Strict):
    """Experiment tracking and model registry settings."""

    backend: Literal["mlflow", "json"]
    experiment_name: str
    registered_model_name: str
    sqlite_path: Path
    artifact_root: Path
    json_runs_dir: Path
    promote_run: str
    promote_stage: str
    promote_alias: str


class FidelityConfig(_Strict):
    """Reference metrics and tolerances of the fidelity test."""

    auc_tolerance: float = Field(gt=0)
    sensitivity_tolerance: float = Field(gt=0)
    reference_auc_cv: float = Field(gt=0, lt=1)
    reference_sensitivity_cv: float = Field(gt=0, lt=1)


class AcceptanceConfig(_Strict):
    """Acceptance criteria of specific objective OE5."""

    min_auc: float
    min_sensitivity: float
    max_p95_latency_ms: int
    slo_p95_latency_ms: int
    subgroup_min_size: int = Field(ge=1)


class DriftConfig(_Strict):
    """PSI and KS thresholds."""

    psi_warning: float = Field(gt=0)
    psi_alert: float = Field(gt=0)
    ks_statistic_alert: float = Field(gt=0)
    ks_pvalue_alert: float = Field(gt=0, lt=1)
    psi_bins: int = Field(ge=2)
    epsilon: float = Field(gt=0, lt=1)
    ks_min_rows: int = Field(ge=2)


class NumericShift(_Strict):
    """Deterministic shift of a numeric variable in a drift scenario."""

    add: float = 0.0
    multiply: float = 1.0


class CategoricalOverride(_Strict):
    """Force ``share`` of the rows of a categorical variable to ``code``."""

    code: int
    share: float = Field(gt=0, le=1)


class ScenarioConfig(_Strict):
    """One reproducible drift scenario applied to the holdout window."""

    description: str
    expect_alerts: bool
    numeric_shifts: dict[str, NumericShift]
    categorical_overrides: dict[str, CategoricalOverride]


class RetrainingConfig(_Strict):
    """When drift is strong enough to recommend training the model again."""

    min_variables_in_alert: int = Field(ge=1)
    flag_file: str


class MonitoringConfig(_Strict):
    """Drift detection settings, the API observation window and the scenarios."""

    drift: DriftConfig
    retraining: RetrainingConfig
    reference_window: Literal["train"]
    reference_file: str
    window_size: int = Field(ge=1)
    min_rows_for_drift: int = Field(ge=2)
    scenario_fixture_rows: int = Field(ge=1)
    scenarios: dict[str, ScenarioConfig]

    @model_validator(mode="after")
    def _window_holds_minimum(self) -> MonitoringConfig:
        if self.min_rows_for_drift > self.window_size:
            raise ValueError("monitoring.min_rows_for_drift cannot exceed window_size")
        return self


class RiskBandLabels(_Strict):
    """Portuguese labels of the three risk bands."""

    low: str
    moderate: str
    high: str


class ApiConfig(_Strict):
    """API binding, model source, batch limits and clinical disclaimer."""

    host: str
    port: int = Field(ge=1, le=65535)
    model_stage: str
    model_source: Literal["mlflow", "local"]
    local_run_dir: Path
    # Credencial opcional. Nunca vem do YAML: apenas de HTN_API__AUTH_TOKEN (variavel de
    # ambiente local ou GitHub Secret). SecretStr evita que apareca em log ou repr.
    auth_token: SecretStr | None = None
    batch_max_records: int = Field(ge=1)
    top_contributions: int = Field(ge=1)
    top_coefficients: int = Field(ge=1)
    roc_max_points: int = Field(ge=2)
    risk_band_labels: RiskBandLabels
    clinical_disclaimer: str


class AppConfig(_Strict):
    """Streamlit client settings (the app never loads the model)."""

    api_url: str
    # Credencial opcional para falar com a API protegida. So de HTN_APP__API_TOKEN.
    api_token: SecretStr | None = None
    prometheus_url: str
    request_timeout_seconds: float = Field(gt=0)
    batch_chunk_size: int = Field(ge=1)
    highlighted_variables: list[str]
    top_contributions: int = Field(ge=1)


class LoadTestConfig(_Strict):
    """Defaults of ``scripts/load_test.py``."""

    requests: int = Field(ge=1)
    concurrency: int = Field(ge=1)
    warmup_requests: int = Field(ge=0)
    sample_interval_seconds: float = Field(gt=0)
    # Custo horario declarado da maquina que serve a API, usado apenas para converter
    # duracao em custo por mil requisicoes. E premissa do estudo, nao medicao.
    cost_per_hour: float = Field(ge=0)
    cost_currency: str
    cost_basis: str


class Settings(BaseSettings):
    """Root configuration object.

    Sources, in order of precedence: init kwargs, environment variables
    (``HTN_`` prefix, ``__`` nesting), then the YAML file.
    """

    model_config = SettingsConfigDict(
        env_prefix="HTN_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )

    project: ProjectConfig
    logging: LoggingConfig
    paths: PathsConfig
    data: DataConfig
    population_filters: PopulationFiltersConfig
    target: TargetConfig
    split: SplitConfig
    features: FeaturesConfig
    model: ModelConfig
    tracking: TrackingConfig
    fidelity: FidelityConfig
    acceptance: AcceptanceConfig
    monitoring: MonitoringConfig
    api: ApiConfig
    app: AppConfig
    load_test: LoadTestConfig

    @model_validator(mode="after")
    def _promoted_run_exists(self) -> Settings:
        names = {r.name for r in self.model.runs}
        if self.tracking.promote_run not in names:
            raise ValueError(f"tracking.promote_run '{self.tracking.promote_run}' is not a run")
        if self.app.batch_chunk_size > self.api.batch_max_records:
            raise ValueError("app.batch_chunk_size cannot exceed api.batch_max_records")
        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Add the YAML file as the lowest-precedence source."""
        yaml_file = settings_cls.model_config.get("yaml_file") or DEFAULT_CONFIG_FILE
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=yaml_file),
        )


def _format_validation_error(error: ValidationError, config_file: Path) -> str:
    """Turn a pydantic error into one line per problem, naming the YAML key."""
    lines = [f"Invalid configuration in '{config_file}':"]
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "<root>"
        lines.append(f"  - {location}: {issue['msg']} ({issue['type']})")
    return "\n".join(lines)


def load_config(config_file: str | Path | None = None) -> Settings:
    """Load and validate the project configuration.

    Parameters
    ----------
    config_file : str or Path, optional
        YAML file to load. Defaults to ``configs/config.yaml`` under the
        project root. Relative paths are resolved against the project root.

    Returns
    -------
    Settings
        Frozen, fully validated configuration.

    Raises
    ------
    ConfigError
        If the file does not exist, is not valid YAML, has unknown keys or is
        missing required keys. The message lists every offending key.
    """
    path = resolve(config_file) if config_file is not None else DEFAULT_CONFIG_FILE
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: '{path}'")

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(
            env_prefix="HTN_",
            env_nested_delimiter="__",
            extra="forbid",
            frozen=True,
            protected_namespaces=(),
            yaml_file=path,
        )

    try:
        return _FileSettings()
    except ValidationError as error:
        raise ConfigError(_format_validation_error(error, path)) from error


@lru_cache(maxsize=1)
def get_config() -> Settings:
    """Return the default configuration, loaded once per process."""
    return load_config()
