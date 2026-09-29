"""Configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config import Settings, load_config
from src.exceptions import ConfigError
from src.paths import DEFAULT_CONFIG_FILE, PROJECT_ROOT


def test_default_config_loads_and_is_typed() -> None:
    cfg = load_config()
    assert isinstance(cfg, Settings)
    assert cfg.project.seed == 42
    assert cfg.split.n_splits == 10
    assert cfg.model.decision_threshold == 0.5
    assert cfg.features.income.minimum_wage_brl == 678.00
    assert cfg.features.laboratory.knn_imputer_neighbors == 5
    assert len(cfg.data.raw_sha256) == 64


def test_inventory_numbers_are_present() -> None:
    cfg = load_config()
    assert cfg.population_filters.expected_removed == [1016, 101, 118, 1501, 6, 7]
    assert cfg.data.expected_rows_after_filters == 6203
    assert cfg.target.systolic_threshold_mmhg == 140
    assert cfg.target.diastolic_threshold_mmhg == 90
    assert cfg.fidelity.auc_tolerance == 0.005
    assert cfg.fidelity.sensitivity_tolerance == 0.01


def test_paths_resolve_under_project_root() -> None:
    cfg = load_config()
    raw = cfg.paths.absolute("raw_dataset")
    assert raw.is_absolute()
    assert PROJECT_ROOT in raw.parents


def test_config_is_frozen() -> None:
    cfg = load_config()
    with pytest.raises(ValidationError):
        cfg.project.seed = 1  # type: ignore[misc]


def test_missing_key_raises_informative_error(tmp_path: Path) -> None:
    with DEFAULT_CONFIG_FILE.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    del content["project"]["seed"]
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(broken)
    assert "project.seed" in str(excinfo.value)
    assert "Field required" in str(excinfo.value)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    with DEFAULT_CONFIG_FILE.open(encoding="utf-8") as handle:
        content = yaml.safe_load(handle)
    content["project"]["typo_key"] = 1
    broken = tmp_path / "config.yaml"
    broken.write_text(yaml.safe_dump(content), encoding="utf-8")

    with pytest.raises(ConfigError) as excinfo:
        load_config(broken)
    assert "project.typo_key" in str(excinfo.value)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTN_LOGGING__LEVEL", "DEBUG")
    cfg = load_config()
    assert cfg.logging.level == "DEBUG"
