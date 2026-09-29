"""Configuration validators and lookup errors not covered by test_config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Settings, load_config
from src.exceptions import ConfigError
from src.paths import DEFAULT_CONFIG_FILE


def _write_variant(tmp_path: Path, mutate) -> Path:  # type: ignore[no-untyped-def]
    content = yaml.safe_load(DEFAULT_CONFIG_FILE.read_text(encoding="utf-8"))
    mutate(content)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


def test_run_lookup_and_paths_absolute_errors(cfg: Settings) -> None:
    with pytest.raises(ConfigError, match="run 'nope'"):
        cfg.model.run("nope")
    with pytest.raises(ConfigError, match="not a path attribute"):
        cfg.paths.absolute("mlflow_tracking_uri")
    assert cfg.paths.absolute("evidence_dir").name == "evidencias"


def test_duplicate_run_names_rejected(tmp_path: Path) -> None:
    def mutate(c: dict) -> None:  # type: ignore[type-arg]
        c["model"]["runs"][1]["name"] = c["model"]["runs"][0]["name"]

    with pytest.raises(ConfigError, match="unique"):
        load_config(_write_variant(tmp_path, mutate))


def test_exactly_one_fidelity_run(tmp_path: Path) -> None:
    def mutate(c: dict) -> None:  # type: ignore[type-arg]
        c["model"]["runs"][1]["role"] = "fidelity"

    with pytest.raises(ConfigError, match="fidelity"):
        load_config(_write_variant(tmp_path, mutate))


def test_promoted_run_must_exist(tmp_path: Path) -> None:
    def mutate(c: dict) -> None:  # type: ignore[type-arg]
        c["tracking"]["promote_run"] = "ghost"

    with pytest.raises(ConfigError, match="promote_run"):
        load_config(_write_variant(tmp_path, mutate))


def test_invalid_yaml_type_reports_field(tmp_path: Path) -> None:
    def mutate(c: dict) -> None:  # type: ignore[type-arg]
        c["split"]["test_size"] = 1.5

    with pytest.raises(ConfigError, match=r"split\.test_size"):
        load_config(_write_variant(tmp_path, mutate))
