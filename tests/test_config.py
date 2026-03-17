"""Tests for configuration models and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gwsim_noise.config import NoiseConfig, OutputConfig, load_config


def test_noise_config_defaults() -> None:
    """NoiseConfig uses sensible defaults when minimal fields are provided."""
    config = NoiseConfig()
    assert config.detectors == ["H1", "L1"]
    assert config.duration == 4.0
    assert config.sampling_frequency == 4096.0
    assert config.output.directory == Path(".")
    assert config.output.prefix == "noise"
    assert config.seed is None


def test_noise_config_custom_values() -> None:
    """NoiseConfig accepts custom values."""
    config = NoiseConfig(
        detectors=["H1", "L1", "V1"],
        duration=8.0,
        sampling_frequency=2048.0,
        output=OutputConfig(directory=Path("out"), prefix="run1"),
        seed=123,
    )
    assert config.detectors == ["H1", "L1", "V1"]
    assert config.duration == 8.0
    assert config.sampling_frequency == 2048.0
    assert config.output.directory == Path("out")
    assert config.output.prefix == "run1"
    assert config.seed == 123


def test_noise_config_validates_duration() -> None:
    """NoiseConfig rejects non-positive duration."""
    with pytest.raises(ValidationError, match="greater than 0"):
        NoiseConfig(duration=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        NoiseConfig(duration=-1.0)


def test_noise_config_validates_detectors() -> None:
    """NoiseConfig requires at least one detector."""
    with pytest.raises(ValidationError, match="at least 1"):
        NoiseConfig(detectors=[])


def test_load_config_yaml(tmp_path: Path) -> None:
    """load_config loads and validates YAML files."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
detectors: [H1, L1]
duration: 4.0
sampling_frequency: 4096.0
output:
  directory: ./output
  prefix: test
seed: 42
"""
    )
    config = load_config(config_file)
    assert config.detectors == ["H1", "L1"]
    assert config.duration == 4.0
    assert config.output.directory == Path("./output")
    assert config.seed == 42


def test_load_config_json(tmp_path: Path) -> None:
    """load_config loads and validates JSON files."""
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps(
            {
                "detectors": ["H1"],
                "duration": 2.0,
                "sampling_frequency": 1024.0,
            }
        )
    )
    config = load_config(config_file)
    assert config.detectors == ["H1"]
    assert config.duration == 2.0
    assert config.sampling_frequency == 1024.0


def test_load_config_toml(tmp_path: Path) -> None:
    """load_config loads and validates TOML files."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        """
detectors = ["H1", "L1"]
duration = 4.0
sampling_frequency = 4096.0

[output]
directory = "./output"
prefix = "test"

seed = 42
"""
    )
    config = load_config(config_file)
    assert config.detectors == ["H1", "L1"]
    assert config.duration == 4.0
    assert config.output.directory == Path("./output")


def test_load_config_nested_noise_key(tmp_path: Path) -> None:
    """load_config extracts noise section when nested under 'noise' key."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
noise:
  detectors: [V1]
  duration: 1.0
  sampling_frequency: 2048.0
"""
    )
    config = load_config(config_file)
    assert config.detectors == ["V1"]
    assert config.duration == 1.0


def test_load_config_file_not_found() -> None:
    """load_config raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("/nonexistent/path/config.yaml")


def test_load_config_unsupported_format(tmp_path: Path) -> None:
    """load_config raises ValueError for unsupported file formats."""
    config_file = tmp_path / "config.txt"
    config_file.write_text("detectors: [H1]")
    with pytest.raises(ValueError, match="Unsupported config format"):
        load_config(config_file)
