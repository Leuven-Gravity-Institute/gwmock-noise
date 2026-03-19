"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

import tomllib
import yaml

from gwmock_noise.config.models import NoiseConfig


def load_config(path: Path | str) -> NoiseConfig:
    """Load and validate noise configuration from a file.

    Supports TOML, YAML, and JSON formats.

    Args:
        path: Path to the configuration file.

    Returns:
        Validated NoiseConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the file format is unsupported or validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".toml":
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    elif suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .toml, .yaml, .yml, or .json.")

    if data is None:
        data = {}

    # Allow config to be nested under a "noise" key for composition with larger configs
    if "noise" in data and isinstance(data["noise"], dict):
        data = data["noise"]

    return NoiseConfig.model_validate(data)
