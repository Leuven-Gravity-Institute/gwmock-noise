"""Tests for generic component-based configuration and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gwmock_noise.config import NoiseComponentConfig, NoiseConfig, OutputConfig, load_config


def test_noise_config_defaults() -> None:
    """NoiseConfig defaults to one white-noise component."""
    config = NoiseConfig()
    assert config.detectors == ["H1", "L1"]
    assert config.duration == 4.0
    assert config.sampling_frequency == 4096.0
    assert config.output.directory == Path(".")
    assert config.output.prefix == "noise"
    assert config.output.format == "npy"
    assert config.output.gps_start == 0.0
    assert config.output.channel_prefix == "MOCK"
    assert config.seed is None
    assert config.components == [NoiseComponentConfig(simulator="white", options={})]


def test_noise_config_custom_values() -> None:
    """NoiseConfig accepts custom runtime values and component lists."""
    config = NoiseConfig(
        detectors=["H1", "L1", "V1"],
        duration=8.0,
        sampling_frequency=2048.0,
        output=OutputConfig(
            directory=Path("out"),
            prefix="run1",
            format="gwf",
            gps_start=1234567890.5,
            channel_prefix="GWMOCK",
        ),
        seed=123,
        components=[
            {"simulator": "colored", "psd_file": "noise_psd.txt"},
            {"simulator": "spectral_lines", "lines": [{"frequency": 60.0, "amplitude": 1.0e-3}]},
        ],
    )
    assert config.detectors == ["H1", "L1", "V1"]
    assert config.duration == 8.0
    assert config.sampling_frequency == 2048.0
    assert config.output.directory == Path("out")
    assert config.output.prefix == "run1"
    assert config.output.format == "gwf"
    assert config.output.gps_start == 1234567890.5
    assert config.output.channel_prefix == "GWMOCK"
    assert config.seed == 123
    assert [component.simulator for component in config.components] == ["colored", "spectral_lines"]
    assert config.components[0].options == {"psd_file": "noise_psd.txt"}


def test_component_config_accepts_string_shorthand() -> None:
    """String component definitions normalize to an empty-options component."""
    config = NoiseConfig(components=["white"])
    assert config.components == [NoiseComponentConfig(simulator="white", options={})]


def test_component_config_accepts_flat_mapping() -> None:
    """Flat component mappings are normalized into ``options``."""
    component = NoiseComponentConfig.model_validate(
        {"simulator": "colored", "psd_file": "flat_psd.txt", "low_frequency_cutoff": 8.0}
    )
    assert component.simulator == "colored"
    assert component.options == {"psd_file": "flat_psd.txt", "low_frequency_cutoff": 8.0}


def test_component_config_merges_explicit_options_with_flat_fields() -> None:
    """Explicit ``options`` mappings merge with other non-reserved keys."""
    component = NoiseComponentConfig.model_validate(
        {
            "simulator": "colored",
            "options": {"psd_file": "flat_psd.txt"},
            "high_frequency_cutoff": 128.0,
        }
    )
    assert component.options == {"psd_file": "flat_psd.txt", "high_frequency_cutoff": 128.0}


def test_component_config_rejects_duplicate_explicit_option_keys() -> None:
    """The same option cannot appear both inside and outside ``options``."""
    with pytest.raises(ValidationError, match="duplicate explicit fields"):
        NoiseComponentConfig.model_validate(
            {
                "simulator": "colored",
                "options": {"psd_file": "flat_psd.txt"},
                "psd_file": "other_psd.txt",
            }
        )


def test_component_config_requires_simulator_name() -> None:
    """Components must name a simulator."""
    with pytest.raises(ValidationError, match="define a simulator name"):
        NoiseComponentConfig.model_validate({"psd_file": "flat_psd.txt"})


def test_component_config_rejects_non_mapping_options() -> None:
    """Explicit ``options`` must be a mapping."""
    with pytest.raises(ValidationError, match="component options must be a mapping"):
        NoiseComponentConfig.model_validate({"simulator": "white", "options": 123})


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


def test_noise_config_rejects_unknown_output_format() -> None:
    """OutputConfig exposes only the supported artifact formats."""
    with pytest.raises(ValidationError, match=r"npy|gwf"):
        NoiseConfig.model_validate({"output": {"format": "json"}})


def test_load_config_supports_nested_noise_key_and_component_list(tmp_path: Path) -> None:
    """Loader unwraps a nested ``noise`` section and preserves components."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[noise]
detectors = ["H1"]
duration = 2.0

[[noise.components]]
simulator = "white"

[[noise.components]]
simulator = "spectral_lines"
lines = [{ frequency = 32.0, amplitude = 0.01 }]
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.detectors == ["H1"]
    assert config.duration == 2.0
    assert [component.simulator for component in config.components] == ["white", "spectral_lines"]
    assert config.components[1].options["lines"] == [{"frequency": 32.0, "amplitude": 0.01}]


def test_load_config_supports_yaml_components(tmp_path: Path) -> None:
    """Loader accepts YAML component definitions."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
detectors: [H1]
components:
  - simulator: colored
    psd_file: flat_psd.txt
  - white
""",
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert [component.simulator for component in config.components] == ["colored", "white"]
    assert config.components[0].options["psd_file"] == "flat_psd.txt"


def test_load_config_supports_json_components(tmp_path: Path) -> None:
    """Loader accepts JSON component definitions."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "detectors": ["H1"],
                "components": [{"simulator": "glitches", "models": []}],
            }
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)
    assert config.components[0].simulator == "glitches"
    assert config.components[0].options["models"] == []
