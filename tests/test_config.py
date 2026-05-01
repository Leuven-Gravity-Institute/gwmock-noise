"""Tests for configuration models and loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from gwmock_noise.config import NoiseConfig, OutputConfig, load_config


def test_noise_config_defaults() -> None:
    """NoiseConfig uses sensible defaults when minimal fields are provided."""
    config = NoiseConfig()
    assert config.detectors == ["H1", "L1"]
    assert config.duration == 4.0
    assert config.sampling_frequency == 4096.0
    assert config.output.directory == Path(".")
    assert config.output.prefix == "noise"
    assert config.seed is None
    assert config.psd_file is None
    assert config.psd_schedule is None
    assert config.psd_files is None
    assert config.csd_files is None
    assert config.low_frequency_cutoff == 2.0
    assert config.high_frequency_cutoff is None


def test_noise_config_custom_values() -> None:
    """NoiseConfig accepts custom values."""
    config = NoiseConfig(
        detectors=["H1", "L1", "V1"],
        duration=8.0,
        sampling_frequency=2048.0,
        output=OutputConfig(directory=Path("out"), prefix="run1"),
        seed=123,
        psd_files={"H1": Path("h1_psd.txt"), "L1": Path("l1_psd.txt"), "V1": Path("v1_psd.txt")},
        csd_files={
            "H1-L1": Path("h1_l1_csd.txt"),
            "H1-V1": Path("h1_v1_csd.txt"),
            "L1-V1": Path("l1_v1_csd.txt"),
        },
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=512.0,
    )
    assert config.detectors == ["H1", "L1", "V1"]
    assert config.duration == 8.0
    assert config.sampling_frequency == 2048.0
    assert config.output.directory == Path("out")
    assert config.output.prefix == "run1"
    assert config.seed == 123
    assert config.psd_file is None
    assert config.psd_files == {"H1": Path("h1_psd.txt"), "L1": Path("l1_psd.txt"), "V1": Path("v1_psd.txt")}
    assert config.csd_files == {
        "H1-L1": Path("h1_l1_csd.txt"),
        "H1-V1": Path("h1_v1_csd.txt"),
        "L1-V1": Path("l1_v1_csd.txt"),
    }
    assert config.low_frequency_cutoff == 8.0
    assert config.high_frequency_cutoff == 512.0


def test_noise_config_validates_duration() -> None:
    """NoiseConfig rejects non-positive duration."""
    with pytest.raises(ValidationError, match="greater than 0"):
        NoiseConfig(duration=0)
    with pytest.raises(ValidationError, match="greater than 0"):
        NoiseConfig(duration=-1.0)


def test_noise_config_accepts_time_varying_psd_schedule() -> None:
    """NoiseConfig accepts a sorted time-varying PSD schedule."""
    config = NoiseConfig(
        detectors=["H1"],
        psd_schedule=[
            (0.0, Path("psd_start.txt")),
            (128.0, Path("psd_end.txt")),
        ],
    )

    assert config.psd_schedule == [
        (0.0, Path("psd_start.txt")),
        (128.0, Path("psd_end.txt")),
    ]


def test_noise_config_validates_detectors() -> None:
    """NoiseConfig requires at least one detector."""
    with pytest.raises(ValidationError, match="at least 1"):
        NoiseConfig(detectors=[])


def test_noise_config_validates_cutoff_ordering() -> None:
    """NoiseConfig requires high_frequency_cutoff to be greater than low_frequency_cutoff."""
    with pytest.raises(ValidationError, match="greater than low_frequency_cutoff"):
        NoiseConfig(
            sampling_frequency=1024.0,
            low_frequency_cutoff=128.0,
            high_frequency_cutoff=128.0,
        )


def test_noise_config_validates_cutoffs_are_not_above_nyquist() -> None:
    """NoiseConfig rejects cutoff values above the Nyquist frequency."""
    with pytest.raises(ValidationError, match="low_frequency_cutoff must be <= Nyquist"):
        NoiseConfig(sampling_frequency=1024.0, low_frequency_cutoff=600.0)

    with pytest.raises(ValidationError, match="high_frequency_cutoff must be <= Nyquist"):
        NoiseConfig(
            sampling_frequency=1024.0,
            low_frequency_cutoff=16.0,
            high_frequency_cutoff=600.0,
        )


def test_noise_config_rejects_mixed_single_and_multi_detector_psd_inputs() -> None:
    """NoiseConfig rejects simultaneous psd_file and psd_files inputs."""
    with pytest.raises(ValidationError, match="mutually exclusive"):
        NoiseConfig(
            detectors=["H1"],
            psd_file=Path("shared_psd.txt"),
            psd_files={"H1": Path("h1_psd.txt")},
        )


def test_noise_config_rejects_mixed_time_varying_and_multi_detector_psd_inputs() -> None:
    """NoiseConfig rejects simultaneous psd_schedule and psd_files inputs."""
    with pytest.raises(ValidationError, match="psd_schedule and psd_files are mutually exclusive"):
        NoiseConfig(
            detectors=["H1"],
            psd_schedule=[(0.0, Path("shared_psd.txt"))],
            psd_files={"H1": Path("h1_psd.txt")},
        )


@pytest.mark.parametrize(
    ("psd_schedule", "message"),
    [
        (
            [(16.0, Path("late.txt")), (0.0, Path("early.txt"))],
            "psd_schedule entries must be sorted by GPS offset",
        ),
        (
            [(0.0, Path("first.txt")), (0.0, Path("second.txt"))],
            "psd_schedule entries must use distinct GPS offsets",
        ),
    ],
)
def test_noise_config_validates_time_varying_psd_schedule(
    psd_schedule: list[tuple[float, Path]],
    message: str,
) -> None:
    """NoiseConfig validates PSD schedule ordering and uniqueness."""
    with pytest.raises(ValidationError, match=message):
        NoiseConfig(detectors=["H1"], psd_schedule=psd_schedule)


def test_noise_config_requires_per_detector_psds_for_csd_inputs() -> None:
    """NoiseConfig requires psd_files when csd_files are configured."""
    with pytest.raises(ValidationError, match="requires psd_files"):
        NoiseConfig(
            detectors=["H1", "L1"],
            csd_files={"H1-L1": Path("h1_l1_csd.txt")},
        )


def test_noise_config_validates_psd_file_keys_match_detectors() -> None:
    """NoiseConfig requires psd_files keys to exactly match detectors."""
    with pytest.raises(ValidationError, match="must exactly match detectors"):
        NoiseConfig(
            detectors=["H1", "L1"],
            psd_files={"H1": Path("h1_psd.txt")},
        )


@pytest.mark.parametrize(
    ("csd_files", "message"),
    [
        ({"badkey": Path("bad.txt")}, "DET1-DET2"),
        ({"H1-H1": Path("duplicate.txt")}, "two distinct detectors"),
        ({"H1-L1": Path("one.txt"), "L1-H1": Path("two.txt")}, "duplicate detector pairs"),
        ({"H1-V1": Path("missing.txt")}, "configured detectors"),
    ],
)
def test_noise_config_validates_csd_file_keys(
    csd_files: dict[str, Path],
    message: str,
) -> None:
    """NoiseConfig validates CSD detector-pair keys."""
    with pytest.raises(ValidationError, match=message):
        NoiseConfig(
            detectors=["H1", "L1"],
            psd_files={"H1": Path("h1_psd.txt"), "L1": Path("l1_psd.txt")},
            csd_files=csd_files,
        )


def test_noise_config_validator_defensive_negative_cutoff_branches() -> None:
    """Model validator defensive checks reject negative cutoff values."""
    low_negative = NoiseConfig.model_construct(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=1024.0,
        output=OutputConfig(),
        seed=None,
        psd_file=None,
        psd_files=None,
        csd_files=None,
        low_frequency_cutoff=-1.0,
        high_frequency_cutoff=256.0,
    )
    with pytest.raises(ValueError, match="low_frequency_cutoff must be >= 0"):
        low_negative.validate_frequency_cutoffs()

    high_negative = NoiseConfig.model_construct(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=1024.0,
        output=OutputConfig(),
        seed=None,
        psd_file=None,
        psd_files=None,
        csd_files=None,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=-1.0,
    )
    with pytest.raises(ValueError, match="high_frequency_cutoff must be >= 0"):
        high_negative.validate_frequency_cutoffs()


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


def test_load_config_empty_yaml_uses_defaults(tmp_path: Path) -> None:
    """load_config treats empty YAML as an empty config mapping."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    config = load_config(config_file)
    assert config == NoiseConfig()


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
