"""Tests for configuration models and loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from gwmock_noise.config import (
    BlipGlitch,
    GengliBlipGlitch,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
    ScatteredLightGlitch,
    SpectralLine,
    load_config,
)
from gwmock_noise.glitches.gengli import write_blip_population_file


def test_noise_config_defaults() -> None:
    """NoiseConfig uses sensible defaults when minimal fields are provided."""
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
    assert config.psd_file is None
    assert config.psd_schedule is None
    assert config.psd_files is None
    assert config.csd_files is None
    assert config.low_frequency_cutoff == 2.0
    assert config.high_frequency_cutoff is None
    assert config.spectral_lines is None
    assert config.glitches is None


def test_noise_config_custom_values() -> None:
    """NoiseConfig accepts custom values."""
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
    assert config.output.format == "gwf"
    assert config.output.gps_start == 1234567890.5
    assert config.output.channel_prefix == "GWMOCK"
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


def test_noise_config_rejects_unknown_output_format() -> None:
    """OutputConfig exposes a fixed set of artifact formats."""
    with pytest.raises(ValidationError, match=r"npy|gwf"):
        NoiseConfig.model_validate({"output": {"format": "json"}})


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


def test_noise_config_accepts_spectral_lines_from_mappings() -> None:
    """NoiseConfig parses configured spectral-line mappings into dataclasses."""
    config = NoiseConfig.model_validate(
        {
            "detectors": ["H1"],
            "spectral_lines": [
                {"frequency": 60.0, "amplitude": 1.5e-23},
                {"frequency": 120.0, "amplitude": 5.0e-24, "phase": 0.5, "drift_rate": 0.125},
            ],
        }
    )

    assert config.spectral_lines == [
        SpectralLine(frequency=60.0, amplitude=1.5e-23),
        SpectralLine(frequency=120.0, amplitude=5.0e-24, phase=0.5, drift_rate=0.125),
    ]


def test_noise_config_accepts_glitches_from_mappings() -> None:
    """NoiseConfig parses configured glitch mappings into dataclasses."""
    config = NoiseConfig.model_validate(
        {
            "detectors": ["H1"],
            "glitches": [
                {
                    "kind": "blip",
                    "rate": 0.5,
                    "width": 0.01,
                    "amplitude_distribution": {
                        "distribution": "lognormal",
                        "mean": 1.0e-23,
                        "std": 1.0e-24,
                    },
                },
                {
                    "kind": "scattered_light",
                    "rate": 0.1,
                    "duration": 0.5,
                    "peak_frequency": 24.0,
                    "arch_exponent": 1.5,
                    "amplitude_distribution": {
                        "distribution": "lognormal",
                        "mean": 5.0e-24,
                        "std": 0.0,
                    },
                },
            ],
        }
    )

    assert config.glitches == [
        BlipGlitch(
            rate=0.5,
            width=0.01,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0e-23, std=1.0e-24),
        ),
        ScatteredLightGlitch(
            rate=0.1,
            duration=0.5,
            peak_frequency=24.0,
            arch_exponent=1.5,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=5.0e-24, std=0.0),
        ),
    ]


def test_noise_config_accepts_gengli_glitch_mapping(tmp_path: Path) -> None:
    """NoiseConfig parses gengli glitch mappings into dataclass instances."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([8.0, 12.0]))
    np.savetxt(psd_file, np.column_stack((np.array([0.0, 128.0]), np.array([0.0, 1.0]))))

    config = NoiseConfig.model_validate(
        {
            "detectors": ["H1"],
            "glitches": [
                {
                    "kind": "gengli_blip",
                    "rate": 0.2,
                    "population_file": str(population_file),
                    "psd_file": str(psd_file),
                    "gengli_detector": "L1",
                    "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
                }
            ],
        }
    )

    assert isinstance(config.glitches[0], GengliBlipGlitch)
    assert config.glitches[0].population_file == population_file
    assert config.glitches[0].psd_file == psd_file


def test_noise_config_accepts_glitch_instance_inputs() -> None:
    """NoiseConfig accepts GlitchModel instances directly."""
    blip = BlipGlitch(
        rate=0.5,
        width=0.01,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0e-23, std=1.0e-24),
    )
    config = NoiseConfig.model_validate({"detectors": ["H1"], "glitches": [blip]})
    assert config.glitches == [blip]


def test_noise_config_accepts_amplitude_distribution_instances_in_glitch_mapping() -> None:
    """Amplitude-distribution dataclass instances are accepted in glitch mappings."""
    config = NoiseConfig.model_validate(
        {
            "detectors": ["H1"],
            "glitches": [
                {
                    "kind": "blip",
                    "rate": 0.5,
                    "width": 0.01,
                    "amplitude_distribution": LogNormalAmplitudeDistribution(mean=1.0, std=0.1),
                }
            ],
        }
    )
    assert isinstance(config.glitches[0], BlipGlitch)


def test_noise_config_requires_amplitude_distribution_in_glitch_mapping() -> None:
    """Glitch mapping must include amplitude_distribution."""
    with pytest.raises(ValidationError, match="require an amplitude_distribution mapping"):
        NoiseConfig.model_validate({"detectors": ["H1"], "glitches": [{"kind": "blip", "rate": 0.5, "width": 0.01}]})


def test_lognormal_amplitude_distribution_samples_nonzero_std() -> None:
    """Nonzero std path samples finite positive amplitudes."""
    distribution = LogNormalAmplitudeDistribution(mean=1.0, std=0.5)
    sample = distribution.sample(np.random.default_rng(1))
    assert sample > 0.0


def test_scattered_light_glitch_rejects_nonpositive_sampling_frequency() -> None:
    """Scattered-light waveform generation validates sampling frequency."""
    glitch = ScatteredLightGlitch(
        rate=0.1,
        duration=0.5,
        peak_frequency=24.0,
        arch_exponent=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )
    with pytest.raises(ValueError, match="sampling_frequency must be greater than zero"):
        glitch.generate_waveform(0.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mean": 0.0}, "mean must be greater than zero"),
        ({"std": -1.0}, "std must be non-negative"),
    ],
)
def test_lognormal_amplitude_distribution_validates_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    """Amplitude distribution validates mean/std limits."""
    with pytest.raises(ValueError, match=message):
        LogNormalAmplitudeDistribution(**kwargs)


def test_glitch_model_base_generate_is_abstract() -> None:
    """Base GlitchModel.generate_waveform raises NotImplementedError."""
    base = GlitchModel(rate=0.1, amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0))
    with pytest.raises(NotImplementedError):
        base.generate_waveform(256.0)


def test_glitch_model_subclass_validators_reject_invalid_ranges() -> None:
    """Blip/scattered-light validators reject invalid field ranges."""
    with pytest.raises(ValueError, match="blip width must be greater than zero"):
        BlipGlitch(
            rate=0.1,
            width=0.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )
    with pytest.raises(ValueError, match="scattered-light duration must be greater than zero"):
        ScatteredLightGlitch(
            rate=0.1,
            duration=0.0,
            peak_frequency=12.0,
            arch_exponent=1.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )
    with pytest.raises(ValueError, match="scattered-light peak_frequency must be greater than zero"):
        ScatteredLightGlitch(
            rate=0.1,
            duration=0.5,
            peak_frequency=0.0,
            arch_exponent=1.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )
    with pytest.raises(ValueError, match="scattered-light arch_exponent must be greater than zero"):
        ScatteredLightGlitch(
            rate=0.1,
            duration=0.5,
            peak_frequency=12.0,
            arch_exponent=0.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )


def test_glitch_model_rejects_negative_rate() -> None:
    """Common glitch validator rejects negative rates."""
    with pytest.raises(ValueError, match="glitch rate must be non-negative"):
        BlipGlitch(
            rate=-0.1,
            width=0.01,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )


def test_blip_glitch_rejects_nonpositive_sampling_frequency() -> None:
    """Blip waveform generation validates sampling frequency."""
    glitch = BlipGlitch(
        rate=0.1,
        width=0.01,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )
    with pytest.raises(ValueError, match="sampling_frequency must be greater than zero"):
        glitch.generate_waveform(0.0)


def test_blip_glitch_handles_zero_variance_carrier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blip generation handles the zero-carrier-std branch safely."""

    class ZeroNormalRng:
        def normal(self, size: int) -> np.ndarray:
            return np.zeros(size, dtype=float)

    glitch = BlipGlitch(
        rate=0.1,
        width=0.01,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )
    waveform = glitch.generate_waveform(1024.0, rng=ZeroNormalRng())  # type: ignore[arg-type]
    assert waveform.shape[0] > 0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("frequency", -1.0, "spectral line frequency must be non-negative"),
        ("amplitude", -1.0, "spectral line amplitude must be non-negative"),
    ],
)
def test_noise_config_rejects_invalid_spectral_line_values(field: str, value: float, message: str) -> None:
    """SpectralLine validation rejects negative frequency and amplitude."""
    with pytest.raises(ValidationError, match=message):
        NoiseConfig.model_validate(
            {
                "spectral_lines": [
                    {
                        "frequency": 32.0,
                        "amplitude": 1.0e-23,
                        field: value,
                    }
                ]
            }
        )


@pytest.mark.parametrize(
    ("glitch_config", "message"),
    [
        (
            [
                {
                    "kind": "unknown",
                    "rate": 0.5,
                    "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
                }
            ],
            "glitch kind must be one of",
        ),
        (
            [
                {
                    "kind": "blip",
                    "rate": 0.5,
                    "width": 0.01,
                    "amplitude_distribution": {"distribution": "gaussian", "mean": 1.0, "std": 0.0},
                }
            ],
            "Only the 'lognormal' amplitude distribution is supported",
        ),
    ],
)
def test_noise_config_rejects_invalid_glitch_mappings(
    glitch_config: list[dict[str, object]],
    message: str,
) -> None:
    """NoiseConfig rejects unsupported glitch kinds and amplitude distributions."""
    with pytest.raises(ValidationError, match=message):
        NoiseConfig.model_validate({"glitches": glitch_config})


def test_noise_config_rejects_invalid_amplitude_distribution_type() -> None:
    """Glitch amplitude_distribution must be a mapping/dataclass."""
    with pytest.raises(ValidationError, match="amplitude_distribution must be a mapping"):
        NoiseConfig.model_validate(
            {
                "glitches": [
                    {
                        "kind": "blip",
                        "rate": 0.5,
                        "width": 0.01,
                        "amplitude_distribution": 1.23,
                    }
                ]
            }
        )


def test_noise_config_rejects_non_mapping_glitch_entries() -> None:
    """Glitch entries must be mappings or dataclass instances."""
    with pytest.raises(ValidationError, match="glitches entries must be mappings"):
        NoiseConfig.model_validate({"glitches": [123]})


def test_noise_config_parse_glitches_accepts_none() -> None:
    """Parsing keeps a null glitches field as None."""
    config = NoiseConfig.model_validate({"glitches": None})
    assert config.glitches is None


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


def test_noise_config_rejects_mixed_single_and_time_varying_psd_inputs() -> None:
    """NoiseConfig rejects simultaneous psd_file and psd_schedule inputs."""
    with pytest.raises(ValidationError, match="psd_file and psd_schedule are mutually exclusive"):
        NoiseConfig(
            detectors=["H1"],
            psd_file=Path("shared_psd.txt"),
            psd_schedule=[(0.0, Path("shared_psd.txt"))],
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
