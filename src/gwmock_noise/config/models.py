"""Pydantic models for noise simulation configuration."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal, Self
from urllib.parse import urlparse

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

DETECTOR_PAIR_SIZE = 2
TWO_PI = 2.0 * np.pi
PSD_PRESET_PACKAGE = "gwmock_noise.data.psd"
PSD_PRESET_SUFFIX = ".txt"
REMOTE_PSD_SCHEMES = {"http", "https"}


def _is_remote_psd_reference(value: str) -> bool:
    """Return whether a PSD reference is an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in REMOTE_PSD_SCHEMES and bool(parsed.netloc)


def _resolve_bundled_psd_preset(value: str) -> Path | None:
    """Resolve a bare preset name to a bundled PSD asset."""
    if any(separator in value for separator in ("/", "\\")) or Path(value).suffix:
        return None

    resource = resources.files(PSD_PRESET_PACKAGE).joinpath(f"{value}{PSD_PRESET_SUFFIX}")
    if not resource.is_file():
        return None

    return Path(str(resource))


@dataclass(slots=True)
class SpectralLine:
    """Configuration for a narrow-band spectral line."""

    frequency: float
    amplitude: float
    phase: float | None = None
    drift_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate scalar line parameters."""
        if self.frequency < 0:
            raise ValueError("spectral line frequency must be non-negative.")
        if self.amplitude < 0:
            raise ValueError("spectral line amplitude must be non-negative.")


@dataclass(slots=True)
class LogNormalAmplitudeDistribution:
    """Log-normal amplitude sampler parameterized by linear mean and std."""

    distribution: str = "lognormal"
    mean: float = 1.0
    std: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configured distribution parameters."""
        if self.distribution != "lognormal":
            raise ValueError("Only the 'lognormal' amplitude distribution is supported.")
        if self.mean <= 0.0:
            raise ValueError("amplitude distribution mean must be greater than zero.")
        if self.std < 0.0:
            raise ValueError("amplitude distribution std must be non-negative.")

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one amplitude sample."""
        if self.std == 0.0:
            return self.mean

        sigma_squared = float(np.log1p((self.std**2) / (self.mean**2)))
        sigma = float(np.sqrt(sigma_squared))
        mu = float(np.log(self.mean) - (0.5 * sigma_squared))
        return float(rng.lognormal(mean=mu, sigma=sigma))


@dataclass(slots=True)
class GlitchModel:
    """Base dataclass for transient glitch generators."""

    rate: float
    amplitude_distribution: LogNormalAmplitudeDistribution
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate common glitch parameters."""
        if self.rate < 0.0:
            raise ValueError("glitch rate must be non-negative.")

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a single glitch waveform."""
        raise NotImplementedError

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return {
            "kind": self.kind,
            "rate": self.rate,
            "amplitude_distribution": {
                "distribution": self.amplitude_distribution.distribution,
                "mean": self.amplitude_distribution.mean,
                "std": self.amplitude_distribution.std,
            },
        }


@dataclass(slots=True)
class BlipGlitch(GlitchModel):
    """Gaussian-windowed broadband burst."""

    width: float = 0.01
    kind: Literal["blip"] = field(init=False, default="blip")

    def __post_init__(self) -> None:
        """Validate blip-specific parameters."""
        GlitchModel.__post_init__(self)
        if self.width <= 0.0:
            raise ValueError("blip width must be greater than zero.")

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a Gaussian-windowed white-noise burst."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        half_span = max(1, int(np.ceil(3.0 * self.width * sampling_frequency)))
        sample_offsets = np.arange(-half_span, half_span + 1, dtype=float)
        sample_times = sample_offsets / sampling_frequency
        sigma = self.width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        envelope = np.exp(-0.5 * np.square(sample_times / sigma))

        carrier = generator.normal(size=envelope.size)
        carrier_std = float(np.std(carrier))
        if carrier_std > 0.0:
            carrier /= carrier_std

        amplitude = self.amplitude_distribution.sample(generator)
        return amplitude * carrier * envelope

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return GlitchModel.serialize(self) | {"width": self.width}


@dataclass(slots=True)
class ScatteredLightGlitch(GlitchModel):
    """Arch-shaped scattered-light transient with a Gaussian envelope."""

    duration: float = 0.5
    peak_frequency: float = 24.0
    arch_exponent: float = 1.0
    phase: float = 0.0
    kind: Literal["scattered_light"] = field(init=False, default="scattered_light")

    def __post_init__(self) -> None:
        """Validate scattered-light parameters."""
        GlitchModel.__post_init__(self)
        if self.duration <= 0.0:
            raise ValueError("scattered-light duration must be greater than zero.")
        if self.peak_frequency <= 0.0:
            raise ValueError("scattered-light peak_frequency must be greater than zero.")
        if self.arch_exponent <= 0.0:
            raise ValueError("scattered-light arch_exponent must be greater than zero.")

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a chirping scattered-light glitch."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        n_samples = max(1, round(self.duration * sampling_frequency))
        times = np.arange(n_samples, dtype=float) / sampling_frequency
        normalized_times = times / max(self.duration, 1.0 / sampling_frequency)
        centered_times = times - (0.5 * self.duration)
        envelope_sigma = self.duration / 6.0
        envelope = np.exp(-0.5 * np.square(centered_times / envelope_sigma))
        instantaneous_frequency = self.peak_frequency * np.power(
            np.abs(np.sin(np.pi * normalized_times)),
            self.arch_exponent,
        )
        phase = self.phase + (TWO_PI * np.cumsum(instantaneous_frequency) / sampling_frequency)
        amplitude = self.amplitude_distribution.sample(generator)
        return amplitude * envelope * np.sin(phase)

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return GlitchModel.serialize(self) | {
            "duration": self.duration,
            "peak_frequency": self.peak_frequency,
            "arch_exponent": self.arch_exponent,
            "phase": self.phase,
        }


def supported_glitch_kinds() -> dict[str, type[GlitchModel]]:
    """Return all supported glitch-model kinds."""
    return {
        "blip": BlipGlitch,
        "scattered_light": ScatteredLightGlitch,
        "gengli_blip": importlib.import_module("gwmock_noise.glitches").GengliBlipGlitch,
    }


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")
    format: Literal["npy", "gwf"] = Field(
        default="npy",
        description="Artifact format written by BaseNoiseSimulator.run().",
    )
    gps_start: float = Field(
        default=0.0,
        description="GPS start time used for timestamped output formats such as GWF.",
    )
    channel_prefix: str = Field(
        default="MOCK",
        description="Channel-name prefix used for GWF frame output.",
    )


class NoiseConfig(BaseModel):
    """Configuration for gravitational wave detector noise simulation.

    This model is designed to be imported and composed into larger configuration
    structures by upstream packages (e.g., gwmock).
    """

    detectors: list[str] = Field(
        default=["H1", "L1"],
        description="List of detector names to simulate.",
        min_length=1,
    )
    duration: float = Field(
        default=4.0,
        gt=0,
        description="Duration of the noise realization in seconds.",
    )
    sampling_frequency: float = Field(
        default=4096.0,
        gt=0,
        description="Sampling frequency in Hz.",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. If None, use system entropy.",
    )
    psd_file: str | Path | None = Field(
        default=None,
        description="Optional PSD reference for FFT-based colored-noise simulation as a local path, HTTP(S) URL, or bundled preset name.",
    )
    psd_schedule: list[tuple[float, Path]] | None = Field(
        default=None,
        description="Optional time-ordered list of (gps_offset_seconds, PSD file) anchors for time-varying colored-noise simulation.",
        min_length=1,
    )
    psd_files: dict[str, Path] | None = Field(
        default=None,
        description="Optional per-detector PSD files for correlated-noise simulation.",
    )
    csd_files: dict[str, Path] | None = Field(
        default=None,
        description="Optional pairwise CSD files keyed as 'DET1-DET2' for correlated-noise simulation.",
    )
    low_frequency_cutoff: float = Field(
        default=2.0,
        ge=0,
        description="Lower frequency cutoff applied during colored-noise generation.",
    )
    high_frequency_cutoff: float | None = Field(
        default=None,
        gt=0,
        description="Upper frequency cutoff applied during colored-noise generation.",
    )
    spectral_lines: list[SpectralLine] | None = Field(
        default=None,
        description="Optional additive spectral lines injected on top of the configured simulator.",
    )
    glitches: list[GlitchModel] | None = Field(
        default=None,
        description="Optional transient glitches injected on top of the configured simulator.",
        min_length=1,
    )

    @staticmethod
    def _parse_amplitude_distribution(value: Any) -> LogNormalAmplitudeDistribution:
        """Normalize glitch amplitude-distribution inputs."""
        if isinstance(value, LogNormalAmplitudeDistribution):
            return value
        if not isinstance(value, dict):
            raise ValueError("amplitude_distribution must be a mapping or LogNormalAmplitudeDistribution.")
        return LogNormalAmplitudeDistribution(**value)

    @classmethod
    def _parse_glitch(cls, value: Any) -> GlitchModel:
        """Normalize heterogeneous glitch-config inputs."""
        if isinstance(value, GlitchModel):
            return value
        if not isinstance(value, dict):
            raise ValueError("glitches entries must be mappings or GlitchModel instances.")

        kinds = supported_glitch_kinds()
        kind = value.get("kind")
        if kind not in kinds:
            raise ValueError(f"glitch kind must be one of {', '.join(sorted(kinds))}.")

        parsed = dict(value)
        if "amplitude_distribution" not in parsed:
            raise ValueError("glitch configs require an amplitude_distribution mapping.")
        parsed["amplitude_distribution"] = cls._parse_amplitude_distribution(parsed["amplitude_distribution"])
        parsed.pop("kind", None)
        glitch_class = kinds[kind]
        return glitch_class(**parsed)

    @field_validator("glitches", mode="before")
    @classmethod
    def parse_glitches(cls, value: Any) -> Any:
        """Parse configured glitch mappings into dataclass instances."""
        if value is None:
            return value
        return [cls._parse_glitch(entry) for entry in value]

    @field_validator("psd_file", mode="before")
    @classmethod
    def parse_psd_file(cls, value: Any) -> str | Path | None:
        """Normalize PSD inputs to local paths, remote URLs, or bundled presets."""
        if value is None or isinstance(value, Path):
            return value
        if not isinstance(value, str):
            raise TypeError("psd_file must be a string, Path, or None.")

        bundled_preset = _resolve_bundled_psd_preset(value)
        if bundled_preset is not None:
            return bundled_preset
        if _is_remote_psd_reference(value):
            return value
        return Path(value)

    def _validate_frequency_bounds(self, *, nyquist: float) -> None:
        """Validate low/high cutoff values against Nyquist."""
        low = self.low_frequency_cutoff
        high = self.high_frequency_cutoff

        if low < 0:
            raise ValueError("low_frequency_cutoff must be >= 0.")
        if low > nyquist:
            raise ValueError(
                f"low_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
            )

        if high is None:
            return

        if high < 0:
            raise ValueError("high_frequency_cutoff must be >= 0.")
        if high > nyquist:
            raise ValueError(
                f"high_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
            )
        if high <= low:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")

    def _validate_psd_inputs(self) -> None:
        """Validate PSD configuration choices."""
        if self.psd_file is not None and self.psd_files is not None:
            raise ValueError("psd_file and psd_files are mutually exclusive.")
        if self.psd_file is not None and self.psd_schedule is not None:
            raise ValueError("psd_file and psd_schedule are mutually exclusive.")
        if self.psd_schedule is not None and self.psd_files is not None:
            raise ValueError("psd_schedule and psd_files are mutually exclusive.")

        if self.psd_files is not None and set(self.psd_files) != set(self.detectors):
            raise ValueError("psd_files keys must exactly match detectors.")
        if self.psd_schedule is None:
            return

        offsets = [offset for offset, _ in self.psd_schedule]
        if offsets != sorted(offsets):
            raise ValueError("psd_schedule entries must be sorted by GPS offset.")
        if len(offsets) != len(set(offsets)):
            raise ValueError("psd_schedule entries must use distinct GPS offsets.")

    def _validate_csd_inputs(self) -> None:
        """Validate pairwise CSD configuration keys."""
        if self.csd_files is None:
            return

        if self.psd_files is None:
            raise ValueError("csd_files requires psd_files to be configured.")

        seen_pairs: set[tuple[str, str]] = set()
        for pair_key in self.csd_files:
            detectors = pair_key.split("-")
            if len(detectors) != DETECTOR_PAIR_SIZE or not all(detectors):
                raise ValueError("csd_files keys must use the 'DET1-DET2' format.")

            detector_a, detector_b = tuple(sorted(detectors))
            if detector_a == detector_b:
                raise ValueError("csd_files keys must reference two distinct detectors.")
            if detector_a not in self.detectors or detector_b not in self.detectors:
                raise ValueError("csd_files keys must reference configured detectors.")

            normalized_pair = (detector_a, detector_b)
            if normalized_pair in seen_pairs:
                raise ValueError("csd_files contains duplicate detector pairs.")
            seen_pairs.add(normalized_pair)

    @model_validator(mode="after")
    def validate_frequency_cutoffs(self) -> Self:
        """Validate cutoff ordering and Nyquist limits."""
        nyquist = self.sampling_frequency / 2
        self._validate_frequency_bounds(nyquist=nyquist)
        self._validate_psd_inputs()
        self._validate_csd_inputs()

        return self

    model_config = {"frozen": False, "extra": "ignore"}
