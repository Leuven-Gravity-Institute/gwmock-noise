"""Runtime glitch-model definitions."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

TWO_PI = 2.0 * np.pi


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
        "gengli_blip": importlib.import_module("gwmock_noise.glitches.gengli").GengliBlipGlitch,
    }


def _parse_amplitude_distribution(value: Any) -> LogNormalAmplitudeDistribution:
    """Normalize glitch amplitude-distribution inputs."""
    if isinstance(value, LogNormalAmplitudeDistribution):
        return value
    if not isinstance(value, dict):
        raise ValueError("amplitude_distribution must be a mapping or LogNormalAmplitudeDistribution.")
    return LogNormalAmplitudeDistribution(**value)


def normalize_glitch_models(value: Any) -> list[GlitchModel]:
    """Normalize heterogeneous glitch-config inputs."""
    if not isinstance(value, list):
        raise ValueError("glitch component options must provide a list of models.")

    kinds = supported_glitch_kinds()
    normalized: list[GlitchModel] = []
    for entry in value:
        if isinstance(entry, GlitchModel):
            normalized.append(entry)
            continue
        if not isinstance(entry, dict):
            raise ValueError("glitch model entries must be mappings or GlitchModel instances.")

        kind = entry.get("kind")
        if kind not in kinds:
            raise ValueError(f"glitch kind must be one of {', '.join(sorted(kinds))}.")

        parsed = dict(entry)
        if "amplitude_distribution" not in parsed:
            raise ValueError("glitch configs require an amplitude_distribution mapping.")
        parsed["amplitude_distribution"] = _parse_amplitude_distribution(parsed["amplitude_distribution"])
        parsed.pop("kind", None)
        normalized.append(kinds[kind](**parsed))
    return normalized
