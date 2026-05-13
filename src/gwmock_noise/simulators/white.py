"""White-noise component simulator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators.base import ConfigurableNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig


class WhiteNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate Gaussian white noise as a composable component."""

    simulator_name = "white"

    def __init__(
        self,
        *,
        duration: float = 4.0,
        sampling_frequency: float = 4096.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the white-noise component state."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.seed = seed

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> WhiteNoiseSimulator:
        """Construct one white-noise component from generic config."""
        options = dict(component.options)
        return cls(
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            detectors=config.detectors,
            seed=config.seed,
            **options,
        )

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return Gaussian white-noise strain arrays."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        rng = np.random.default_rng(seed)
        n_samples = round(duration * sampling_frequency)
        return {detector: rng.standard_normal(n_samples).astype(float, copy=False) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield white-noise strain chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the white-noise component."""
        return {
            "implementation": "white",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "white_noise": {"distribution": "standard_normal"},
        }
