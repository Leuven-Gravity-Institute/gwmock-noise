"""White-noise component simulator."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators.base import ConfigurableNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig


def draw_white_noise(rng: np.random.Generator, n_samples: int, detectors: list[str]) -> dict[str, np.ndarray]:
    """Draw ``n_samples`` of standard-normal noise per detector from one shared generator.

    The draw is time-major -- all detectors' sample ``k`` before any detector's sample ``k + 1`` -- so
    consecutive calls on one generator concatenate into the same per-detector arrays one longer call
    returns, which is what a stream's continuation needs. Drawing each detector's full array in turn
    held that for a single detector only. A single detector's values are the generator's own sequence
    either way.
    """
    draws = rng.standard_normal((n_samples, len(detectors)))
    return {detector: np.ascontiguousarray(draws[:, index]) for index, detector in enumerate(detectors)}


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
        self._rng: np.random.Generator | None = None

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
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        elif self._rng is None:
            self._rng = np.random.default_rng(self.seed)
        rng = self._rng
        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")
        return draw_white_noise(rng, n_samples, detectors)

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
