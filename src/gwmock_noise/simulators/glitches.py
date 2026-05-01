"""Transient glitch injection wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.config.models import GlitchModel
from gwmock_noise.simulators.protocol import NoiseSimulator


class _ZeroNoiseSimulator:
    """Minimal zero-noise base simulator for glitch-only realizations."""

    def __init__(
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        seed: int | None,
    ) -> None:
        """Initialize the zero-valued protocol-compatible state."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed

    def reset(self) -> None:
        """Reset the zero-noise base state."""
        return None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate zero-valued strain arrays."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield zero-valued chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata for the zero-valued base simulator."""
        return {
            "implementation": "zero",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
        }


class InjectGlitches:
    """Wrap a base simulator and inject transient glitches additively."""

    def __init__(self, base: NoiseSimulator, glitch_models: list[GlitchModel]) -> None:
        """Initialize the additive glitch wrapper."""
        if not glitch_models:
            raise ValueError("glitch_models must contain at least one glitch model.")

        self.base = base
        self.glitch_models = list(glitch_models)
        self.duration = base.duration
        self.sampling_frequency = base.sampling_frequency
        self.detectors = list(base.detectors)
        self.seed = base.seed

        self._elapsed_time = 0.0
        self._rng: np.random.Generator | None = None
        self._next_event_times = np.full(len(self.glitch_models), np.inf, dtype=float)
        self._event_counts = np.zeros(len(self.glitch_models), dtype=int)

    def _initialize_process(self, seed: int | None) -> None:
        """Reset the Poisson-process state."""
        self.seed = seed
        self._elapsed_time = 0.0
        self._rng = np.random.default_rng(seed)
        self._event_counts = np.zeros(len(self.glitch_models), dtype=int)
        self._next_event_times = np.array(
            [self._draw_interarrival(model.rate) for model in self.glitch_models],
            dtype=float,
        )

    def _draw_interarrival(self, rate: float) -> float:
        """Draw the next waiting time for one glitch process."""
        if rate == 0.0:
            return float(np.inf)
        if self._rng is None:
            raise RuntimeError("glitch RNG was not initialized.")
        return float(self._rng.exponential(1.0 / rate))

    def reset(self) -> None:
        """Reset the additive wrapper and any resettable base state."""
        self._initialize_process(self.seed)
        if hasattr(self.base, "reset"):
            self.base.reset()

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate base noise and inject the configured glitches."""
        runtime_detectors = list(detectors)
        base_result = self.base.generate(duration, sampling_frequency, runtime_detectors, seed=seed)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        if seed is not None or self._rng is None:
            self._initialize_process(seed if seed is not None else self.seed)

        n_samples = round(duration * sampling_frequency)
        combined: dict[str, np.ndarray] = {}
        for detector in runtime_detectors:
            if detector not in base_result:
                raise KeyError(f"Base simulator did not return detector '{detector}'.")
            base_strain = np.asarray(base_result[detector], dtype=float)
            if base_strain.shape != (n_samples,):
                raise ValueError(
                    "Base simulator output shape must match the requested duration and sampling_frequency."
                )
            combined[detector] = base_strain.copy()

        segment_start = self._elapsed_time
        segment_end = segment_start + duration
        if self._rng is None:
            raise RuntimeError("glitch RNG was not initialized.")

        for index, model in enumerate(self.glitch_models):
            event_time = float(self._next_event_times[index])
            while event_time < segment_end:
                sample_index = int((event_time - segment_start) * sampling_frequency)
                waveform = model.generate_waveform(sampling_frequency, rng=self._rng)
                stop_index = min(n_samples, sample_index + waveform.size)
                if stop_index > sample_index:
                    segment = waveform[: stop_index - sample_index]
                    for detector in runtime_detectors:
                        combined[detector][sample_index:stop_index] += segment
                    self._event_counts[index] += 1
                event_time += self._draw_interarrival(model.rate)
            self._next_event_times[index] = event_time

        self._elapsed_time = segment_end
        return combined

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield glitch-injected chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return additive-wrapper metadata."""
        base_metadata = dict(self.base.metadata)
        return base_metadata | {
            "implementation": "inject_glitches",
            "base_implementation": base_metadata.get("implementation"),
            "glitches": {
                "models": [model.serialize() for model in self.glitch_models],
                "elapsed_time_seconds": self._elapsed_time,
                "counts": [
                    {"kind": model.kind, "count": int(count)}
                    for model, count in zip(self.glitch_models, self._event_counts, strict=True)
                ],
            },
        }
