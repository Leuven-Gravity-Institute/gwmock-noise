"""Transient glitch injection wrappers."""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.glitches.models import GlitchModel
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
    """Wrap a base simulator and inject transient glitches additively.

    Each glitch model runs an independent Poisson process per detector, so
    every detector receives its own event times and waveform realizations.
    ``GlitchModel.rate`` is therefore the event rate seen by each individual
    detector.

    Every ``(model, detector)`` pair draws from its own random generator,
    derived from the top-level seed and the detector name via an independent
    ``SeedSequence`` stream. A detector's realization is therefore reproducible
    and independent of which other detectors are present or the order in which
    they are requested.
    """

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
        self._seed_sequence: np.random.SeedSequence | None = None
        self._rngs: dict[tuple[int, str], np.random.Generator] = {}
        self._next_event_times: dict[tuple[int, str], float] = {}
        self._event_counts: dict[tuple[int, str], int] = {}

    def _initialize_process(self, seed: int | None) -> None:
        """Reset the per-model, per-detector Poisson-process state."""
        self.seed = seed
        self._elapsed_time = 0.0
        self._seed_sequence = np.random.SeedSequence(seed)
        self._rngs = {}
        self._next_event_times = {}
        self._event_counts = {}

    def _rng_for(self, model_index: int, detector: str) -> np.random.Generator:
        """Return the dedicated generator for one (model, detector) pair.

        The stream is spawned from the top-level seed and a stable hash of the
        detector name, so it does not depend on the presence or ordering of any
        other detector.
        """
        key = (model_index, detector)
        rng = self._rngs.get(key)
        if rng is None:
            if self._seed_sequence is None:
                raise RuntimeError("glitch RNG was not initialized.")
            detector_key = zlib.crc32(detector.encode("utf-8"))
            child = np.random.SeedSequence(
                entropy=self._seed_sequence.entropy,
                spawn_key=(model_index, detector_key),
            )
            rng = np.random.default_rng(child)
            self._rngs[key] = rng
        return rng

    @staticmethod
    def _draw_interarrival(rate: float, rng: np.random.Generator) -> float:
        """Draw the next waiting time for one glitch process."""
        if rate == 0.0:
            return float(np.inf)
        return float(rng.exponential(1.0 / rate))

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
        if seed is not None or self._seed_sequence is None:
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
        if self._seed_sequence is None:
            raise RuntimeError("glitch RNG was not initialized.")

        for index, model in enumerate(self.glitch_models):
            for detector in runtime_detectors:
                key = (index, detector)
                rng = self._rng_for(index, detector)
                if key not in self._next_event_times:
                    # A pair first seen mid-stream (e.g. a detector added between
                    # chunks) starts its Poisson clock at the current segment start.
                    self._next_event_times[key] = segment_start + self._draw_interarrival(model.rate, rng)
                event_time = self._next_event_times[key]
                while event_time < segment_end:
                    sample_index = int((event_time - segment_start) * sampling_frequency)
                    waveform = model.generate_waveform(sampling_frequency, rng=rng)
                    stop_index = min(n_samples, sample_index + waveform.size)
                    if stop_index > sample_index:
                        combined[detector][sample_index:stop_index] += waveform[: stop_index - sample_index]
                        self._event_counts[key] = self._event_counts.get(key, 0) + 1
                    event_time += self._draw_interarrival(model.rate, rng)
                self._next_event_times[key] = event_time

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
        counts = []
        for index, model in enumerate(self.glitch_models):
            count_by_detector = {
                detector: int(self._event_counts.get((model_index, detector), 0))
                for model_index, detector in sorted(self._next_event_times)
                if model_index == index
            }
            counts.append(
                {
                    "kind": model.kind,
                    "count": sum(count_by_detector.values()),
                    "count_by_detector": count_by_detector,
                }
            )
        return base_metadata | {
            "implementation": "inject_glitches",
            "base_implementation": base_metadata.get("implementation"),
            "glitches": {
                "models": [model.serialize() for model in self.glitch_models],
                "elapsed_time_seconds": self._elapsed_time,
                "counts": counts,
            },
        }
