"""Composition helpers for additive multi-component simulations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.simulators.protocol import NoiseSimulator


def _stable_component_seed(base_seed: int, *, index: int, simulator_name: str) -> int:
    """Derive a stable per-component seed from the run seed and component index."""
    payload = f"{base_seed}:{index}:{simulator_name}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


class CompositeNoiseSimulator:
    """Additively combine multiple component simulators."""

    def __init__(
        self,
        components: list[tuple[str, NoiseSimulator]],
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        seed: int | None,
    ) -> None:
        """Initialize the composed simulator."""
        if not components:
            raise ValueError("CompositeNoiseSimulator requires at least one component.")
        self._components = list(components)
        self.detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate all component realizations and add them together."""
        runtime_detectors = list(detectors)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        effective_seed = self.seed if seed is None else seed
        self.seed = effective_seed

        combined: dict[str, np.ndarray] | None = None
        for index, (simulator_name, simulator) in enumerate(self._components):
            component_seed = None
            if effective_seed is not None:
                component_seed = _stable_component_seed(effective_seed, index=index, simulator_name=simulator_name)
            component_result = simulator.generate(duration, sampling_frequency, runtime_detectors, seed=component_seed)

            if combined is None:
                combined = {
                    detector: np.asarray(strain, dtype=float).copy() for detector, strain in component_result.items()
                }
                continue

            for detector in runtime_detectors:
                combined[detector] += np.asarray(component_result[detector], dtype=float)

        if combined is None:
            raise RuntimeError("Composite simulator produced no component outputs.")
        return combined

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield additive component chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the composed run."""
        return {
            "implementation": "composed",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "components": [
                {"simulator": simulator_name, "metadata": simulator.metadata}
                for simulator_name, simulator in self._components
            ],
        }
