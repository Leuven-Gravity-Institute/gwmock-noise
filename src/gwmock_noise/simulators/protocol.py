"""Structural protocol for noise simulators."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class NoiseSimulator(Protocol):
    """Structural interface for downstream noise simulators."""

    duration: float
    sampling_frequency: float
    detectors: list[str]
    seed: int | None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector strain arrays."""

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield per-detector strain chunks lazily."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
