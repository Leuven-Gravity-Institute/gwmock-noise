"""Default noise simulator implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.config import NoiseConfig
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult


class DefaultNoiseSimulator(BaseNoiseSimulator):
    """Default noise simulator implementation.

    For the first milestone, this implementation validates the configuration
    and writes metadata to the output directory. Actual noise generation
    (Gaussian, glitches) will be added in subsequent milestones.
    """

    def __init__(
        self,
        *,
        duration: float = 4.0,
        sampling_frequency: float = 4096.0,
        detectors: list[str] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the simulator with protocol-compatible state."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.seed = seed

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the current simulator state."""
        return {
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
        }

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return placeholder per-detector strain arrays."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        return {detector: np.array([], dtype=float) for detector in detectors}

    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """
        self.generate(
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            detectors=config.detectors,
            seed=config.seed,
        )
        out_dir = Path(config.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = config.output.prefix

        output_paths: dict[str, Path] = {}
        for detector in config.detectors:
            meta_path = out_dir / f"{prefix}_{detector}.json"
            file_metadata = self.metadata | {"detector": detector}
            meta_path.write_text(json.dumps(file_metadata, indent=2))
            output_paths[detector] = meta_path

        return SimulationResult(output_paths=output_paths, config=config)
