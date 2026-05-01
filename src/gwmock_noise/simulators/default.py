"""Default noise simulator implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.config import NoiseConfig
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult
from gwmock_noise.simulators.colored import ColoredNoiseSimulator
from gwmock_noise.simulators.correlated import CorrelatedNoiseSimulator, parse_csd_file_map


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
        self._active_metadata: dict[str, Any] | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the current simulator state."""
        base_metadata = {
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
        }
        return base_metadata if self._active_metadata is None else base_metadata | self._active_metadata

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
        self._active_metadata = None
        return {detector: np.array([], dtype=float) for detector in detectors}

    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """
        # Note: strain data is generated for metadata capture but not persisted.
        # Future milestones will add strain data output.
        if config.psd_files is not None or config.csd_files is not None:
            correlated_simulator = CorrelatedNoiseSimulator(
                psd_files=config.psd_files or {},
                csd_files=parse_csd_file_map(config.csd_files),
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
                low_frequency_cutoff=config.low_frequency_cutoff,
                high_frequency_cutoff=config.high_frequency_cutoff,
            )
            correlated_simulator.generate(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=config.seed,
            )
            self.duration = config.duration
            self.sampling_frequency = config.sampling_frequency
            self.detectors = list(config.detectors)
            self.seed = config.seed
            self._active_metadata = correlated_simulator.metadata
        elif config.psd_file is None and config.psd_schedule is None:
            self.generate(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=config.seed,
            )
        else:
            colored_simulator = ColoredNoiseSimulator(
                psd_file=config.psd_file,
                psd_schedule=config.psd_schedule,
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
                low_frequency_cutoff=config.low_frequency_cutoff,
                high_frequency_cutoff=config.high_frequency_cutoff,
            )
            colored_simulator.generate(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=config.seed,
            )
            self.duration = config.duration
            self.sampling_frequency = config.sampling_frequency
            self.detectors = list(config.detectors)
            self.seed = config.seed
            self._active_metadata = colored_simulator.metadata

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
