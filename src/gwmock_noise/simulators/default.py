"""Default noise simulator implementation."""

from __future__ import annotations

import json
from pathlib import Path

from gwmock_noise.config import NoiseConfig
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult


class DefaultNoiseSimulator(BaseNoiseSimulator):
    """Default noise simulator implementation.

    For the first milestone, this implementation validates the configuration
    and writes metadata to the output directory. Actual noise generation
    (Gaussian, glitches) will be added in subsequent milestones.
    """

    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """
        out_dir = Path(config.output.directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = config.output.prefix

        output_paths: dict[str, Path] = {}
        for detector in config.detectors:
            meta_path = out_dir / f"{prefix}_{detector}.json"
            meta_path.write_text(
                json.dumps(
                    {
                        "detector": detector,
                        "duration": config.duration,
                        "sampling_frequency": config.sampling_frequency,
                        "seed": config.seed,
                    },
                    indent=2,
                )
            )
            output_paths[detector] = meta_path

        return SimulationResult(output_paths=output_paths, config=config)
