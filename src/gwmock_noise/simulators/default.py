"""Default noise simulator implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.output.frame import FrameWriter
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult
from gwmock_noise.simulators.composite import CompositeNoiseSimulator
from gwmock_noise.simulators.glitches import _ZeroNoiseSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator
from gwmock_noise.simulators.registry import build_component_simulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseConfig


class DefaultNoiseSimulator(BaseNoiseSimulator):
    """Default noise simulator implementation."""

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
            "implementation": "white",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "white_noise": {"distribution": "standard_normal"},
        }
        return base_metadata if self._active_metadata is None else base_metadata | self._active_metadata

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
        self._active_metadata = None
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

    def _configure_simulator(self, config: NoiseConfig) -> NoiseSimulator:
        """Build the runtime simulator implied by the validated component config."""
        self._active_metadata = None
        if not config.components:
            return _ZeroNoiseSimulator(
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
            )

        built_components = [
            (component.simulator, build_component_simulator(component, config)) for component in config.components
        ]
        if len(built_components) == 1:
            return built_components[0][1]

        return CompositeNoiseSimulator(
            built_components,
            detectors=config.detectors,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
        )

    def _write_numpy_outputs(
        self,
        *,
        config: NoiseConfig,
        strain_by_detector: dict[str, np.ndarray],
    ) -> dict[str, Path]:
        """Persist per-detector strain arrays as NumPy artifacts."""
        output_paths: dict[str, Path] = {}
        for detector, strain in strain_by_detector.items():
            output_path = Path(config.output.directory) / f"{config.output.prefix}_{detector}.npy"
            np.save(output_path, strain)
            output_paths[detector] = output_path
        return output_paths

    def _write_frame_outputs(
        self,
        *,
        config: NoiseConfig,
        simulator: NoiseSimulator,
    ) -> dict[str, Path]:
        """Persist per-detector strain arrays as GWF frame files."""
        writer = FrameWriter(
            simulator,
            gps_start=config.output.gps_start,
            output_dir=Path(config.output.directory),
            channel=config.output.channel,
            channels=config.output.channels,
            prefix=config.output.prefix,
        )
        return writer.write(
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            detectors=config.detectors,
            seed=config.seed,
        )

    def _write_metadata_sidecars(
        self,
        *,
        config: NoiseConfig,
        output_paths: dict[str, Path],
    ) -> None:
        """Write metadata sidecars describing the emitted detector artifacts."""
        for detector, artifact_path in output_paths.items():
            metadata_path = Path(config.output.directory) / f"{config.output.prefix}_{detector}.json"
            file_metadata = self.metadata | {
                "detector": detector,
                "artifact_format": config.output.format,
                "artifact_path": str(artifact_path),
            }
            metadata_path.write_text(json.dumps(file_metadata, indent=2))

    def _sync_public_state(self, *, config: NoiseConfig, metadata: dict[str, Any] | None = None) -> None:
        """Mirror the active runtime state onto the public orchestrator instance."""
        self.duration = config.duration
        self.sampling_frequency = config.sampling_frequency
        self.detectors = list(config.detectors)
        self.seed = config.seed
        self._active_metadata = metadata

    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration."""
        Path(config.output.directory).mkdir(parents=True, exist_ok=True)

        simulator = self._configure_simulator(config)

        if config.output.format == "gwf":
            output_paths = self._write_frame_outputs(config=config, simulator=simulator)
            self._sync_public_state(config=config, metadata=simulator.metadata)
        else:
            strain_by_detector = simulator.generate(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=config.seed,
            )
            self._sync_public_state(config=config, metadata=simulator.metadata)
            output_paths = self._write_numpy_outputs(
                config=config,
                strain_by_detector=strain_by_detector,
            )

        self._write_metadata_sidecars(config=config, output_paths=output_paths)
        return SimulationResult(output_paths=output_paths, config=config)
