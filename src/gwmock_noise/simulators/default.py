"""Default noise simulator implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.config import NoiseConfig
from gwmock_noise.output.frame import FrameWriter
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult
from gwmock_noise.simulators.colored import ColoredNoiseSimulator
from gwmock_noise.simulators.correlated import CorrelatedNoiseSimulator, parse_csd_file_map
from gwmock_noise.simulators.glitches import InjectGlitches, _ZeroNoiseSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator
from gwmock_noise.simulators.spectral_lines import AddLines, SpectralLineSimulator


class DefaultNoiseSimulator(BaseNoiseSimulator):
    """Default noise simulator implementation.

    This implementation keeps ``run(config)`` as the public orchestration
    boundary, dispatching to lower-level generators and writing real strain
    artifacts plus descriptive metadata sidecars.
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

    def _configure_simulator(self, config: NoiseConfig) -> NoiseSimulator | None:
        """Build the runtime simulator implied by the validated config."""
        simulator: NoiseSimulator | None = None
        if config.psd_files is not None or config.csd_files is not None:
            simulator = CorrelatedNoiseSimulator(
                psd_files=config.psd_files or {},
                csd_files=parse_csd_file_map(config.csd_files),
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
                low_frequency_cutoff=config.low_frequency_cutoff,
                high_frequency_cutoff=config.high_frequency_cutoff,
            )
        elif config.psd_file is not None or config.psd_schedule is not None:
            simulator = ColoredNoiseSimulator(
                psd_file=config.psd_file,
                psd_schedule=config.psd_schedule,
                detectors=config.detectors,
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                seed=config.seed,
                low_frequency_cutoff=config.low_frequency_cutoff,
                high_frequency_cutoff=config.high_frequency_cutoff,
            )

        if config.spectral_lines is not None:
            if not config.spectral_lines:
                raise ValueError("spectral_lines must contain at least one spectral line.")

            if simulator is None:
                simulator = SpectralLineSimulator(
                    lines=config.spectral_lines,
                    detectors=config.detectors,
                    duration=config.duration,
                    sampling_frequency=config.sampling_frequency,
                    seed=config.seed,
                )
            else:
                simulator = AddLines(simulator, config.spectral_lines)

        if config.glitches is not None:
            if simulator is None:
                simulator = _ZeroNoiseSimulator(
                    detectors=config.detectors,
                    duration=config.duration,
                    sampling_frequency=config.sampling_frequency,
                    seed=config.seed,
                )
            simulator = InjectGlitches(simulator, config.glitches)

        return simulator

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
            channel_prefix=config.output.channel_prefix,
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
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """
        Path(config.output.directory).mkdir(parents=True, exist_ok=True)

        simulator = self._configure_simulator(config)

        if config.output.format == "gwf":
            active_simulator = self if simulator is None else simulator
            output_paths = self._write_frame_outputs(config=config, simulator=active_simulator)
            if simulator is None:
                self._sync_public_state(config=config)
            else:
                self._sync_public_state(config=config, metadata=simulator.metadata)
        else:
            if simulator is None:
                strain_by_detector = self.generate(
                    duration=config.duration,
                    sampling_frequency=config.sampling_frequency,
                    detectors=config.detectors,
                    seed=config.seed,
                )
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
