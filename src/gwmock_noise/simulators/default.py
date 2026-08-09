"""Default noise simulator implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from gwmock_noise.naming import check_artifact_names
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

    def _write_hdf5_outputs(
        self,
        *,
        config: NoiseConfig,
        strain_by_detector: dict[str, np.ndarray],
    ) -> dict[str, Path]:
        """Persist per-detector strain as HDF5, carrying the same grid a frame would.

        Written with ``h5py``, which is a required dependency, rather than through GWpy, which is not:
        GWpy is an extra here (``frame``, ``gwpy``, ``gwosc``), and a *primary* output format must not
        need an optional package to produce. The first version of this did use GWpy and was wrong for
        that reason.

        The attributes are the ones GWpy's own HDF5 writer uses -- ``x0``, ``dx``, ``channel``, ``name``,
        ``unit``, ``xunit`` -- so ``TimeSeries.read`` loads these files unchanged and a reader cannot tell
        which library produced them. That compatibility is asserted in the tests rather than assumed.

        A bare array would lose where the samples sit, leaving the epoch and rate to travel out of band;
        the sibling project has already had a content hash go blind to exactly that.

        Args:
            config: The noise config, providing the grid, the channel and the naming.
            strain_by_detector: The generated strain, one array per detector.

        Returns:
            The path written for each detector.
        """
        output_paths: dict[str, Path] = {}
        for detector, strain in strain_by_detector.items():
            channel = self._channel_for(config=config, detector=detector)
            output_path = Path(config.output.directory) / self._hdf5_name(
                config=config, detector=detector, channel=channel
            )
            with h5py.File(output_path, "w") as handle:
                dataset = handle.create_dataset(channel, data=np.asarray(strain, dtype=float))
                dataset.attrs["x0"] = float(config.output.gps_start)
                dataset.attrs["dx"] = 1.0 / float(config.sampling_frequency)
                dataset.attrs["xunit"] = "s"
                dataset.attrs["channel"] = channel
                dataset.attrs["name"] = channel
                dataset.attrs["unit"] = "strain"
            output_paths[detector] = output_path
        return output_paths

    @staticmethod
    def _channel_for(*, config: NoiseConfig, detector: str) -> str:
        """Return the channel name for a detector, honouring a per-detector override.

        Same rule as the frame writer's: without it the two formats would name the same data
        differently, and a reader would have to know which one it was handed.

        Used by the HDF5 writer and by the metadata sidecar, so the channel a sidecar advertises is by
        construction the one the artifact holds rather than a second implementation of the same rule.
        """
        channels = config.output.channels
        if channels is not None:
            override = channels.get(detector)
            if override is not None:
                return override
        return f"{detector}:{config.output.channel}"

    @staticmethod
    def _hdf5_name(*, config: NoiseConfig, detector: str, channel: str) -> str:
        """Return the file name for one detector's HDF5 artifact.

        Named for the **detector**, not the channel: ``H-H1_1000000000-4.hdf5``. The dataset inside
        carries the channel, so the name does not have to, and one detector writes one file -- which
        makes the name unique by construction.

        Three attempts got here, and the discarded two are worth stating because each looked right.

        Keeping the frame writer's channel-in-the-name shape put a colon in the file name. That is fine
        for GWF, which cannot be written on Windows at all, but this writer runs on that leg of the
        matrix, and on NTFS a colon opens an alternate data stream rather than a file.

        Escaping the colon to an underscore then made the name **not injective**: with
        ``channels={"H1": "H1:A:B", "H2": "H1:A_B"}`` both detectors produced
        ``H-H1_A_B_1000000000-4.hdf5``, so one silently overwrote the other and both returned paths
        pointed at the single surviving file. A reviewer demonstrated it on disk. The default channel
        already contains an underscore, so the name could not be parsed back either.

        Dropping the channel from the *name* removes both of those, at the cost of a name that says less
        than a frame's. It does not make the writer indifferent to what a channel contains: the channel
        is still the HDF5 dataset path, where `/` is a group separator, so a slash there produced a
        nested group and a file GWpy could not read -- silently, since the name was by then clean. That
        is rejected where the config is built, and asserted again in this writer: the boundary is
        bypassable (`model_construct` skips validators, and this repo's tests use it), so a guarantee
        resting on it alone would hold for validator-built configs only.

        Args:
            config: The noise config, providing the epoch, the duration and the prefix.
            detector: The detector this artifact belongs to.
            channel: The channel stored inside the file; not part of the name.

        Returns:
            The file name.
        """
        start_token = FrameWriter._format_time_token(config.output.gps_start)
        duration_token = FrameWriter._format_time_token(config.duration)
        name = f"{detector[0]}-{detector}_{start_token}-{duration_token}.hdf5"
        if config.output.prefix:
            name = f"{config.output.prefix}_{name}"
        return name

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
            if config.output.format in {"gwf", "hdf5"}:
                # The channel, for the formats that carry one. HDF5 artifacts are named for the detector,
                # so unlike a frame their name does not say which channel they hold -- without this a
                # consumer would have to open every file to find out, which is a cost the rename
                # introduced. `npy` has no channel to advertise: it is a bare array.
                file_metadata["channel"] = self._channel_for(config=config, detector=detector)
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
        # First, before anything is created or loaded. The config boundary is bypassable --
        # `model_construct` skips validators, and this repo's own tests use it -- so the rule is
        # re-asserted here, and three earlier placements were each too late. Inside the writing loop, a run
        # with one bad name wrote the good detectors' files and then raised. Inside the hdf5 branch alone,
        # a bypassed detector still reached the numpy and frame writers. Below the `mkdir` and
        # `_configure_simulator` calls, a refused run still created the output directory, and a component
        # that failed to load raised first, so the caller was told about the component and never about the
        # name. All three were reviewers'. It depends on nothing but the config, so nothing needs to happen
        # before it. Every format names its artifact *and* its sidecar after the detector, so that rule is
        # universal; only the formats that carry a channel have a channel to check.
        check_artifact_names(
            detectors=config.detectors,
            channels=(
                {detector: self._channel_for(config=config, detector=detector) for detector in config.detectors}
                if config.output.format in {"gwf", "hdf5"}
                else {}
            ),
            prefix=config.output.prefix,
        )

        Path(config.output.directory).mkdir(parents=True, exist_ok=True)

        simulator = self._configure_simulator(config)

        if config.output.format == "gwf":
            output_paths = self._write_frame_outputs(config=config, simulator=simulator)
            self._sync_public_state(config=config, metadata=simulator.metadata)
        elif config.output.format == "hdf5":
            strain_by_detector = simulator.generate(
                duration=config.duration,
                sampling_frequency=config.sampling_frequency,
                detectors=config.detectors,
                seed=config.seed,
            )
            self._sync_public_state(config=config, metadata=simulator.metadata)
            output_paths = self._write_hdf5_outputs(config=config, strain_by_detector=strain_by_detector)
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
