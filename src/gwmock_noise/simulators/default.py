"""Default noise simulator implementation."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from gwmock_noise.naming import (
    check_artifact_names,
    reject_colliding_names,
    reject_overlong,
    reject_repeated,
)
from gwmock_noise.output.frame import FrameWriter, compose_frame_name
from gwmock_noise.simulators.base import BaseNoiseSimulator, SimulationResult
from gwmock_noise.simulators.composite import CompositeNoiseSimulator
from gwmock_noise.simulators.glitches import _ZeroNoiseSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator
from gwmock_noise.simulators.registry import build_component_simulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseConfig


#: How each format's artifact is named in an error message. Spelled out rather than interpolating the
#: format token, so the message reads as a person would write it -- and so it matches the wording
#: `FrameWriter` already uses for the same name.
_ARTIFACT_DESCRIPTIONS = {
    "hdf5": "HDF5 artifact name",
    "npy": "NumPy artifact name",
    "gwf": "GWF frame name",
}


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

    @staticmethod
    def _numpy_name(*, config: NoiseConfig, detector: str) -> str:
        """Return the file name for one detector's NumPy artifact.

        Here rather than inline in the writer so the length pre-flight can ask for the name the writer
        will actually use. The pre-flight originally re-derived it and got the empty-prefix case wrong --
        it modelled `f"{detector}.npy"` while the writer wrote `f"_{detector}.npy"` -- so a name one byte
        under the limit passed the check and failed the write, leaving the artifact behind and raising on
        the sidecar. A reviewer found it. Two expressions for one name is the defect; one function is the
        fix.
        """
        return f"{config.output.prefix}_{detector}.npy"

    @staticmethod
    def _sidecar_name(*, config: NoiseConfig, detector: str) -> str:
        """Return the file name for one detector's JSON metadata sidecar. Written by every format."""
        return f"{config.output.prefix}_{detector}.json"

    def _write_numpy_outputs(
        self,
        *,
        config: NoiseConfig,
        strain_by_detector: dict[str, np.ndarray],
    ) -> dict[str, Path]:
        """Persist per-detector strain arrays as NumPy artifacts."""
        output_paths: dict[str, Path] = {}
        for detector, strain in strain_by_detector.items():
            output_path = Path(config.output.directory) / self._numpy_name(config=config, detector=detector)
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
        carries the channel, so the name does not have to, and one detector writes one file.

        That is *not* the same as unique on disk, which this docstring used to claim: the name is
        injective in the detector as a Python string, while APFS and NTFS compare file names without
        regard to case, so ``["H1", "h1"]`` returned two paths and wrote one file. A reviewer caught the
        claim. Collisions are now rejected before anything is written.

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
            metadata_path = Path(config.output.directory) / self._sidecar_name(config=config, detector=detector)
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

    def _check_artifact_lengths(self, config: NoiseConfig) -> None:
        """Check the composed names, which is where the filesystem's limit actually applies.

        Separate from `check_artifact_names` because it needs the names as *composed*, and composition is
        per format: only the writer knows that HDF5 adds an epoch and a duration while `npy` adds
        neither. Every format also writes the JSON sidecar, so that name is checked whatever the format.

        `gwf` was absent here on purpose until round 16, on the grounds that `FrameWriter` composes and
        checks its own names. It does, but only once it exists, and constructing it needs a backend and
        creates the output directory -- so the name error arrived after the side effect, or not at all.
        It is composed here now, from the writer's own `compose_frame_name`.

        Args:
            config: The configuration whose artifacts are about to be written.

        Raises:
            ValueError: If a composed name exceeds the filesystem's per-component limit, or if two
                detectors compose one name -- as artifacts, or as sidecars.
        """
        # Before the names are collected, not after. Each map below is keyed by detector, so an exact
        # repeat collapses into one entry and the collision check then sees nothing wrong -- which is how
        # `model_construct(detectors=["H1", "H1"])` reported one artifact for two detectors and wrote one
        # file. Both reviewers found it: the config validator alone was never going to be enough, on the
        # very path this pre-flight exists to cover.
        reject_repeated(config.detectors)

        sidecars = {detector: self._sidecar_name(config=config, detector=detector) for detector in config.detectors}
        artifacts: dict[str, str] = {}
        if config.output.format == "hdf5":
            artifacts = {
                detector: self._hdf5_name(
                    config=config,
                    detector=detector,
                    channel=self._channel_for(config=config, detector=detector),
                )
                for detector in config.detectors
            }
        elif config.output.format == "npy":
            artifacts = {detector: self._numpy_name(config=config, detector=detector) for detector in config.detectors}
        elif config.output.format == "gwf":
            # Composed here as well as in the writer, from the writer's own function. Leaving GWF names to
            # `FrameWriter` was mine and was wrong: its `__init__` requires a backend and creates the
            # output directory, so an over-long frame name got as far as creating the directory, and on a
            # machine without a backend the `ImportError` masked the name error entirely. A reviewer found
            # it. Re-deriving the name here instead would be round 13's defect again, so this calls
            # `compose_frame_name`, which the writer also calls.
            artifacts = {
                detector: compose_frame_name(
                    detector=detector,
                    channel=self._channel_for(config=config, detector=detector),
                    gps_start=config.output.gps_start,
                    duration=config.duration,
                    prefix=config.output.prefix,
                )
                for detector in config.detectors
            }

        for name in sidecars.values():
            reject_overlong(name, described_as="metadata sidecar name")
        for name in artifacts.values():
            reject_overlong(
                name,
                described_as=_ARTIFACT_DESCRIPTIONS[config.output.format],
            )

        # Two detectors may compose one name. `_hdf5_name` claimed uniqueness "by construction" because
        # one detector writes one file; the name is injective in the detector as a string, which is not
        # the same as unique on disk. `["H1", "h1"]` returned two paths and wrote one file on APFS.
        #
        if artifacts:
            reject_colliding_names(artifacts, described_as=f"{config.output.format} artifact names")

        # The sidecars too, and the argument for leaving them out was wrong. It ran: a sidecar name is
        # `{prefix}_{detector}.json`, so two detectors can only collide there if they collide in the
        # detector, which collides their artifact names as well -- making a second call no test could
        # distinguish. That holds for `npy` and `hdf5`, whose names are composed from the detector. It
        # does not hold for `gwf`, whose name embeds the resolved *channel*: with
        # `channels={"H1": "X1:A", "h1": "Y1:B"}` the frame names differ, the artifact check passes, and
        # the sidecars still fold onto one file. Measured before this line existed -- two frames written,
        # one `noise_H1.json` left holding `h1`'s metadata, no error raised. Codex found it in round 17.
        #
        # After the artifact check, not before it, so that the formats where both collide go on blaming
        # the artifact name -- the one the caller chose.
        reject_colliding_names(sidecars, described_as="metadata sidecar names")

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

        self._check_artifact_lengths(config)

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
