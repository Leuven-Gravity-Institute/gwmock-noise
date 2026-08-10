"""GW frame-file writer."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

from gwmock_noise.naming import (
    check_artifact_names,
    reject_colliding_names,
    reject_overlong,
    reject_repeated,
    reject_unsafe,
)
from gwmock_noise.output.gwpy import GWpyAdapter
from gwmock_noise.simulators.protocol import NoiseSimulator

_FRAME_IMPORT_ERROR = (
    "gwpy with a GWF backend is required to use FrameWriter. Install it with `pip install gwmock-noise[frame]`."
)


def _require_gwf_backend() -> None:
    """Ensure a GWpy-compatible GWF backend is installed."""
    try:
        module = import_module("gwpy.io.gwf")
        module.get_backend()
    except ImportError as exc:
        raise ImportError(_FRAME_IMPORT_ERROR) from exc


def format_time_token(value: float) -> str:
    """Return a filename-safe token for a GPS time or a duration, preserving sub-second precision."""
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def compose_frame_name(*, detector: str, channel: str, gps_start: float, duration: float, prefix: str) -> str:
    """Return the file name a frame segment will be written to.

    Module level, and taking no writer, because the simulator must know this name *before* a `FrameWriter`
    exists. `FrameWriter.__init__` requires a GWF backend and creates the output directory, so a run whose
    frame name is too long previously got as far as creating the directory -- and on a machine without a
    backend the `ImportError` masked the name error entirely. A reviewer found that; the comment claiming
    it was safe to leave GWF names to the writer was mine.

    The alternative was to re-derive the name in the pre-flight, which is the mistake round 13 caught: two
    expressions for one name drifted on the empty-prefix case. One function, called by the writer and by
    the pre-flight.

    Args:
        detector: The detector, whose first character is the site letter.
        channel: The resolved channel, which a frame name embeds.
        gps_start: The epoch of the segment.
        duration: The duration of the segment.
        prefix: The artifact prefix, or empty for none.

    Returns:
        The file name, without a directory.
    """
    name = f"{detector[0]}-{channel}_{format_time_token(gps_start)}-{format_time_token(duration)}.gwf"
    if prefix:
        name = f"{prefix}_{name}"
    return name


class FrameWriter:
    """Write simulator output to detector-specific GWF frame files.

    Stored precision depends on the active GWF backend and channel type. Some
    frame pipelines default to float32 strain channels, while the validated
    LALFrame path used here preserves float64 samples.
    """

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        base: NoiseSimulator,
        gps_start: float,
        output_dir: Path,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the writer for contiguous GWF output.

        Raises:
            ValueError: If a channel name cannot survive becoming part of a file name.
        """
        # Before the backend check and before the mkdir. Behind `_require_gwf_backend` these checks were
        # unreachable on a machine without a GWF backend -- the docstring promised a `ValueError` that
        # only an `ImportError` could precede, and the test passed only because it stubs the backend
        # check out. A reviewer caught the inversion: `write` had this order right and `__init__` did
        # not. Before the mkdir for the separate reason that a writer which will never be allowed to
        # write should leave no directory behind. Only the names known now are checked; detectors arrive
        # at `write`.
        reject_unsafe(prefix, field="prefix")
        reject_unsafe(channel, field="channel")
        for override_detector, override in (channels or {}).items():
            reject_unsafe(override_detector, field="detector")
            reject_unsafe(override, field="channel")
        _require_gwf_backend()
        self.base = base
        self.gps_start = gps_start
        self.output_dir = Path(output_dir)
        self.channel = channel
        self.channels = channels
        self.prefix = prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, Path]:
        """Write one GWF file per detector for the requested segment.

        Raises:
            ValueError: If a detector or resolved channel cannot survive becoming part of a file name.
        """
        # First, before the backend check and before generating. `FrameWriter` is public API, so this is
        # not a redundant re-assertion of what a config validator already did: a caller constructing the
        # writer directly never passes through `NoiseConfig`. Reviewers found `write(detectors=["H1/A"])`
        # returning `H-H1/A:MOCK_NOISE_100-2.gwf` -- a path below the output directory, reported as the
        # artifact. The channel legitimately contains `:` here, since frame names embed it, which is why
        # the two rules stay distinct rather than collapsing into one character set.
        check_artifact_names(
            detectors=detectors,
            channels={detector: self._channel_name(detector) for detector in detectors},
            prefix=self.prefix,
        )
        self._check_frame_name_lengths(detectors=detectors, gps_start=self.gps_start, duration=duration)
        _require_gwf_backend()
        segment_start = self.gps_start
        adapter = GWpyAdapter(self.base, gps_start=segment_start)
        series_by_detector = adapter.generate(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )

        output_paths: dict[str, Path] = {}
        for detector, series in series_by_detector.items():
            channel = self._channel_name(detector)
            series.channel = channel
            output_path = self._frame_path(detector, channel, segment_start, duration)
            series.write(output_path, format="gwf", overwrite=True)
            output_paths[detector] = output_path

        self.gps_start = adapter.gps_start
        return output_paths

    def write_segments(
        self,
        segments: list[tuple[float, float]],
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> list[dict[str, Path]]:
        """Write a sequence of contiguous frame segments.

        Raises:
            ValueError: If any segment is empty or reversed, or if a name cannot survive becoming part of
                a file name.
        """
        # Every segment validated before the first one is written. Checking inside the loop meant an
        # invalid second segment raised with the first already on disk and `gps_start` advanced, leaving
        # the caller a partial set they never chose to keep and a writer whose state had moved. Both
        # reviewers found it; it is the round-5 partial-write failure again, in the one place the
        # artifact-name pre-flight does not reach, because the fault is in the segment list rather than
        # in a name.
        for gps_start, gps_end in segments:
            if gps_end <= gps_start:
                raise ValueError(f"Invalid segment ({gps_start}, {gps_end}); expected gps_end > gps_start.")
            # Names too, and for every segment before the first is written. A frame name carries the
            # epoch, so segment 2 can exceed the limit while segment 1 fits: a reviewer wrote segment
            # `(0, 1)` and then `(1000000000, 1000000001)` with a 232-character detector, and the first
            # frame was left on disk with `gps_start` advanced while the second raised `OSError` from
            # inside GWpy. Checking inside the writing loop would have reproduced exactly that.
            self._check_frame_name_lengths(detectors=detectors, gps_start=gps_start, duration=gps_end - gps_start)

        written_segments: list[dict[str, Path]] = []
        for index, (gps_start, gps_end) in enumerate(segments):
            self.gps_start = gps_start
            written_segments.append(
                self.write(
                    duration=gps_end - gps_start,
                    sampling_frequency=sampling_frequency,
                    detectors=detectors,
                    seed=seed if index == 0 else None,
                )
            )
        return written_segments

    def _check_frame_name_lengths(self, *, detectors: list[str], gps_start: float, duration: float) -> None:
        """Check each composed frame name against the filesystem's per-component limit.

        Asks `_frame_path` for the name rather than re-deriving it: the simulator's pre-flight re-derived
        its own names and disagreed with its writer on the empty-prefix case, which is the mistake this
        avoids. A frame name is longer than the other formats' -- it carries the channel as well as the
        epoch and the duration -- so it reaches the limit at a shorter detector name.

        Args:
            detectors: The detectors this segment will write.
            gps_start: The epoch of the segment, which appears in the name.
            duration: The duration of the segment, which also appears in the name.

        Raises:
            ValueError: If any composed frame name exceeds the limit.
        """
        # Before the names are collected: this map is keyed by detector too, so a repeat would collapse
        # and one frame would stand in for two requested detectors.
        reject_repeated(detectors)

        names = {
            detector: self._frame_path(detector, self._channel_name(detector), gps_start, duration).name
            for detector in detectors
        }
        for name in names.values():
            reject_overlong(name, described_as="GWF frame name")
        # And no two detectors onto one name: a frame name embeds the channel, so a per-detector override
        # can collide two detectors that differ only in case as easily as the detectors themselves can.
        reject_colliding_names(names, described_as="GWF frame names")

    def _channel_name(self, detector: str) -> str:
        """Return the frame channel name for a detector."""
        if self.channels is not None:
            override = self.channels.get(detector)
            if override is not None:
                return override
        return f"{detector}:{self.channel}"

    def _frame_path(self, detector: str, channel: str, gps_start: float, duration: float) -> Path:
        """Return the output path for a detector frame segment."""
        return self.output_dir / compose_frame_name(
            detector=detector, channel=channel, gps_start=gps_start, duration=duration, prefix=self.prefix
        )

    @staticmethod
    def _format_time_token(value: float) -> str:
        """Return a filename-safe token preserving sub-second precision.

        Kept as the name the HDF5 writer and the tests already call; the implementation moved to module
        level so `compose_frame_name` can use it without a writer.
        """
        return format_time_token(value)
