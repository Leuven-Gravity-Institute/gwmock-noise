"""GW frame-file writer."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

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


class FrameWriter:
    """Write simulator output to detector-specific GWF frame files.

    Stored precision depends on the active GWF backend and channel type. Some
    frame pipelines default to float32 strain channels, while the validated
    LALFrame path used here preserves float64 samples.
    """

    def __init__(  # noqa: PLR0913
        self,
        base: NoiseSimulator,
        gps_start: float,
        output_dir: Path,
        channel: str = "MOCK_NOISE",
        channels: dict[str, str] | None = None,
        prefix: str = "",
    ) -> None:
        """Initialize the writer for contiguous GWF output."""
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
        """Write one GWF file per detector for the requested segment."""
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
        """Write a sequence of contiguous frame segments."""
        written_segments: list[dict[str, Path]] = []
        for index, (gps_start, gps_end) in enumerate(segments):
            if gps_end <= gps_start:
                raise ValueError(f"Invalid segment ({gps_start}, {gps_end}); expected gps_end > gps_start.")
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

    def _channel_name(self, detector: str) -> str:
        """Return the frame channel name for a detector."""
        if self.channels is not None:
            override = self.channels.get(detector)
            if override is not None:
                return override
        return f"{detector}:{self.channel}"

    def _frame_path(self, detector: str, channel: str, gps_start: float, duration: float) -> Path:
        """Return the output path for a detector frame segment."""
        start_token = self._format_time_token(gps_start)
        duration_token = self._format_time_token(duration)
        name = f"{detector[0]}-{channel}_{start_token}-{duration_token}.gwf"
        if self.prefix:
            name = f"{self.prefix}_{name}"
        return self.output_dir / name

    @staticmethod
    def _format_time_token(value: float) -> str:
        """Return a filename-safe token preserving sub-second precision."""
        if float(value).is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".").replace(".", "p")
