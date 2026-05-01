"""Tests for the optional GWF frame writer."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.output import FrameWriter


class FixedNoiseSimulator:
    """Minimal simulator that returns deterministic detector arrays."""

    def __init__(self) -> None:
        """Set protocol-compatible state."""
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.detectors: list[str] = []
        self.seed: int | None = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return a fixed ramp for each detector."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        sample_count = int(duration * sampling_frequency)
        return {
            detector: np.linspace(index, index + sample_count - 1, sample_count, dtype=float)
            for index, detector in enumerate(detectors)
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose placeholder metadata."""
        return {"implementation": "fixed"}


def _require_frame_backend() -> Any:
    """Skip unless a GWpy-compatible GWF backend is available."""
    pytest.importorskip("gwpy")
    gwf = import_module("gwpy.io.gwf")
    try:
        gwf.get_backend()
    except ImportError as exc:
        pytest.skip(str(exc))
    return import_module("gwpy.timeseries")


def test_frame_writer_is_importable_from_top_level_package() -> None:
    """FrameWriter is re-exported lazily from the top-level package."""
    assert gwmock_noise.FrameWriter is FrameWriter


def test_frame_writer_raises_clear_error_when_backend_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating the writer without a GWF backend raises a helpful error."""
    frame_output = import_module("gwmock_noise.output.frame")
    original_import_module = frame_output.import_module

    def fake_import_module(name: str):
        if name == "gwpy.io.gwf":
            raise ImportError("No module named 'gwpy.io.gwf'")
        return original_import_module(name)

    monkeypatch.setattr(frame_output, "import_module", fake_import_module)
    with pytest.raises(ImportError, match=r"pip install gwmock-noise\[frame\]"):
        FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=Path("."))


def test_frame_writer_round_trips_gwf_output(tmp_path: Path) -> None:
    """Written frame files are readable and preserve the data."""
    timeseries = _require_frame_backend()
    writer = FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=tmp_path)

    output_paths = writer.write(duration=2.0, sampling_frequency=4.0, detectors=["H1", "L1"])

    assert output_paths["H1"].name == "H-H1:MOCK_NOISE_100-2.gwf"
    assert output_paths["L1"].name == "L-L1:MOCK_NOISE_100-2.gwf"

    recovered = timeseries.TimeSeries.read(output_paths["H1"], "H1:MOCK_NOISE", start=100, end=102)
    assert np.allclose(recovered.value, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))


def test_frame_writer_writes_multiple_segments(tmp_path: Path) -> None:
    """write_segments writes each requested interval with contiguous filenames."""
    _require_frame_backend()
    writer = FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=tmp_path)

    written = writer.write_segments(
        segments=[(100.0, 102.0), (102.0, 103.0)],
        sampling_frequency=4.0,
        detectors=["H1"],
        seed=7,
    )

    assert [segment["H1"].name for segment in written] == [
        "H-H1:MOCK_NOISE_100-2.gwf",
        "H-H1:MOCK_NOISE_102-1.gwf",
    ]
    assert writer.gps_start == pytest.approx(103.0)
