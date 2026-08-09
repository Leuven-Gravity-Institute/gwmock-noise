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

    prefixed = FrameWriter(
        FixedNoiseSimulator(),
        gps_start=100.0,
        output_dir=tmp_path,
        prefix="run_a",
    )
    prefixed_paths = prefixed.write(duration=2.0, sampling_frequency=4.0, detectors=["H1", "L1"])
    assert prefixed_paths["H1"].name == "run_a_H-H1:MOCK_NOISE_100-2.gwf"
    assert prefixed_paths["L1"].name == "run_a_L-L1:MOCK_NOISE_100-2.gwf"

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

    prefixed_writer = FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=tmp_path, prefix="seg")
    prefixed_written = prefixed_writer.write_segments(
        segments=[(100.0, 102.0), (102.0, 103.0)],
        sampling_frequency=4.0,
        detectors=["H1"],
        seed=7,
    )
    assert [segment["H1"].name for segment in prefixed_written] == [
        "seg_H-H1:MOCK_NOISE_100-2.gwf",
        "seg_H-H1:MOCK_NOISE_102-1.gwf",
    ]
    assert writer.gps_start == pytest.approx(103.0)


def test_frame_writer_write_and_write_segments_without_real_gwpy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Frame writer logic can be exercised via fake gwpy backend/series objects."""
    frame_output = import_module("gwmock_noise.output.frame")

    class FakeGwfModule:
        @staticmethod
        def get_backend() -> str:
            return "fake"

    class FakeSeries:
        def __init__(self) -> None:
            self.channel = ""
            self.writes: list[tuple[Path, str, bool]] = []

        def write(self, path: Path, *, format: str, overwrite: bool) -> None:  # noqa: A002
            self.writes.append((path, format, overwrite))

    class FakeAdapter:
        def __init__(self, base: FixedNoiseSimulator, gps_start: float) -> None:
            self.base = base
            self.gps_start = gps_start
            self._series = {"H1": FakeSeries()}

        def generate(self, *, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None):
            self.base.generate(duration, sampling_frequency, detectors, seed=seed)
            self.gps_start += duration
            return {detector: self._series[detector] for detector in detectors}

    monkeypatch.setattr(frame_output, "import_module", lambda name: FakeGwfModule())
    monkeypatch.setattr(frame_output, "GWpyAdapter", FakeAdapter)

    writer = FrameWriter(FixedNoiseSimulator(), gps_start=100.25, output_dir=tmp_path, prefix="unit")
    output = writer.write(duration=1.25, sampling_frequency=8.0, detectors=["H1"], seed=9)

    path = output["H1"]
    assert path.name == "unit_H-H1:MOCK_NOISE_100p25-1p25.gwf"
    assert writer.gps_start == pytest.approx(101.5)

    with pytest.raises(ValueError, match="expected gps_end > gps_start"):
        writer.write_segments(segments=[(3.0, 3.0)], sampling_frequency=8.0, detectors=["H1"])

    assert FrameWriter._format_time_token(10.0) == "10"
    assert FrameWriter._format_time_token(10.125) == "10p125"


def test_frame_writer_channel_name_uses_channel_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_channel_name assembles {detector}:{channel} from the channel field."""
    frame_output = import_module("gwmock_noise.output.frame")
    monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
    writer = FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=tmp_path, channel="STRAIN_NOISE")
    assert writer._channel_name("H1") == "H1:STRAIN_NOISE"
    assert writer._channel_name("L1") == "L1:STRAIN_NOISE"


def test_frame_writer_channels_dict_takes_precedence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_channel_name returns verbatim per-detector name from channels dict, ignoring channel."""
    frame_output = import_module("gwmock_noise.output.frame")
    monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
    writer = FrameWriter(
        FixedNoiseSimulator(),
        gps_start=0.0,
        output_dir=tmp_path,
        channel="FALLBACK",
        channels={"H1": "H1:CUSTOM", "L1": "L1:CUSTOM"},
    )
    assert writer._channel_name("H1") == "H1:CUSTOM"
    assert writer._channel_name("L1") == "L1:CUSTOM"


def test_frame_writer_channels_dict_falls_back_for_unmapped_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_channel_name falls back to channel field for detectors absent from channels dict."""
    frame_output = import_module("gwmock_noise.output.frame")
    monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
    writer = FrameWriter(
        FixedNoiseSimulator(),
        gps_start=0.0,
        output_dir=tmp_path,
        channel="FALLBACK",
        channels={"H1": "H1:CUSTOM"},
    )
    assert writer._channel_name("H1") == "H1:CUSTOM"
    assert writer._channel_name("L1") == "L1:FALLBACK"


class TestTheWriterChecksItsOwnInputs:
    """`FrameWriter` is public API, so a caller can reach it without a `NoiseConfig`.

    Both reviewers found this in round 8 of the HDF5 work: the simulator's pre-flight closed the path
    through `run()`, and left the writer itself open. It is not a second copy of the config's rule -- a
    direct caller never passes through the config at all -- but it is the same rule, from `naming`.
    """

    @staticmethod
    def _writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> FrameWriter:
        frame_output = import_module("gwmock_noise.output.frame")
        monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
        return FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=tmp_path, **kwargs)

    def test_a_detector_carrying_path_syntax_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reviewer's exact reproduction: this returned `H-H1/A:MOCK_NOISE_100-2.gwf`."""
        writer = self._writer(tmp_path, monkeypatch)
        (tmp_path / "H-H1").mkdir()

        with pytest.raises(ValueError, match="path syntax"):
            writer.write(duration=2.0, sampling_frequency=4.0, detectors=["H1/A"])

        assert list((tmp_path / "H-H1").iterdir()) == []

    def test_a_channel_replaced_after_construction_is_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Why `write` re-checks the channel when `__init__` already did.

        A reviewer's second reproduction was `FrameWriter(..., channel="MOCK/NOISE")`, which is refused at
        construction now, so that input cannot reach `write` at all. `channel` is a plain public
        attribute, though, so assigning to it afterwards gets there instead, and without the check in
        `write` that produced `H-H1:MOCK/NOISE_100-2.gwf`. Written because the construction-time check
        made the one in `write` look redundant, and it is not.
        """
        writer = self._writer(tmp_path, monkeypatch)
        writer.channel = "MOCK/NOISE"
        (tmp_path / "H-H1:MOCK").mkdir()

        with pytest.raises(ValueError, match="group separator"):
            writer.write(duration=2.0, sampling_frequency=4.0, detectors=["H1"])

        assert list((tmp_path / "H-H1:MOCK").iterdir()) == []

    def test_a_bad_channel_is_refused_at_construction_before_the_directory_is_made(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The channel is known before any detector is, so it is refused before the `mkdir`.

        `__init__` creates `output_dir`. Refusing only at `write` would leave a directory made for a
        writer that was never going to be allowed to write -- the mistake the simulator made and a
        reviewer caught there.
        """
        target = tmp_path / "not-created-yet"

        with pytest.raises(ValueError, match="group separator"):
            self._writer(target, monkeypatch, channel="MOCK/NOISE")

        assert not target.exists()

    def test_a_bad_override_is_refused_even_for_a_detector_never_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Matching the config layer, which rejects override keys rather than ignoring them.

        An override for a detector absent from `detectors` never reaches a file name, so this rejects
        slightly more than the artifacts strictly require. That is deliberate and matches
        `_validate_channel_names`: a key that will silently never apply is a configuration error worth
        reporting, not a name worth permitting.
        """
        with pytest.raises(ValueError, match="path syntax"):
            self._writer(tmp_path, monkeypatch, channels={"H1/A": "H1:STRAIN"})

    def test_a_bad_override_channel_is_refused_at_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override's value, checked before any detector list exists.

        `write` re-derives channels through `_channel_name`, so an override for a detector that *is*
        written is caught there regardless -- which is why a mutation deleting this line survived until
        this test existed. What only `__init__` can catch is an override for a detector this writer is
        never asked to write: its value reaches no file name, so `write` has nothing to check. Reported
        for the same reason the config layer reports it, and the same reason the bad *key* above is
        reported: a name that can only ever silently not apply is a mistake, not a permission.
        """
        with pytest.raises(ValueError, match="group separator"):
            self._writer(tmp_path, monkeypatch, channels={"V1": "V1:STRAIN/NOISE"})

    def test_an_ordinary_writer_is_unaffected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The colon in a frame channel is normal and must survive: frame names embed the channel."""
        writer = self._writer(tmp_path, monkeypatch, channels={"H1": "H1:STRAIN_NOISE"})

        assert writer._channel_name("H1") == "H1:STRAIN_NOISE"
        assert writer._frame_path("H1", "H1:STRAIN_NOISE", 100.0, 2.0).name == "H-H1:STRAIN_NOISE_100-2.gwf"


class TestRoundNineGaps:
    """Three things round 9 found: an unchecked prefix, a masked check, and a partial write."""

    @staticmethod
    def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> Any:
        frame_output = import_module("gwmock_noise.output.frame")

        class FakeSeries:
            def __init__(self) -> None:
                self.channel = ""

            def write(self, path: Path, *, format: str, overwrite: bool) -> None:  # noqa: A002
                path.write_bytes(b"")

        class FakeAdapter:
            def __init__(self, base: FixedNoiseSimulator, gps_start: float) -> None:
                self.base = base
                self.gps_start = gps_start

            def generate(self, *, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None):
                self.gps_start += duration
                return {detector: FakeSeries() for detector in detectors}

        monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
        monkeypatch.setattr(frame_output, "GWpyAdapter", FakeAdapter)
        return frame_output

    def test_a_prefix_carrying_path_syntax_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The prefix is a file-name component, and for nine rounds it was the unchecked one."""
        self._fake_backend(monkeypatch)

        with pytest.raises(ValueError, match="path syntax"):
            FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=tmp_path, prefix="sub/run")

    def test_the_name_check_precedes_the_backend_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Deliberately does NOT stub the backend out, which is what hid this.

        With `_require_gwf_backend` first, a machine without a GWF backend got `ImportError` for a name
        the docstring promised would raise `ValueError` -- the checks were unreachable there, and the
        other tests could not tell because they all stub the backend check. This one forces the missing
        backend and demands the name error anyway.
        """
        frame_output = import_module("gwmock_noise.output.frame")
        original_import_module = frame_output.import_module

        def fake_import_module(name: str):
            if name == "gwpy.io.gwf":
                raise ImportError("No module named 'gwpy.io.gwf'")
            return original_import_module(name)

        monkeypatch.setattr(frame_output, "import_module", fake_import_module)

        with pytest.raises(ValueError, match="group separator"):
            FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=tmp_path, channel="MOCK/NOISE")

    def test_an_invalid_later_segment_writes_nothing_at_all(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The partial write both reviewers reproduced.

        The existing test at the top of this file passes a single invalid segment, which fails before
        anything is written and so cannot see this. Here the first segment is valid: it used to be
        written, and `gps_start` advanced, before the second was rejected.
        """
        self._fake_backend(monkeypatch)
        writer = FrameWriter(FixedNoiseSimulator(), gps_start=100.0, output_dir=tmp_path)

        with pytest.raises(ValueError, match="expected gps_end > gps_start"):
            writer.write_segments(
                segments=[(100.0, 102.0), (102.0, 102.0)],
                sampling_frequency=4.0,
                detectors=["H1"],
            )

        assert list(tmp_path.iterdir()) == []
        assert writer.gps_start == pytest.approx(100.0)
