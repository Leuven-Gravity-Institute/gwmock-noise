"""Tests for the optional GWF frame writer."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gwmock_noise
import gwmock_noise.simulators.default
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

    assert output_paths["H1"].name == "H-H1_MOCK_NOISE_100-2.gwf"
    assert output_paths["L1"].name == "L-L1_MOCK_NOISE_100-2.gwf"

    prefixed = FrameWriter(
        FixedNoiseSimulator(),
        gps_start=100.0,
        output_dir=tmp_path,
        prefix="run_a",
    )
    prefixed_paths = prefixed.write(duration=2.0, sampling_frequency=4.0, detectors=["H1", "L1"])
    assert prefixed_paths["H1"].name == "run_a_H-H1_MOCK_NOISE_100-2.gwf"
    assert prefixed_paths["L1"].name == "run_a_L-L1_MOCK_NOISE_100-2.gwf"

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
        "H-H1_MOCK_NOISE_100-2.gwf",
        "H-H1_MOCK_NOISE_102-1.gwf",
    ]

    prefixed_writer = FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=tmp_path, prefix="seg")
    prefixed_written = prefixed_writer.write_segments(
        segments=[(100.0, 102.0), (102.0, 103.0)],
        sampling_frequency=4.0,
        detectors=["H1"],
        seed=7,
    )
    assert [segment["H1"].name for segment in prefixed_written] == [
        "seg_H-H1_MOCK_NOISE_100-2.gwf",
        "seg_H-H1_MOCK_NOISE_102-1.gwf",
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
    assert path.name == "unit_H-H1_MOCK_NOISE_100p25-1p25.gwf"
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
        `write` that produced `H-H1_MOCK/NOISE_100-2.gwf`. Written because the construction-time check
        made the one in `write` look redundant, and it is not.

        The decoy directory used to be `H-H1:MOCK`, which is the shape the name had before the channel's
        `IFO:` prefix was dropped -- and creating it raised `NotADirectoryError [WinError 267]` on
        Windows, so this test failed there for a reason that had nothing to do with what it asserts. That
        was the second of the two Windows CI failures.
        """
        writer = self._writer(tmp_path, monkeypatch)
        writer.channel = "MOCK/NOISE"
        (tmp_path / "H-H1_MOCK").mkdir()

        with pytest.raises(ValueError, match="group separator"):
            writer.write(duration=2.0, sampling_frequency=4.0, detectors=["H1"])

        assert list((tmp_path / "H-H1_MOCK").iterdir()) == []

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
        assert writer._frame_path("H1", "H1:STRAIN_NOISE", 100.0, 2.0).name == "H-H1_STRAIN_NOISE_100-2.gwf"


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


def _writer_over_fake_backend(
    directory: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded: dict[str, str] | None = None,
    **kwargs: object,
) -> FrameWriter:
    """Return a `FrameWriter` over a fake GWF backend, writing real (empty) files at the composed paths.

    One helper rather than the three near-copies this file had grown: CodeRabbit flagged the duplication
    on the PR, and two of the copies were already identical, which is how a fake drifts from the real
    interface it stands for without anything failing.

    Args:
        directory: The output directory the writer is given.
        monkeypatch: Used to replace the backend check and the GWpy adapter.
        recorded: If given, filled with the channel each detector's frame was written with -- the channel
            still has to reach the frame, now that it no longer reaches the file name.
        **kwargs: Passed to `FrameWriter`.

    Returns:
        The writer.
    """
    frame_output = import_module("gwmock_noise.output.frame")

    class FakeSeries:
        def __init__(self, detector: str) -> None:
            self.channel = ""
            self._detector = detector

        def write(self, path: Path, *, format: str, overwrite: bool) -> None:  # noqa: A002
            # At write time, not at generate time: `write` assigns `series.channel` in between, and that
            # assignment is what keeps the channel in the frame after it left the file name.
            if recorded is not None:
                recorded[self._detector] = self.channel
            path.write_bytes(b"")

    class FakeAdapter:
        def __init__(self, base: FixedNoiseSimulator, gps_start: float) -> None:
            self.base = base
            self.gps_start = gps_start

        def generate(self, *, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None):
            self.gps_start += duration
            return {detector: FakeSeries(detector) for detector in detectors}

    monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
    monkeypatch.setattr(frame_output, "GWpyAdapter", FakeAdapter)
    return FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=directory, **kwargs)


class TestComposedFrameNameLength:
    """Frames compose their own names, so they need their own length check.

    Left out of the first length fix deliberately rather than guessed at, and flagged as absent in the
    round-13 brief; a reviewer then demonstrated both ways it bites.
    """

    _writer = staticmethod(_writer_over_fake_backend)

    def test_an_overlong_frame_name_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A frame name carries the channel as well as the epoch, so it hits the limit soonest.

        The boundary was measured, not assumed: at epoch `0` with no prefix, 234 characters composes to
        exactly 255 bytes and 235 to 256. This test first used 232 -- the length from the reviewer's
        *two-segment* reproduction, where the second segment's ten-digit epoch is what pushes it over --
        and failed with DID NOT RAISE, because at epoch `0` that name fits.
        """
        writer = self._writer(tmp_path, monkeypatch, prefix="")

        with pytest.raises(ValueError, match="GWF frame name"):
            writer.write(duration=1.0, sampling_frequency=4.0, detectors=["D" * 235])

        assert list(tmp_path.iterdir()) == []

    def test_the_last_frame_name_that_fits_is_still_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """234 characters composes to exactly 255 bytes, so it must go through."""
        writer = self._writer(tmp_path, monkeypatch, prefix="")

        output = writer.write(duration=1.0, sampling_frequency=4.0, detectors=["D" * 234])

        assert len(output["D" * 234].name.encode("utf-8")) == 255

    def test_a_later_segment_that_will_not_fit_stops_the_whole_sequence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reviewer's reproduction: segment 2's epoch is longer than segment 1's.

        `(0, 1)` composes a name that fits and `(1000000000, 1000000001)` does not, so the first frame was
        written and left behind with `gps_start` advanced while the second raised `OSError` from GWpy.
        Every segment's name is now checked before any is written.
        """
        writer = self._writer(tmp_path, monkeypatch, prefix="")

        with pytest.raises(ValueError, match="GWF frame name"):
            writer.write_segments(
                segments=[(0.0, 1.0), (1000000000.0, 1000000001.0)],
                sampling_frequency=4.0,
                detectors=["D" * 232],
            )

        assert list(tmp_path.iterdir()) == []
        assert writer.gps_start == pytest.approx(0.0)

    def test_an_ordinary_frame_name_is_unaffected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The check must not disturb the names the other frame tests assert."""
        writer = self._writer(tmp_path, monkeypatch, prefix="unit")

        output = writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1"])

        assert output["H1"].name == "unit_H-H1_MOCK_NOISE_0-1.gwf"


class TestFrameNameCollisions:
    """Frame names embed the detector *and* the channel, so only the detector can collide them now.

    This class used to say an override could collide two distinct detectors as easily as case could, and
    it could: the name was `{site}-{channel}_{epoch}-{duration}.gwf`, so `channels={"H1": "X1:A", "H2":
    "x1:a"}` put `H1` and `H2` on one file. Dropping the channel's `IFO:` prefix -- forced by Windows
    reserving `:` -- put the detector into the name and closed that case: two detectors that differ at
    all now differ in the name, whatever their channels say.

    The collision surface for frames is therefore now exactly the collision surface for detectors, which
    is what the other two formats have always had. The override test below is kept, inverted: "this can
    no longer happen" is worth asserting where it used to happen.
    """

    _writer = staticmethod(_writer_over_fake_backend)

    def test_detectors_differing_only_in_case_are_refused(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nothing written, rather than one frame standing in for two detectors."""
        writer = self._writer(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="collide"):
            writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1", "h1"])

        assert list(tmp_path.iterdir()) == []

    def test_two_overrides_can_no_longer_collide_distinct_detectors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact pair that used to name one file, asserted to name two.

        `channels={"H1": "X1:A", "H2": "x1:a"}` collided when the name was composed from the channel
        alone. The name now carries the detector, so `H1` and `H2` cannot meet however their channels
        fold. Written as two files on disk, not as two strings: the collision it replaces was only ever
        visible on the filesystem.
        """
        writer = self._writer(tmp_path, monkeypatch, channels={"H1": "X1:A", "H2": "x1:a"})

        output = writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1", "H2"])

        assert len({str(path) for path in output.values()}) == 2
        assert len(list(tmp_path.glob("*.gwf"))) == 2

    def test_the_override_still_reaches_the_frame_content(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dropping `IFO:` from the *name* must not drop the channel from the frame.

        The whole justification for taking the colon out of the file name is that GWF carries the channel
        natively, so nothing is lost. That is only true if the writer still sets it.
        """
        recorded: dict[str, str] = {}
        writer = self._writer(tmp_path, monkeypatch, channels={"H1": "X1:A"}, recorded=recorded)

        writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1"])

        assert recorded == {"H1": "X1:A"}

    def test_distinct_detectors_still_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two ordinary detectors keep producing two frames."""
        writer = self._writer(tmp_path, monkeypatch)

        output = writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1", "L1"])

        assert len({str(path) for path in output.values()}) == 2


class TestARepeatedDetectorReachesTheFrameWriter:
    """`FrameWriter` is public API, so the repeat rule has to hold here without any config."""

    def test_a_repeat_is_refused(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """It returned one path for two detectors and wrote one frame."""
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
        writer = FrameWriter(FixedNoiseSimulator(), gps_start=0.0, output_dir=tmp_path)

        with pytest.raises(ValueError, match="more than once"):
            writer.write(duration=1.0, sampling_frequency=4.0, detectors=["H1", "H1"])

        assert list(tmp_path.iterdir()) == []


class TestGwfNamesAreCheckedBeforeAnythingExists:
    """The GWF path had no pre-flight at all, and `FrameWriter` cannot supply one.

    Round 16. `run()` composed names for `npy` and `hdf5` and left GWF to the writer -- my comment said so
    -- but `FrameWriter.__init__` requires a backend and creates the output directory, so an over-long
    frame name got as far as creating the directory, and on a machine with no GWF backend the `ImportError`
    masked the name error entirely. A reviewer found it.
    """

    def test_an_overlong_gwf_name_is_refused_before_the_directory_is_created(self, tmp_path: Path) -> None:
        """No backend is stubbed here, deliberately: that is the environment where it was masked."""
        target = tmp_path / "not-created-yet"
        config = gwmock_noise.NoiseConfig(
            detectors=["D" * 235],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=[],
            output=gwmock_noise.OutputConfig(format="gwf", prefix="", directory=target, gps_start=0.0),
        )

        with pytest.raises(ValueError, match="GWF frame name"):
            gwmock_noise.simulators.default.DefaultNoiseSimulator().run(config)

        assert not target.exists()

    def test_the_composer_is_the_one_the_writer_uses(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """One expression for the frame name, checked by comparing the two callers' output.

        Re-deriving it in the pre-flight is round 13's defect; this asserts the pre-flight's name is the
        writer's name rather than trusting that they were written to agree.
        """
        frame_output = import_module("gwmock_noise.output.frame")
        monkeypatch.setattr(frame_output, "_require_gwf_backend", lambda: None)
        writer = FrameWriter(FixedNoiseSimulator(), gps_start=1234.5, output_dir=tmp_path, prefix="run")

        composed = frame_output.compose_frame_name(
            detector="H1", channel="H1:MOCK_NOISE", gps_start=1234.5, duration=2.0, prefix="run"
        )

        assert writer._frame_path("H1", "H1:MOCK_NOISE", 1234.5, 2.0).name == composed


class TestWriteSegmentsChecksNamesToo:
    """`write` ran the character rules and `write_segments` did not.

    CodeRabbit found it on the PR, and the consequence was not cosmetic. `write_segments` went straight
    to `_check_frame_name_lengths`, which reaches `compose_frame_name` and evaluates `detector[0]`, so an
    empty detector raised `IndexError` from inside the composer while the docstring promised `ValueError`.
    The empty detector is exactly the case `reject_unsafe` exists to turn into a statement about the name.

    This is the reach problem the branch has now hit four times -- a rule present at one entry point and
    missing from another -- and `test_rule_reach.py` could not see it, because it drives `run`, and `run`
    never calls `write_segments`.
    """

    @staticmethod
    def _writer(directory: Path, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> FrameWriter:
        return _writer_over_fake_backend(directory, monkeypatch, **kwargs)

    def test_an_empty_detector_raises_a_value_error_rather_than_an_index_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reported failure: `IndexError` from `detector[0]`, where the docstring promises `ValueError`."""
        writer = self._writer(tmp_path, monkeypatch)

        with pytest.raises(ValueError, match="empty"):
            writer.write_segments([(0.0, 1.0)], sampling_frequency=4.0, detectors=[""])

    @pytest.mark.parametrize(
        ("detectors", "channel", "expected"),
        [
            (["H1/A"], "MOCK_NOISE", "path syntax"),
            (["H1|A"], "MOCK_NOISE", "reserved by Windows"),
            (["H1\nA"], "MOCK_NOISE", "Windows reserves"),
            (["H1", "h1"], "MOCK_NOISE", "collide"),
            (["H1", "H1"], "MOCK_NOISE", "more than once"),
        ],
    )
    def test_every_name_rule_reaches_write_segments(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        detectors: list[str],
        channel: str,
        expected: str,
    ) -> None:
        """The same rules `write` enforces, at the entry point that was missing them.

        Parametrised over the rules rather than asserting one of them, because the defect was never a
        wrong rule -- it was a rule that did not reach here.
        """
        writer = self._writer(tmp_path, monkeypatch, channel=channel)

        with pytest.raises(ValueError, match=expected):
            writer.write_segments([(0.0, 1.0)], sampling_frequency=4.0, detectors=detectors)

        assert list(tmp_path.glob("*.gwf")) == [], "a refusal must not leave a frame behind"

    def test_a_channel_replaced_after_construction_reaches_write_segments_too(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The channel half of the rule, which cannot be reached through the constructor.

        `FrameWriter(..., channel="MOCK/NOISE")` is refused at construction, so the only way a bad channel
        reaches either writing method is assignment afterwards -- `channel` is a plain public attribute.
        `write` already had a test for exactly this; `write_segments` did not.
        """
        writer = self._writer(tmp_path, monkeypatch)
        writer.channel = "MOCK/NOISE"

        with pytest.raises(ValueError, match="group separator"):
            writer.write_segments([(0.0, 1.0)], sampling_frequency=4.0, detectors=["H1"])

        assert list(tmp_path.glob("*.gwf")) == []

    def test_ordinary_segments_still_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The check must not disturb the case `write_segments` exists for."""
        writer = self._writer(tmp_path, monkeypatch)

        written = writer.write_segments([(0.0, 1.0), (1.0, 2.0)], sampling_frequency=4.0, detectors=["H1"])

        assert [path["H1"].name for path in written] == ["H-H1_MOCK_NOISE_0-1.gwf", "H-H1_MOCK_NOISE_1-1.gwf"]


class TestNoColonSurvivesIntoAFrameName:
    """Round 19's own defect: the `IFO:` strip fell back to the raw channel when the suffix was empty.

    `compose_frame_name` took `channel.partition(":")` and then `channel_name or channel`, so a channel
    whose colon is trailing -- `"H1:"`, or `":"` alone -- produced an empty suffix, fell back to the
    unstripped channel, and put the colon straight back into the file name: `noise_H-H1_H1:_0-1.gwf`.
    The one-colon rule passed it, because there *is* only one colon.

    Codex found it by probing the composer rather than by reading it, which is the only way this was
    going to be found: the fallback exists for the colon-less case, and `partition` returns an empty
    suffix for both, so the two cases are indistinguishable to `or`.
    """

    @pytest.mark.parametrize("channel", ["H1:", ":", "X1:"])
    def test_a_channel_whose_colon_is_trailing_is_refused(self, channel: str) -> None:
        """An empty channel name is not a name, whatever precedes the colon."""
        with pytest.raises(ValueError, match="empty"):
            gwmock_noise.OutputConfig(format="gwf", channels={"H1": channel})

    @pytest.mark.parametrize("channel", ["H1:MOCK_NOISE", "MOCK_NOISE", "X1:A"])
    def test_the_composed_name_never_carries_a_colon(self, channel: str) -> None:
        """The property the composer's comment claims, asserted over the shapes a channel can take.

        Including the colon-less channel, which is what the fallback was there for.
        """
        frame_output = import_module("gwmock_noise.output.frame")

        name = frame_output.compose_frame_name(
            detector="H1", channel=channel, gps_start=0.0, duration=1.0, prefix="noise"
        )

        assert ":" not in name, name

    @pytest.mark.parametrize("channel", ["H1:", ":", "X1:"])
    def test_the_composer_refuses_to_emit_a_colon_even_when_called_directly(self, channel: str) -> None:
        """`reject_unsafe` stops these at the config, and `compose_frame_name` is still module-level.

        A direct caller goes through no config at all -- the same reason `FrameWriter` re-checks names
        that `OutputConfig` already checked. Without this test the composer's explicit branch is
        unfalsifiable: a mutation restoring the `or channel` fallback survived the entire suite, because
        every other path that reaches the composer has had the input rejected for it. A defence no test
        can distinguish is a claim, not a defence -- this file has removed one such check before.
        """
        frame_output = import_module("gwmock_noise.output.frame")

        name = frame_output.compose_frame_name(
            detector="H1", channel=channel, gps_start=0.0, duration=1.0, prefix="noise"
        )

        assert ":" not in name, name

    def test_a_colon_less_channel_still_reaches_the_name(self) -> None:
        """The fallback's actual purpose, kept: `partition` returns an empty suffix here too."""
        frame_output = import_module("gwmock_noise.output.frame")

        name = frame_output.compose_frame_name(
            detector="H1", channel="MOCK_NOISE", gps_start=0.0, duration=1.0, prefix=""
        )

        assert name == "H-H1_MOCK_NOISE_0-1.gwf"
