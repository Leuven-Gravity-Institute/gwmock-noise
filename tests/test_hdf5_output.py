# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""Writing noise as HDF5.

Added because gwmock is making HDF5 its primary output format, and could not: every noise write goes
through :class:`OutputConfig`, whose ``format`` field rejected anything but ``npy`` and ``gwf``, so the
format could not even be *represented*, let alone written.

What matters beyond "a file appears" is that the file carries the same information a frame would. A bare
array of samples loses where those samples sit, and a consumer then has to be told the epoch and the
sample rate out of band -- which is exactly the gap that made the HDF5 content hash weaker than the GWF
one in the sibling project. So these tests assert the grid and the channel, not just the bytes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import NoiseConfig, OutputConfig
from gwmock_noise.simulators.default import DefaultNoiseSimulator

pytestmark = pytest.mark.unit


def _config(directory: Path, **overrides: object) -> NoiseConfig:
    output = {
        "directory": directory,
        "prefix": "",
        "format": "hdf5",
        "gps_start": 1000000000.0,
        "channel": "MOCK_NOISE",
    }
    output.update(overrides)
    return NoiseConfig(
        detectors=["H1", "L1"],
        duration=4.0,
        sampling_frequency=128.0,
        output=OutputConfig(**output),
        seed=7,
    )


class TestTheFormatIsRepresentable:
    """The validation that blocked this before any writer existed."""

    @pytest.mark.parametrize("fmt", ["npy", "gwf", "hdf5"])
    def test_every_supported_format_is_accepted(self, fmt: str) -> None:
        """Each format the writer can produce is also a format the config can hold."""
        assert OutputConfig(format=fmt).format == fmt

    def test_an_unsupported_format_is_still_rejected(self) -> None:
        """Widening the vocabulary must not turn the field into a free-text one."""
        with pytest.raises(ValueError, match="format"):
            OutputConfig(format="parquet")


class TestWhatGetsWritten:
    """The artifact, read back through the reader a consumer would use."""

    def test_one_file_per_detector(self, tmp_path: Path) -> None:
        """One artifact per detector, as the other formats produce."""
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        assert sorted(result.output_paths) == ["H1", "L1"]
        for path in result.output_paths.values():
            assert path.exists()
            assert path.suffix == ".hdf5"

    def test_the_file_carries_the_grid_not_just_the_samples(self, tmp_path: Path) -> None:
        """Epoch and sample rate survive the round trip.

        Without them the file is an array whose meaning lives somewhere else, and every downstream check
        -- including a content hash -- becomes blind to a segment written for the wrong time.
        """
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        with h5py.File(result.output_paths["H1"], "r") as handle:
            dataset = handle["H1:MOCK_NOISE"]
            assert float(dataset.attrs["x0"]) == pytest.approx(1000000000.0)
            assert float(dataset.attrs["dx"]) == pytest.approx(1.0 / 128.0)
            assert dataset.shape == (512,)  # 4 s at 128 Hz
            # Also the unit, so the file says what its samples *are*. Dropping it leaves a readable
            # file of anonymous numbers, which GWpy loads as dimensionless without complaint -- a
            # mutation removing it survived every other assertion here.
            assert dataset.attrs["unit"] == "strain"
            # `xunit` and `name` complete the set GWpy writes. Nothing asserted them, so a mutation
            # dropping either survived every h5py check while quietly narrowing what a reader gets.
            assert dataset.attrs["xunit"] == "s"
            assert dataset.attrs["name"] == "H1:MOCK_NOISE"

    def test_the_channel_is_named_as_a_frame_would_name_it(self, tmp_path: Path) -> None:
        """`DETECTOR:CHANNEL`, so a reader does not need to know which format it was handed."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        with h5py.File(result.output_paths["L1"], "r") as handle:
            assert list(handle) == ["L1:MOCK_NOISE"]
            assert handle["L1:MOCK_NOISE"].attrs["channel"] == "L1:MOCK_NOISE"

    def test_a_per_detector_channel_override_is_honoured(self, tmp_path: Path) -> None:
        """The frame writer supports overrides, so this must too, or the formats disagree."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:CUSTOM"}))

        with h5py.File(result.output_paths["H1"], "r") as handle:
            assert list(handle) == ["H1:CUSTOM"]

    def test_the_samples_are_the_generated_noise(self, tmp_path: Path) -> None:
        """Not zeros, and not a placeholder: the file holds what the simulator produced."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        with h5py.File(result.output_paths["H1"], "r") as handle:
            values = np.asarray(handle["H1:MOCK_NOISE"][()])
        assert np.isfinite(values).all()
        assert np.count_nonzero(values) > 0


class TestNaming:
    """The file name, which is how a directory listing is read."""

    def test_it_follows_the_frame_convention_with_an_hdf5_extension(self, tmp_path: Path) -> None:
        """`H-H1_<start>-<duration>.hdf5`: the detector, the epoch and the duration.

        The numpy writer's `prefix_detector` shape was one candidate and hides the epoch and duration,
        which the file carries. The frame writer's channel-in-the-name shape was the other, and put a
        colon in a name this writer must produce on Windows -- then escaping that colon made two
        channels collide onto one file. Naming for the detector avoids both; the channel lives in the
        dataset.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        assert result.output_paths["H1"].name == "H-H1_1000000000-4.hdf5"

    def test_a_prefix_is_applied(self, tmp_path: Path) -> None:
        """A configured prefix leads the name, matching the frame writer."""
        result = DefaultNoiseSimulator().run(_config(tmp_path, prefix="run7"))

        assert result.output_paths["H1"].name.startswith("run7_")


class TestTheOtherFormatsStillWork:
    """A widened vocabulary must not disturb what already worked."""

    def test_numpy_output_is_unchanged(self, tmp_path: Path) -> None:
        """Including its leading underscore, which is pre-existing and deliberately left alone.

        With an empty prefix the numpy writer produces `_H1.npy`, because it formats
        `f"{prefix}_{detector}.npy"` unconditionally. gwmock's own copy of that writer guards the empty
        case and produces `H1.npy`, so the two disagree. Changing it here would rename files for every
        existing user of npy output, which is a separate decision from adding a format -- this test pins
        today's behaviour so the disagreement is visible rather than surprising.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path, format="npy"))

        assert result.output_paths["H1"].name == "_H1.npy"
        assert np.load(result.output_paths["H1"]).size == 512


def test_gwpy_reads_what_h5py_wrote(tmp_path: Path) -> None:
    """The compatibility claim, asserted rather than assumed.

    The writer uses `h5py` because GWpy is an optional extra here and a primary format must not need one.
    The attributes it records are GWpy's own, so a reader with GWpy installed must not be able to tell --
    if that ever stops being true, the format has quietly forked.
    """
    pytest.importorskip("gwpy")
    from gwpy.timeseries import TimeSeries

    result = DefaultNoiseSimulator().run(_config(tmp_path))

    # No `channel=`: gwpy's HDF5 reader forwards **kwargs to `read_hdf5_array`, which does not accept
    # it, so the call raised `TypeError` wherever gwpy was installed and skipped everywhere else -- a
    # guard that had never once run green. A reviewer found it.
    series = TimeSeries.read(str(result.output_paths["H1"]))
    assert float(series.t0.value) == pytest.approx(1000000000.0)
    assert float(series.sample_rate.value) == pytest.approx(128.0)
    assert series.size == 512


class TestTheNameIsUniquePerDetector:
    """Naming is the part of this writer that has been wrong twice, in opposite directions."""

    def test_two_detectors_never_share_a_file(self, tmp_path: Path) -> None:
        """The collision that lost data: two channels escaping onto one name.

        With `channels={"H1": "H1:A:B", "L1": "H1:A_B"}` the colon-escaping name produced
        `H-H1_A_B_...hdf5` for both, so one detector silently overwrote the other and both returned
        paths pointed at the survivor. A reviewer demonstrated it on disk. Naming for the detector makes
        a collision impossible, and this asserts the property rather than that one escape.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:A:B", "L1": "H1:A_B"}))

        assert result.output_paths["H1"] != result.output_paths["L1"]
        assert len(list(tmp_path.glob("*.hdf5"))) == 2

    def test_each_file_holds_its_own_detector_s_channel(self, tmp_path: Path) -> None:
        """The other half of the collision: the data, not just the paths, must be distinct."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:A:B", "L1": "H1:A_B"}))

        with h5py.File(result.output_paths["H1"], "r") as handle:
            assert list(handle) == ["H1:A:B"]
        with h5py.File(result.output_paths["L1"], "r") as handle:
            assert list(handle) == ["H1:A_B"]

    def test_an_override_does_not_move_the_site_letter(self, tmp_path: Path) -> None:
        """`channels={"H1": "L1:CUSTOM"}` is still H1's data, so the name still starts with H."""
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "L1:CUSTOM"}))

        assert result.output_paths["H1"].name.startswith("H-H1_")

    def test_a_channel_containing_a_path_separator_still_writes_one_file(self, tmp_path: Path) -> None:
        """A channel is not a path component, and must not become one.

        `channel="MOCK/NOISE"` put a slash in the name, which opened a directory that did not exist and
        raised `FileNotFoundError`. The channel is no longer in the name at all, so this is structural
        rather than another escape to maintain.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path, channel="MOCK/NOISE"))

        assert result.output_paths["H1"].exists()
        assert result.output_paths["H1"].name == "H-H1_1000000000-4.hdf5"

    def test_the_name_has_no_colon_so_it_is_a_file_on_windows(self, tmp_path: Path) -> None:
        """On NTFS a colon opens an alternate data stream, so the artifact would not be a file."""
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        for path in result.output_paths.values():
            assert ":" not in path.name
