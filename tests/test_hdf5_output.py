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

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
from _pytest.outcomes import Skipped

from gwmock_noise import NoiseComponentConfig, NoiseConfig, OutputConfig
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


def _require_gwpy_or_skip() -> None:
    """Skip without GWpy, unless this is the environment meant to exercise the guard.

    Extracted so the *failure* path can be tested. A guard whose failure branch has never run is a guard
    on paper: this one exists because the compatibility test previously skipped everywhere and read as
    passing for a whole review round.

    Raises:
        AssertionError: If `GWMOCK_NOISE_REQUIRE_GWPY` is set and GWpy is nonetheless absent.
    """
    installed = importlib.util.find_spec("gwpy") is not None
    if os.environ.get("GWMOCK_NOISE_REQUIRE_GWPY") == "1":
        assert installed, (
            "GWMOCK_NOISE_REQUIRE_GWPY is set, so this is the environment meant to exercise the GWpy "
            "compatibility guard -- but GWpy is not installed, so the guard would have skipped instead."
        )
        return
    if not installed:
        # `find_spec` rather than `pytest.importorskip`, so both branches can be exercised: importorskip
        # calls `find_spec` internally, so a test that fakes the module's absence breaks importorskip's
        # own machinery instead of reaching this branch.
        pytest.skip("gwpy is not installed")


class TestTheGuardItself:
    """The skip logic, because a compatibility guard that silently skips is the failure it prevents."""

    def test_it_fails_when_the_environment_demanded_gwpy_and_lacks_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The CI leg that installs the extra must not go quiet if the extra disappears."""
        monkeypatch.setenv("GWMOCK_NOISE_REQUIRE_GWPY", "1")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        with pytest.raises(AssertionError, match="would have skipped"):
            _require_gwpy_or_skip()

    def test_it_skips_where_gwpy_is_merely_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Everywhere else, a missing optional dependency is a skip rather than a failure."""
        monkeypatch.delenv("GWMOCK_NOISE_REQUIRE_GWPY", raising=False)
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)

        with pytest.raises(Skipped):
            _require_gwpy_or_skip()

    def test_it_proceeds_when_gwpy_is_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """And does nothing at all in the ordinary case."""
        monkeypatch.setenv("GWMOCK_NOISE_REQUIRE_GWPY", "1")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())

        _require_gwpy_or_skip()


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

    **This skips where GWpy is absent, and CI has one leg that installs it.** A guard that skips
    everywhere is indistinguishable from one that passes, which is exactly how this test spent a review
    round broken. `GWMOCK_NOISE_REQUIRE_GWPY` is set on that leg, and turns the skip into a failure, so
    losing the leg is loud rather than silent.
    """
    _require_gwpy_or_skip()
    from gwpy.timeseries import TimeSeries

    result = DefaultNoiseSimulator().run(_config(tmp_path))

    # No `channel=`: gwpy's HDF5 reader forwards **kwargs to `read_hdf5_array`, which does not accept
    # it, so the call raised `TypeError` wherever gwpy was installed and skipped everywhere else -- a
    # guard that had never once run green. A reviewer found it.
    series = TimeSeries.read(str(result.output_paths["H1"]))
    assert float(series.t0.value) == pytest.approx(1000000000.0)
    assert float(series.sample_rate.value) == pytest.approx(128.0)
    assert series.size == 512
    # What GWpy *recovers*, not only that it loaded: the h5py tests pin what the writer stores, so a
    # reader silently losing the unit or the channel -- the "anonymous numbers" failure from round 1 --
    # would have passed this guard. A reviewer pointed that out.
    assert str(series.unit) == "strain"
    assert str(series.channel) == "H1:MOCK_NOISE"


class TestTheNameIsUniquePerDetector:
    """Naming is the part of this writer that has been wrong twice, in opposite directions.

    Note what is *not* claimed: that the two formats name a file identically. They deliberately do not --
    a frame keeps the channel (without its `IFO:` prefix, since Windows reserves the colon), this keeps
    the detector alone -- and an earlier version of this docstring said otherwise while the tests below
    checked only the site letter and the absence of a colon.

    The original demonstration used `channels={"H1": "H1:A:B", "L1": "H1:A_B"}`, where the colon-escaping
    name produced `H-H1_A_B_...hdf5` for both detectors and one silently overwrote the other. That exact
    pair can no longer be configured: a channel carrying two colons is refused outright now, because the
    frame name drops only the first. So the property is demonstrated with channels that are *identical*
    instead, which is the strongest form of the same collision.
    """

    def test_two_detectors_never_share_a_file(self, tmp_path: Path) -> None:
        """Two detectors given one and the same channel still write two files.

        Naming for the detector makes the collision impossible by construction, which is the property
        worth asserting -- not that one particular escape sequence is handled.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:A_B", "L1": "H1:A_B"}))

        assert result.output_paths["H1"] != result.output_paths["L1"]
        assert len(list(tmp_path.glob("*.hdf5"))) == 2

    def test_each_file_holds_its_own_detector_s_channel(self, tmp_path: Path) -> None:
        """The other half of the collision: the data, not just the paths, must be distinct."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:A_B", "L1": "L1:C_D"}))

        with h5py.File(result.output_paths["H1"], "r") as handle:
            assert list(handle) == ["H1:A_B"]
        with h5py.File(result.output_paths["L1"], "r") as handle:
            assert list(handle) == ["L1:C_D"]

    def test_a_channel_carrying_two_colons_is_refused(self, tmp_path: Path) -> None:
        """The pair above used to be configurable, and the rule that replaced it is asserted here.

        Only the leading `IFO:` is dropped when a channel enters a frame name, so a second colon would
        survive into the file name and NTFS would read it as an alternate data stream.
        """
        with pytest.raises(ValueError, match="more than one ':'"):
            _config(tmp_path, channels={"H1": "H1:A:B"})

    def test_an_override_does_not_move_the_site_letter(self, tmp_path: Path) -> None:
        """`channels={"H1": "L1:CUSTOM"}` is still H1's data, so the name still starts with H."""
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "L1:CUSTOM"}))

        assert result.output_paths["H1"].name.startswith("H-H1_")

    def test_a_channel_containing_a_path_separator_is_refused(self, tmp_path: Path) -> None:
        """A slash cannot be written at all, so it is rejected where it is configured.

        This test previously asserted that such a channel "still writes one file", and passed -- on an
        artifact GWpy could not read. The slash was out of the *name* but still in the HDF5 dataset path,
        where h5py treats it as a group separator, so the writer produced a nested group, returned a
        path, and reported success. Round 2 at least failed loudly with `FileNotFoundError`; the naming
        rework turned that into a silent corrupt file. Both reviewers caught it.

        Rejected where the config is built *and* asserted again in the writer. The boundary alone was not
        enough: `model_construct` skips validators, this repo's own tests use it, and a reviewer
        reproduced the nested group that way. A guarantee that holds only for validator-built configs is
        not the guarantee the docstring used to claim.
        """
        with pytest.raises(ValueError, match="artifact name"):
            _config(tmp_path, channel="MOCK/NOISE")

    def test_a_detector_carrying_path_syntax_is_refused(self, tmp_path: Path) -> None:
        """`detectors=["H1:A"]` put a colon back into the file name it is used to build.

        The Windows-safety property held only for ordinary detector names until this was rejected -- a
        reviewer pointed out that the guarantee was about the inputs I had tried, not about the inputs
        the config accepts.
        """
        with pytest.raises(ValueError, match="artifact name"):
            NoiseConfig(
                detectors=["H1:A"],
                duration=4.0,
                sampling_frequency=128.0,
                output=OutputConfig(directory=tmp_path, format="hdf5"),
            )

    def test_the_name_has_no_colon_so_it_is_a_file_on_windows(self, tmp_path: Path) -> None:
        """On NTFS a colon opens an alternate data stream, so the artifact would not be a file."""
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        for path in result.output_paths.values():
            assert ":" not in path.name


class TestTheSidecar:
    """What a consumer can learn without opening the artifact."""

    def test_it_records_the_channel_the_file_holds(self, tmp_path: Path) -> None:
        """HDF5 names carry the detector, not the channel, so the sidecar has to carry it.

        A reviewer noted the rename left the channel discoverable only by opening every file. It is not
        a break, but the sidecar exists precisely so a consumer does not have to.
        """
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        sidecar = json.loads((tmp_path / "_H1.json").read_text())
        assert sidecar["channel"] == "H1:MOCK_NOISE"
        assert sidecar["artifact_path"] == str(result.output_paths["H1"])

    def test_it_records_an_overridden_channel(self, tmp_path: Path) -> None:
        """The sidecar reuses the writer's own resolution, so an override cannot desynchronise them."""
        h5py = pytest.importorskip("h5py")
        result = DefaultNoiseSimulator().run(_config(tmp_path, channels={"H1": "H1:CUSTOM"}))

        sidecar = json.loads((tmp_path / "_H1.json").read_text())
        with h5py.File(result.output_paths["H1"], "r") as handle:
            assert list(handle) == [sidecar["channel"]]

    def test_a_numpy_artifact_advertises_no_channel(self, tmp_path: Path) -> None:
        """A bare array has none, and claiming one would be inventing it."""
        DefaultNoiseSimulator().run(_config(tmp_path, format="npy"))

        sidecar = json.loads((tmp_path / "_H1.json").read_text())
        assert "channel" not in sidecar


class TestWhereTheRejectionApplies:
    """Which configurations the name rule binds, and which it must leave alone."""

    def test_a_numpy_config_may_carry_any_channel(self, tmp_path: Path) -> None:
        """`npy` writes a bare array and never reads the channel.

        Rejecting it there turned a configuration that had worked into a hard failure on upgrade, for no
        benefit -- both reviewers flagged it. The rule binds only the formats that use the channel, which
        is why it is a model validator rather than a field one: a field validator cannot see `format`.
        """
        assert OutputConfig(format="npy", channel="MOCK/NOISE").channel == "MOCK/NOISE"
        assert OutputConfig(format="npy", channels={"H1": "H1:A/B"}).channels == {"H1": "H1:A/B"}

    @pytest.mark.parametrize("fmt", ["gwf", "hdf5"])
    def test_a_format_that_uses_the_channel_rejects_it(self, fmt: str) -> None:
        """Both formats put the channel somewhere a slash would break."""
        with pytest.raises(ValueError, match="artifact name"):
            OutputConfig(format=fmt, channel="MOCK/NOISE")

    @pytest.mark.parametrize("fmt", ["gwf", "hdf5"])
    def test_an_override_is_rejected_for_those_formats_too(self, fmt: str) -> None:
        """A per-detector override reaches the same places the default channel does."""
        with pytest.raises(ValueError, match="artifact name"):
            OutputConfig(format=fmt, channels={"H1": "H1:A/B"})

    def test_an_override_key_carrying_path_syntax_is_rejected(self) -> None:
        """An override key carrying path syntax is refused.

        The key is a detector name, and could never match a validated detector, so the entry is inert
        rather than dangerous -- but an inert override is almost certainly a typo, and saying so beats
        ignoring it.
        """
        with pytest.raises(ValueError, match="artifact name"):
            OutputConfig(format="hdf5", channels={"H1:A": "H1:STRAIN"})


class TestTheWriterAssertsItsOwnPrecondition:
    """Because the config boundary is bypassable, and this repo bypasses it.

    Checked before anything is generated, and over every name the run will use -- not per detector as the
    writing proceeds. Both of those were wrong in the first version and both were demonstrated: a
    per-detector check did the whole simulation before refusing, and wrote the good detectors' files
    before reaching the bad one.
    """

    @staticmethod
    def _bypassed(directory: Path, **output: object) -> NoiseConfig:
        """A config built the way `model_construct` builds one: no validators, no rejection."""
        settings: dict[str, object] = {
            "directory": directory,
            "format": "hdf5",
            "channel": "MOCK_NOISE",
            "channels": None,
            "prefix": "",
            "gps_start": 1000000000.0,
        }
        settings.update(output)
        return NoiseConfig.model_construct(
            detectors=list(output.pop("detectors", ["H1"])) if "detectors" in output else ["H1"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(**settings),
        )

    def test_a_bypassed_channel_is_refused(self, tmp_path: Path) -> None:
        """The round-3 nested group, reached through `model_construct`."""
        with pytest.raises(ValueError, match="group separator"):
            DefaultNoiseSimulator().run(self._bypassed(tmp_path, channel="MOCK/NOISE"))

        assert list(tmp_path.glob("*.hdf5")) == []

    def test_a_bypassed_detector_is_refused(self, tmp_path: Path) -> None:
        """A colon in a detector reaches the file name, and the writer checked only channels.

        `detectors=["H1:A"]` wrote `H-H1:A_...hdf5` past the writer-level protection -- the drift between
        the config's rule and the writer's, which both reviewers found. One shared rule now, so "the
        writer re-asserts it" means the same rule rather than a subset of it.
        """
        config = NoiseConfig.model_construct(
            detectors=["H1:A"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="hdf5",
                channel="MOCK_NOISE",
                channels=None,
                prefix="",
                gps_start=1000000000.0,
            ),
        )

        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.glob("*")) == []

    def test_one_bad_name_among_several_leaves_nothing_behind(self, tmp_path: Path) -> None:
        """The partial write: good detectors' files must not survive a refusal.

        With the check inside the writing loop, `detectors=["H1", "L1"]` and a bad override for L1 wrote
        H1's artifact and then raised, leaving the caller a set they never chose to keep. The earlier
        version of this test used a single detector, which fails before any write and so could not see
        it -- a reviewer pointed that out.
        """
        config = NoiseConfig.model_construct(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="hdf5",
                channel="MOCK_NOISE",
                channels={"L1": "H1:A/B"},
                prefix="",
                gps_start=1000000000.0,
            ),
        )

        with pytest.raises(ValueError, match="group separator"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.glob("*")) == []

    def test_an_ordinary_bypassed_config_still_writes(self, tmp_path: Path) -> None:
        """The check must refuse bad names, not `model_construct` itself."""
        config = NoiseConfig.model_construct(
            detectors=["H1", "L1"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="hdf5",
                channel="MOCK_NOISE",
                channels=None,
                prefix="",
                gps_start=1000000000.0,
            ),
        )

        result = DefaultNoiseSimulator().run(config)

        assert sorted(result.output_paths) == ["H1", "L1"]


class TestEveryFormatChecksItsDetectorNames:
    """The check ran in the HDF5 branch alone, so `npy` and `gwf` kept the bypass.

    Both reviewers found it independently in round 6. It matters beyond HDF5 because every format names
    its artifact *and* its JSON sidecar after the detector, so path syntax in a detector escapes the
    output directory whatever the format -- and the failure is worst where it is quietest: with the
    subdirectory already present, the run succeeds and reports a path inside it.
    """

    @staticmethod
    def _bypassed(directory: Path, detectors: list[str], **output: object) -> NoiseConfig:
        settings: dict[str, object] = {
            "directory": directory,
            "format": "npy",
            "channel": "MOCK_NOISE",
            "channels": None,
            "prefix": "noise",
            "gps_start": 1000000000.0,
        }
        settings.update(output)
        return NoiseConfig.model_construct(
            detectors=detectors,
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(**settings),
        )

    def test_numpy_refuses_a_detector_whose_directory_already_exists(self, tmp_path: Path) -> None:
        """The silent case: `H1/A` wrote `noise_H1/A.npy` and the run reported success.

        The pre-existing directory is the point. Without it the run still failed, but loudly and only
        after generating everything; with it, `run()` returned a path one level below where the caller
        asked for output, and wrote the sidecar there too.
        """
        (tmp_path / "noise_H1").mkdir()

        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(self._bypassed(tmp_path, ["H1/A"]))

        assert list((tmp_path / "noise_H1").iterdir()) == []

    def test_numpy_refuses_a_detector_carrying_a_colon(self, tmp_path: Path) -> None:
        """`noise_H1:A.npy` is an ordinary name on this machine and an NTFS data stream elsewhere."""
        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(self._bypassed(tmp_path, ["H1:A"]))

        assert list(tmp_path.iterdir()) == []

    def test_numpy_still_accepts_a_channel_it_never_writes(self, tmp_path: Path) -> None:
        """The channel rule must not follow the detector rule across formats.

        An `npy` artifact is a bare array with no channel in it or in its name, so refusing a slashed
        channel would reject a configuration whose channel is never used. This is the same
        over-broadening that round 4 found at the config boundary, at the writer instead.
        """
        result = DefaultNoiseSimulator().run(self._bypassed(tmp_path, ["H1"], channel="MOCK/NOISE"))

        assert sorted(path.name for path in tmp_path.iterdir()) == ["noise_H1.json", "noise_H1.npy"]
        assert result.output_paths["H1"] == tmp_path / "noise_H1.npy"

    def test_frames_refuse_a_bad_detector_before_reaching_the_writer(self, tmp_path: Path) -> None:
        """`gwf` had the same gap. Asserted without a GWF backend, deliberately.

        The frame writer raises `ImportError` when no backend is installed, so a run that gets that far
        proves the pre-flight did not fire. Demanding `ValueError` here therefore discriminates on any
        machine: with a backend it would otherwise have written `H-H1:A_...gwf`, without one it would
        have raised the import error instead.
        """
        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(self._bypassed(tmp_path, ["H1:A"], format="gwf"))

        assert list(tmp_path.iterdir()) == []

    def test_the_preflight_refuses_a_frame_channel_before_the_directory_exists(self, tmp_path: Path) -> None:
        """What the simulator's pre-flight still adds for `gwf`, now that the writer checks too.

        `FrameWriter` validates its own inputs, so narrowing the pre-flight's channel argument back to
        `hdf5` alone stopped changing the *outcome* -- a mutation doing exactly that survived once N1
        landed. The difference is *when*: the pre-flight runs before `mkdir`, the writer long after it.
        Without the pre-flight this run still refuses, but leaves the output directory behind.
        """
        target = tmp_path / "not-created-yet"

        with pytest.raises(ValueError, match="group separator"):
            DefaultNoiseSimulator().run(self._bypassed(target, ["H1"], format="gwf", channel="MOCK/NOISE"))

        assert not target.exists()

    def test_frames_refuse_a_bad_channel_too(self, tmp_path: Path) -> None:
        """A frame carries a channel, so the channel rule reaches `gwf` as well as HDF5.

        Written because a mutation survived: narrowing the pre-flight's channel argument from
        ``{"gwf", "hdf5"}`` to ``"hdf5"`` alone changed nothing any test could see, which is exactly the
        shape of the round-6 defect -- a rule covering one format while the docstring claims two.
        """
        with pytest.raises(ValueError, match="group separator"):
            DefaultNoiseSimulator().run(self._bypassed(tmp_path, ["H1"], format="gwf", channel="MOCK/NOISE"))

        assert list(tmp_path.iterdir()) == []

    def test_a_refused_run_does_not_create_the_output_directory(self, tmp_path: Path) -> None:
        """The check has to precede `mkdir`, not merely precede writing.

        The other tests in this class cannot see this: they point at `tmp_path`, which pytest has already
        created, so the `mkdir` is a no-op and "nothing left behind" holds either way. A real caller
        naming a directory that does not exist yet had it created and then kept, by a run that refused to
        do anything. A reviewer pointed out both the behaviour and why the tests were blind to it.
        """
        target = tmp_path / "not-created-yet"

        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(self._bypassed(target, ["H1/A"]))

        assert not target.exists()

    def test_a_broken_component_does_not_mask_the_name_error(self, tmp_path: Path) -> None:
        """Which error the caller sees, when the config is wrong in two ways at once.

        `_configure_simulator` ran before the check, so an unloadable component raised first and the
        caller was told about the component while the name went unmentioned -- and, a reviewer noted, a
        colored component reads its PSD file in `__init__`, so the masking error can come from disk I/O
        performed on behalf of a run that was never going to be allowed.
        """
        config = self._bypassed(tmp_path / "not-created-yet", ["H1/A"])
        config.components = [NoiseComponentConfig.model_construct(simulator="does-not-exist", options={})]

        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(config)

        assert not (tmp_path / "not-created-yet").exists()


class TestThePrefixIsANameToo:
    """`prefix` reached every artifact name and no layer checked it, on any path."""

    def test_a_validated_config_refuses_a_prefix_with_path_syntax(self) -> None:
        """Not a bypass: this is the ordinary constructor, and it wrote `sub/run_H1.npy`."""
        with pytest.raises(ValueError, match="path syntax"):
            OutputConfig(format="npy", prefix="sub/run")

    def test_the_prefix_rule_applies_to_every_format(self) -> None:
        """Unlike the channel: every format prepends the prefix, none writes it inside the artifact."""
        for artifact_format in ("npy", "gwf", "hdf5"):
            with pytest.raises(ValueError, match="path syntax"):
                OutputConfig(format=artifact_format, prefix="sub/run")

    def test_a_prefix_carrying_a_colon_is_refused(self) -> None:
        """The prefix takes the detector rule, not the channel one.

        A mutation swapping the two survived: the channel rule permits `:`, which is right for a channel
        written inside a frame name and wrong for a prefix, where it opens an alternate data stream on
        NTFS. Nothing distinguished the two rules for this field until this test.
        """
        with pytest.raises(ValueError, match="path syntax"):
            OutputConfig(format="npy", prefix="run:a")

    def test_an_ordinary_prefix_still_works(self) -> None:
        """The rule must not reject the prefixes the examples and tests already use."""
        assert OutputConfig(format="npy", prefix="noise").prefix == "noise"
        assert OutputConfig(format="hdf5", prefix="run_a").prefix == "run_a"

    def test_a_bypassed_prefix_is_refused_by_the_simulator(self, tmp_path: Path) -> None:
        """And the writer re-asserts it, as it does for the other two names."""
        target = tmp_path / "not-created-yet"
        config = NoiseConfig.model_construct(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=target,
                format="npy",
                channel="MOCK_NOISE",
                channels=None,
                prefix="sub/run",
                gps_start=1000000000.0,
            ),
        )

        with pytest.raises(ValueError, match="path syntax"):
            DefaultNoiseSimulator().run(config)

        assert not target.exists()


class TestAnEmptyNameIsNotAName:
    """A rule written as a character test passes the empty string, which names nothing.

    Found by a reviewer in the confirming round. The other reviewer read these as pre-existing and
    non-blocking; on `main` every HDF5 case raises `ValidationError` instead, because the format could
    not be represented at all -- so the HDF5 failures are surfaces this branch opened, not old ones.
    """

    def test_an_empty_detector_is_refused(self) -> None:
        """`npy` wrote `noise_.npy`; HDF5 and GWF raised `IndexError` from `detector[0]`."""
        with pytest.raises(ValueError, match="empty"):
            NoiseConfig(detectors=[""], duration=1.0, sampling_frequency=4.0, seed=1)

    def test_an_empty_channel_is_refused_for_the_formats_that_carry_one(self) -> None:
        """Empty channels reached h5py as a dataset name."""
        for artifact_format in ("gwf", "hdf5"):
            with pytest.raises(ValueError, match="empty"):
                OutputConfig(format=artifact_format, channel="")

    def test_an_empty_channel_override_is_refused(self) -> None:
        """The worst of the three: it raised `TypeError` *after* creating the file.

        `noise_H-H1_0-1.hdf5` was left behind by a run that then failed -- the partial write this branch
        spent several rounds eliminating, reintroduced through a name nobody thought to reject.
        """
        with pytest.raises(ValueError, match="empty"):
            OutputConfig(format="hdf5", channels={"H1": ""})

    def test_an_empty_prefix_is_still_the_default(self) -> None:
        """The prefix is exempt, because empty is what "no prefix" means.

        The rule must not become "no name may be empty": that would reject the default configuration and
        every example that omits the field.
        """
        # The field's default is "noise", not the empty string -- what matters is that empty is
        # *accepted*, which is what the exemption is for and what the tests here pass explicitly.
        assert OutputConfig(format="hdf5").prefix == "noise"
        assert OutputConfig(format="npy", prefix="").prefix == ""
        assert OutputConfig(format="hdf5", prefix="").prefix == ""

    def test_an_empty_channel_is_still_fine_for_npy(self) -> None:
        """Unused names stay unchecked, as with the character rule.

        `npy` writes a bare array with no channel in it or in its name, so an empty channel there is
        inert. Rejecting it would refuse configurations that work.
        """
        assert OutputConfig(format="npy", channel="").channel == ""


class TestTheCurrentGroupToken:
    """`.` is HDF5's name for the current group, so it cannot name a dataset.

    Found in round 11, and the same shape as the empty channel: h5py raises after opening the file,
    leaving a partial artifact. It is the third degenerate value a pure character rule could not see --
    after the empty string, and unlike whitespace, which is merely odd.
    """

    def test_a_dot_channel_is_refused(self) -> None:
        """It reached `create_dataset` and failed with `noise_H-H1_0-1.hdf5` already written."""
        with pytest.raises(ValueError, match="current group"):
            OutputConfig(format="hdf5", channels={"H1": "."})

    def test_a_dot_channel_is_refused_as_the_shared_channel_too(self) -> None:
        """Not only through an override."""
        with pytest.raises(ValueError, match="current group"):
            OutputConfig(format="hdf5", channel=".")

    def test_a_dot_dot_channel_is_deliberately_allowed(self) -> None:
        """The mirror case, kept working on purpose.

        `..` reads like the same class and is not: h5py creates the dataset, GWpy round-trips it, and it
        is addressable as both `handle[".."]` and `handle["/.."]`. That was measured rather than assumed
        -- the expectation was that it would resolve to the parent group. Rejecting it would refuse a
        configuration that works, which this rule has already had to walk back twice.
        """
        assert OutputConfig(format="hdf5", channels={"H1": ".."}).channels == {"H1": ".."}

    def test_a_dot_prefix_and_detector_are_unaffected(self) -> None:
        """The token only matters where it names an HDF5 object, not a file."""
        assert OutputConfig(format="hdf5", prefix=".").prefix == "."
        assert NoiseConfig(detectors=["."], duration=1.0, sampling_frequency=4.0, seed=1).detectors == ["."]

    def test_the_dot_channel_writes_nothing_when_bypassed(self, tmp_path: Path) -> None:
        """And the writer refuses before the file exists, which is the point of the pre-flight."""
        target = tmp_path / "not-created-yet"
        config = NoiseConfig.model_construct(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=128.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=target,
                format="hdf5",
                channel="MOCK_NOISE",
                channels={"H1": "."},
                prefix="",
                gps_start=1000000000.0,
            ),
        )

        with pytest.raises(ValueError, match="current group"):
            DefaultNoiseSimulator().run(config)

        assert not target.exists()


class TestTheNulByte:
    """The fourth structurally-invalid value, and the only control character that breaks anything.

    Found in round 12, when the brief asked what the next one would be. HDF5 stores names as VLEN
    strings, which cannot embed a NUL: h5py raises after opening the file, leaving a partial artifact.
    A POSIX path cannot contain one either, so it breaks a `.npy` file name as well.
    """

    def test_a_nul_in_a_channel_is_refused(self) -> None:
        """It raised `VLEN strings do not support embedded NULLs` with the file already created."""
        with pytest.raises(ValueError, match="NUL"):
            OutputConfig(format="hdf5", channels={"H1": "a\x00b"})

    def test_a_lone_nul_channel_is_refused(self) -> None:
        """The other message h5py produces, from the same cause."""
        with pytest.raises(ValueError, match="NUL"):
            OutputConfig(format="hdf5", channel="\x00")

    def test_a_nul_in_a_detector_is_refused(self) -> None:
        """A detector becomes a file name, and a POSIX path cannot hold a NUL either."""
        with pytest.raises(ValueError, match="NUL"):
            NoiseConfig(detectors=["a\x00b"], duration=1.0, sampling_frequency=4.0, seed=1)

    def test_a_nul_in_a_prefix_is_refused(self) -> None:
        """The prefix is a file-name component, so the same applies."""
        with pytest.raises(ValueError, match="NUL"):
            OutputConfig(format="npy", prefix="a\x00b")

    @pytest.mark.parametrize("character", ["\n", "\t", "\r", "\x07"])
    def test_control_characters_are_refused_because_windows_reserves_them(self, character: str) -> None:
        """This test asserted the opposite until CI first ran on Windows, and the reversal is the point.

        It used to read "deliberately allowed", on a measurement that each of these round-trips through
        HDF5 and through a file name. The measurement was real and the conclusion was wrong, because it
        was taken on POSIX only: `windows-latest` refused every one of them with `OSError [Errno 22]
        Invalid argument`, for `npy`, `hdf5` and `gwf` alike. Windows reserves everything below `0x20`
        in a file name.

        Refusing them everywhere rather than only on Windows is the deliberate part: a configuration
        should not become invalid by being run on a different machine.
        """
        with pytest.raises(ValueError, match="Windows reserves"):
            OutputConfig(format="hdf5", channel=f"a{character}b")

    def test_del_is_still_allowed(self) -> None:
        """`0x7f` is not in the reserved range, and it was measured to work on all three platforms.

        Kept as its own case so the new rule's *edge* is asserted rather than assumed: a rule written as
        "control characters" rather than "below 0x20" would take this one too, and that would be the
        over-rejection this module has already had to walk back twice.
        """
        assert OutputConfig(format="hdf5", channel="a\x7fb").channel == "a\x7fb"

    @pytest.mark.parametrize("character", ["<", ">", '"', "|", "?", "*"])
    def test_the_windows_reserved_characters_are_refused(self, character: str) -> None:
        """The printable half of the same finding, and refused on every platform for the same reason."""
        with pytest.raises(ValueError, match="reserved by Windows"):
            OutputConfig(format="hdf5", channel=f"a{character}b")


class TestTheComposedNameLength:
    """The fifth structurally-invalid value, and the first found by fuzzing rather than by review.

    The limit belongs to the composed file name, not to any one field: a 248-character detector is fine
    alone and not once `noise_`, an epoch, a duration and `.hdf5` surround it. It is the partial-write
    class again -- with several detectors the over-long one failed *after* the earlier artifacts were on
    disk, so a run reported `OSError` from inside h5py and left an incomplete set.
    """

    def test_an_overlong_detector_is_refused(self, tmp_path: Path) -> None:
        """248 characters plus `noise_` and `.json` is 259 bytes, over the 255-byte component limit.

        Refused by the simulator rather than by the config, because only the writer knows how the name
        is composed: `npy` adds a prefix and a suffix, HDF5 adds an epoch and a duration too.
        """
        config = NoiseConfig(
            detectors=["a" * 248],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format="npy", prefix="noise", gps_start=0.0),
        )

        with pytest.raises(ValueError, match="over the 255-byte limit"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_a_length_that_fits_still_works(self, tmp_path: Path) -> None:
        """The boundary must not move: 240 characters composes to 250 bytes and is written."""
        detector = "a" * 240
        config = NoiseConfig(
            detectors=[detector],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format="npy", prefix="noise", gps_start=0.0),
        )

        path = DefaultNoiseSimulator().run(config).output_paths[detector]

        assert path.exists()
        assert len(path.name.encode("utf-8")) == 250

    def test_the_limit_counts_bytes_not_characters(self, tmp_path: Path) -> None:
        """200 accented characters is 400 bytes in UTF-8, so it must be refused.

        A character-count check would have passed this: the filesystem's limit is on the encoded name.
        """
        config = NoiseConfig.model_construct(
            detectors=["é" * 200],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="npy",
                channel="MOCK_NOISE",
                channels=None,
                prefix="noise",
                gps_start=0.0,
            ),
        )

        with pytest.raises(ValueError, match="bytes"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_one_overlong_detector_among_several_writes_nothing(self, tmp_path: Path) -> None:
        """The partial set the fuzz found: `H1` and `L1` were written, then the run raised."""
        config = NoiseConfig.model_construct(
            detectors=["H1", "L1", "a" * 300],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="hdf5",
                channel="MOCK_NOISE",
                channels=None,
                prefix="noise",
                gps_start=0.0,
            ),
        )

        with pytest.raises(ValueError, match="over the 255-byte limit"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_the_hdf5_name_is_checked_and_not_only_the_sidecar(self, tmp_path: Path) -> None:
        """An HDF5 name is six bytes longer than its sidecar, so one check cannot stand for the other.

        A mutation disabling the artifact-name check survived: for most lengths the sidecar check refuses
        first, so the outcome was unchanged. The gap is real and narrow -- a 239-character detector makes
        a 250-byte sidecar name, which fits, and a 256-byte HDF5 name, which does not, because HDF5 adds
        the site letter, the epoch and the duration. 238 is the last length that fits.
        """
        for length, expectation in ((238, "written"), (239, "refused")):
            directory = tmp_path / str(length)
            directory.mkdir()
            detector = "a" * length
            config = NoiseConfig(
                detectors=[detector],
                duration=1.0,
                sampling_frequency=4.0,
                seed=1,
                components=["white"],
                output=OutputConfig(directory=directory, format="hdf5", prefix="noise", gps_start=0.0),
            )

            if expectation == "written":
                path = DefaultNoiseSimulator().run(config).output_paths[detector]
                assert len(path.name.encode("utf-8")) == 255
            else:
                with pytest.raises(ValueError, match="HDF5 artifact name"):
                    DefaultNoiseSimulator().run(config)
                assert list(directory.iterdir()) == []

    def test_a_long_channel_is_unaffected(self, tmp_path: Path) -> None:
        """An HDF5 dataset name is not a path component, so the limit does not apply to it."""
        channel = "C" * 400
        config = NoiseConfig(
            detectors=["H1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format="hdf5", prefix="noise", gps_start=0.0, channel=channel),
        )

        path = DefaultNoiseSimulator().run(config).output_paths["H1"]

        assert path.exists()


class TestTheWriterAndThePreflightAgreeOnNames:
    """One name, one expression. The pre-flight used to re-derive what the writer composes.

    With `prefix=""` the writer wrote `_H1.npy` while the pre-flight checked `H1.npy`, so a name one byte
    under the limit passed the check and failed the write -- leaving the `.npy` artifact behind and raising
    on the sidecar. A reviewer found it. The check now calls the same helpers the writers call, which is
    why this test asserts the *names*, not just the refusal.
    """

    def test_an_empty_prefix_still_leads_the_name_with_an_underscore(self, tmp_path: Path) -> None:
        """Whatever one thinks of that name, both layers must agree on it."""
        config = _config(tmp_path, format="npy", prefix="")

        result = DefaultNoiseSimulator().run(config)

        assert result.output_paths["H1"].name == "_H1.npy"
        assert (tmp_path / "_H1.json").exists()

    def test_an_empty_prefix_is_measured_the_way_it_is_written(self, tmp_path: Path) -> None:
        """The case that slipped through: 250 characters, no prefix, refused before anything is written."""
        config = NoiseConfig(
            detectors=["D" * 250],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format="npy", prefix="", gps_start=0.0),
        )

        with pytest.raises(ValueError, match="over the 255-byte limit"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []


class TestTwoDetectorsOneFile:
    """`_hdf5_name` claimed the name was unique "by construction". It is injective, not unique.

    A reviewer caught the claim in round 14. The name is injective in the detector *as a Python string*;
    APFS and NTFS compare file names without regard to case, so `["H1", "h1"]` reported two paths and
    wrote one file, one detector's samples overwriting the other's. That is the same silent loss that made
    this writer name artifacts after the detector instead of the channel back in round 2.
    """

    @pytest.mark.parametrize("artifact_format", ["npy", "hdf5", "gwf"])
    def test_detectors_differing_only_in_case_are_refused(self, tmp_path: Path, artifact_format: str) -> None:
        """Every format: with one channel for both detectors, the *artifact* names collide.

        This docstring used to say the sidecar name alone was enough. It was not: the sidecar was never
        checked, and what refused these configurations was always the artifact name. The case the sidecar
        does catch on its own is in `TestSidecarsThatCollideWhenTheArtifactsDoNot`, where the channels
        differ and the frame names therefore do not collide.
        """
        config = NoiseConfig(
            detectors=["H1", "h1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format=artifact_format, prefix="noise", gps_start=0.0),
        )

        with pytest.raises(ValueError, match="collide"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_a_repeated_detector_is_refused(self) -> None:
        """The writers key output by detector, so a repeat silently produced one file for two entries."""
        with pytest.raises(ValueError, match="more than once"):
            NoiseConfig(detectors=["H1", "H1"], duration=1.0, sampling_frequency=4.0, seed=1)

    def test_normalisation_forms_are_refused(self, tmp_path: Path) -> None:
        """NFC and NFD spellings of one detector name one file, on this filesystem, measured directly.

        This test began as its opposite. A probe through the package appeared to show two files surviving,
        so the collision key dropped normalisation; this assertion then failed, and writing the two names
        with no package code involved showed one file holding the second payload -- the first detector's
        samples gone. The probe was wrong. Asserting the outcome on disk, rather than what the rule does,
        is what caught it.
        """
        import unicodedata

        nfc = unicodedata.normalize("NFC", "é1")
        nfd = unicodedata.normalize("NFD", "é1")
        assert nfc != nfd

        config = NoiseConfig(
            detectors=[nfc, nfd],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format="npy", prefix="noise", gps_start=0.0),
        )

        with pytest.raises(ValueError, match="collide"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_ordinary_detectors_are_unaffected(self, tmp_path: Path) -> None:
        """The guard must not disturb the case every other test in this file uses."""
        result = DefaultNoiseSimulator().run(_config(tmp_path))

        assert sorted(result.output_paths) == ["H1", "L1"]


class TestSidecarsThatCollideWhenTheArtifactsDoNot:
    """Round 17: the sidecar name is not checked, and the reason given for not checking it was wrong.

    The pre-flight checked artifact names only, on the argument that a sidecar is
    `{prefix}_{detector}.json`, so two detectors can only collide there if they collide in the detector --
    which collides their artifact names too. That holds for `npy` and `hdf5`, whose names are composed
    from the detector. It does not hold for `gwf`: a frame name embeds the *resolved channel*, so two
    detectors with distinct channel overrides compose distinct frame names while their sidecars still
    fold together.

    Measured on this branch before the fix: `detectors=["H1", "h1"]` with `channels={"H1": "X1:A",
    "h1": "Y1:B"}` wrote two frames and **one** sidecar, `noise_H1.json`, holding `h1`'s metadata. No
    error. The loss is filesystem-dependent -- APFS and NTFS fold case, ext4 does not -- which is exactly
    why `reject_colliding_names` folds case rather than asking the filesystem.

    Codex found it in round 17; the sidecar-name docstring in `TestTwoDetectorsOneFile` had claimed for
    three rounds that the sidecar was what refused these configurations, and it never was.
    """

    @staticmethod
    def _colliding_sidecars(directory: Path) -> NoiseConfig:
        """Two detectors whose frame names differ and whose sidecar names do not."""
        return NoiseConfig(
            detectors=["H1", "h1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(
                directory=directory,
                format="gwf",
                prefix="noise",
                gps_start=0.0,
                channel="MOCK_NOISE",
                channels={"H1": "X1:A", "h1": "Y1:B"},
            ),
        )

    def test_the_preflight_refuses_them(self, tmp_path: Path) -> None:
        """Asserted against the pre-flight directly, so it needs no GWF backend and cannot pass by error.

        Through `run` this raises before `FrameWriter` is constructed, so on a machine without a backend
        the pre-fix failure was an `ImportError` rather than a missing refusal -- a red test for the wrong
        reason. Calling the check itself makes the assertion say what it means on either machine.
        """
        with pytest.raises(ValueError, match="collide"):
            DefaultNoiseSimulator()._check_artifact_lengths(self._colliding_sidecars(tmp_path))

    def test_the_message_names_the_sidecar_rather_than_the_artifact(self, tmp_path: Path) -> None:
        """The frame names are fine here. A message blaming them would send the reader to the wrong name."""
        with pytest.raises(ValueError, match="metadata sidecar names"):
            DefaultNoiseSimulator()._check_artifact_lengths(self._colliding_sidecars(tmp_path))

    def test_a_refused_run_creates_nothing(self, tmp_path: Path) -> None:
        """The refusal has to land before the output directory, as every other name rule here does."""
        target = tmp_path / "not-created-yet"

        with pytest.raises(ValueError, match="collide"):
            DefaultNoiseSimulator().run(self._colliding_sidecars(target))

        assert not target.exists()

    @pytest.mark.parametrize("artifact_format", ["npy", "hdf5"])
    def test_a_format_where_both_collide_still_blames_the_artifact(self, tmp_path: Path, artifact_format: str) -> None:
        """The sidecar check runs after the artifact check, and this is the assertion that says so.

        Without it the ordering is a claim in a comment: put the sidecar check first and every collision
        message changes to name a file the caller never chose, with no test to notice.
        """
        config = NoiseConfig(
            detectors=["H1", "h1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format=artifact_format, prefix="noise", gps_start=0.0),
        )

        with pytest.raises(ValueError, match=f"{artifact_format} artifact names"):
            DefaultNoiseSimulator()._check_artifact_lengths(config)

    def test_distinct_detectors_with_distinct_channels_are_unaffected(self, tmp_path: Path) -> None:
        """The guard must not refuse the ordinary per-detector override, which is the point of the feature."""
        config = NoiseConfig(
            detectors=["H1", "L1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(
                directory=tmp_path,
                format="gwf",
                prefix="noise",
                gps_start=0.0,
                channel="MOCK_NOISE",
                channels={"H1": "X1:A", "L1": "Y1:B"},
            ),
        )

        DefaultNoiseSimulator()._check_artifact_lengths(config)


class TestARepeatedDetectorOnTheBypassPath:
    """The repeat rule needed the same two layers every other name rule here has.

    Round 15: I put it in the config validator only. Both reviewers found that `run` collects its names
    into dicts keyed by detector, so an exact repeat collapses to one entry *before* the collision check
    looks at them -- on the very path the pre-flight exists to cover.
    """

    @pytest.mark.parametrize("artifact_format", ["npy", "hdf5"])
    def test_the_simulator_refuses_a_repeat_a_bypassed_config_carries(
        self, tmp_path: Path, artifact_format: str
    ) -> None:
        """It reported one path for two detectors and wrote one file, raising nothing."""
        config = NoiseConfig.model_construct(
            detectors=["H1", "H1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format=artifact_format,
                channel="MOCK_NOISE",
                channels=None,
                prefix="noise",
                gps_start=0.0,
            ),
        )

        with pytest.raises(ValueError, match="more than once"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []

    def test_case_differing_detectors_are_still_the_collision_rule_s_business(self, tmp_path: Path) -> None:
        """`["H1", "h1"]` are not repeats, and must be caught by the other rule rather than this one.

        The two are deliberately separate: an exact repeat collapses a dict key, while a case collision
        collapses a filename. Folding them together would put the filesystem's opinion about case into a
        check that runs before any name exists.
        """
        config = NoiseConfig.model_construct(
            detectors=["H1", "h1"],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=[],
            output=OutputConfig.model_construct(
                directory=tmp_path,
                format="npy",
                channel="MOCK_NOISE",
                channels=None,
                prefix="noise",
                gps_start=0.0,
            ),
        )

        with pytest.raises(ValueError, match="collide"):
            DefaultNoiseSimulator().run(config)

        assert list(tmp_path.iterdir()) == []
