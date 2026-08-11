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

"""Ask the writers which names work, instead of waiting to be told which ones do not.

Five structurally-invalid values were found one at a time, each after a reviewer named it: an unchecked
prefix, the empty string, HDF5's ``.``, a NUL byte, and a name too long for a path component. Every one
was the same mistake -- a rule that tests characters cannot see a value of the wrong *shape* -- and
finding them one per review round is not a method.

So this asserts the invariant the character rules were only approximating:

    if the config accepts a name, the run writes the artifact it reports, inside the directory it was
    given, and it reads back

The failure it is built to catch is not the exception. It is the artifact left behind by a run that then
raised, because that is the shape every one of the five took.

The corpus is fixed and the generated part is seeded: a fuzz finding that cannot be reproduced is not a
finding. Add to it when a new failure is found rather than replacing it.
"""

from __future__ import annotations

import random
import string
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gwmock_noise import NoiseConfig, OutputConfig
from gwmock_noise.output import frame as frame_output
from gwmock_noise.simulators.default import DefaultNoiseSimulator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gwmock_noise.simulators.base import SimulationResult

pytestmark = pytest.mark.unit


class _FakeSeries:
    """A series that writes real bytes to the path the writer composed, and nothing else."""

    def __init__(self) -> None:
        self.channel = ""

    def write(self, path: Path, *, format: str, overwrite: bool) -> None:  # noqa: A002
        """Write a placeholder payload, so the artifact exists exactly where the writer said it would."""
        Path(path).write_bytes(b"fake-gwf")


class _FakeAdapter:
    """`GWpyAdapter`'s shape, without GWpy. Advances `gps_start` as the real one does."""

    def __init__(self, base: object, gps_start: float) -> None:
        self.base = base
        self.gps_start = gps_start

    def generate(
        self, *, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None
    ) -> dict[str, _FakeSeries]:
        """Return one series per detector, as the real adapter does."""
        self.gps_start += duration
        return {detector: _FakeSeries() for detector in detectors}


@contextmanager
def _stubbed_gwf_serialisation() -> Iterator[None]:
    """Drive the real `gwf` path with only the *serialisation* faked.

    `gwf` was absent from this fuzz until now because `FrameWriter` needs a GWF backend that the review
    machine does not have -- and the round-16 and round-17 defects were both `gwf`-only, which is why
    Isaac approved adding the axis.

    Stubbing the adapter, rather than composing the names here or driving the pre-flight by hand, is the
    point: either of those would be a second expression of what `run` does, which is round 13's defect.
    This drives `run` -> `compose_frame_name` -> `FrameWriter` for real and fakes only the bytes, the one
    part no name rule has a say in.

    What it therefore cannot assert for `gwf` is the read-back, since the payload is not a frame. The
    invariant all five structural findings actually broke -- a run that raises and leaves an artifact
    behind, or reports a path it never wrote -- is asserted in full.

    Yields:
        None, with the frame module's backend check and adapter replaced for the duration.
    """
    original_require = frame_output._require_gwf_backend
    original_adapter = frame_output.GWpyAdapter
    frame_output._require_gwf_backend = lambda: None
    frame_output.GWpyAdapter = _FakeAdapter
    try:
        yield
    finally:
        frame_output._require_gwf_backend = original_require
        frame_output.GWpyAdapter = original_adapter


def _run(config: NoiseConfig) -> SimulationResult:
    """Run the simulation, faking GWF serialisation only for the format that needs a backend.

    Returns:
        The simulation result.
    """
    if config.output.format != "gwf":
        return DefaultNoiseSimulator().run(config)
    with _stubbed_gwf_serialisation():
        return DefaultNoiseSimulator().run(config)


#: The artifact each format leaves on disk, for the assertions that count files rather than read them.
_SUFFIXES = {"npy": ".npy", "hdf5": ".hdf5", "gwf": ".gwf"}

_ALPHABET = string.ascii_letters + string.digits + "_-.: /\\" + "\x00\n\t" + "éλ√" + "🜁"

#: Shapes that have broken something, plus the neighbours of each. The five findings are all here.
_CORPUS: list[str] = [
    "H1",
    "MOCK_NOISE",
    "H1:STRAIN",
    "a",
    "_",
    "-",
    "0",
    "",
    " ",
    "   ",
    ".",
    "..",
    "...",
    ".hidden",
    "trailing.",
    "a.b",
    "\x00",
    "a\x00b",
    "a\nb",
    "a\tb",
    "a\rb",
    "\x7f",
    "\x07",
    "é",
    "λ",
    "√",
    "🜁",
    "a" * 200,
    "a" * 255,
    "a" * 256,
    "a" * 300,
    "é" * 200,  # 400 bytes in UTF-8: the limit is on the encoded name, not the character count
    "CON",
    "NUL",
    "aux",
    "a:b",
    "a/b",
    "a\\b",
    "/abs",
    "./rel",
    "../up",
    "a b",
    "-leading-dash",
    "--flag",
    "%s",
    "{}",
    "$HOME",
    "a;b",
    "a|b",
    "a*b",
    "a?b",
    '"q"',
    "'q'",
]


def _generated(count: int = 60) -> list[str]:
    """Return seeded random names over an alphabet weighted towards the characters that have hurt."""
    rng = random.Random(20260810)  # noqa: S311 -- naming a corpus, not making a secret
    return ["".join(rng.choice(_ALPHABET) for _ in range(rng.randint(1, 12))) for _ in range(count)]


def _refusal_left_nothing(target: Path, error: ValueError) -> tuple[str, str]:
    """Classify a refusal by what it left behind: nothing, a directory, or an artifact.

    Returns:
        ``("rejected", reason)`` if the refusal was clean, ``("broken", detail)`` otherwise.
    """
    if not target.exists():
        return "rejected", str(error).splitlines()[0][:60]
    written = sorted(path.name for path in target.rglob("*") if path.is_file())
    if written:
        return "broken", f"raised {str(error).splitlines()[0][:40]!r} yet left {written}"
    return "broken", f"raised {str(error).splitlines()[0][:40]!r} yet created the output directory"


def _attempt(directory: Path, artifact_format: str, **name: str) -> tuple[str, str]:  # noqa: PLR0911
    """Return ``("accepted", "")``, ``("rejected", reason)``, or ``("broken", what went wrong)``.

    One early return per way an artifact can be wrong. Collapsing them into a single exit would mean
    losing which check failed, and the detail is what makes a fuzz failure actionable rather than a
    report that something, somewhere, is broken.

    The run is pointed at a subdirectory that **does not exist yet**, and a refusal must leave it that
    way. Counting leftover *files* alone was not enough: round 16's defect created the output directory
    and then refused, and when the `gwf` axis was added this harness passed every mutation that restored
    that defect -- a directory is not a file, so nothing here could see it. The failure this fuzz exists
    to catch is the side effect that outlives a refusal, and a created directory is one.
    """
    detector = name.get("detector", "H1")
    target = directory / "out"
    settings = {
        "directory": target,
        "format": artifact_format,
        "prefix": name.get("prefix", "noise"),
        "gps_start": 0.0,
        "channel": name.get("channel", "MOCK_NOISE"),
    }
    try:
        config = NoiseConfig(
            detectors=[detector],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(**settings),
        )
        result = _run(config)
    except ValueError as error:
        # A refusal is allowed -- but it must be a refusal, not a partial write with an exception on top.
        return _refusal_left_nothing(target, error)
    except Exception as error:  # noqa: BLE001
        written = sorted(path.name for path in target.rglob("*") if path.is_file()) if target.exists() else []
        return "broken", f"{type(error).__name__}: {str(error).splitlines()[0][:40]} left={written}"

    path = result.output_paths[detector]
    if target.resolve() not in path.resolve().parents:
        return "broken", f"escaped the output directory: {path}"
    if not path.exists():
        return "broken", f"reported a path that was never written: {path}"
    try:
        if artifact_format == "npy":
            np.load(path)
        elif artifact_format == "gwf":
            # No read-back: the payload came from the stub, not from GWpy. The name is what this axis is
            # here to check, and it has already been asserted to exist inside the output directory.
            pass
        else:
            h5py = pytest.importorskip("h5py")
            # The dataset is always the resolved channel, which with no per-detector override is
            # `{detector}:{channel}`. An earlier version of this line computed it two ways and picked
            # between them on a condition that was always true, so it looked for the bare channel and
            # reported every detector case as broken -- a fuzz harness that reports its own bug as a
            # finding is worse than none.
            resolved_channel = f"{detector}:{settings['channel']}"
            with h5py.File(path, "r") as handle:
                if resolved_channel not in handle:
                    return "broken", f"dataset {resolved_channel!r} absent; keys={list(handle)}"
    except Exception as error:  # noqa: BLE001
        return "broken", f"unreadable: {type(error).__name__}: {str(error)[:40]}"
    return "accepted", ""


#: Epochs and durations enter composed names too, and they are what made the round-13 segment case bite:
#: a name that fits at epoch `0` overflows at a ten-digit GPS time. A single epoch cannot see that.
_GRIDS = [(0.0, 1.0), (1000000000.0, 4.0), (1234567890.5, 0.25)]

#: Pairs that name one file on a case-insensitive or normalising filesystem, which is the round-14 defect.
#: Distinct as Python strings in every case -- that is the point.
_COLLIDING_PAIRS = [
    ("H1", "h1"),
    ("MOCK", "mock"),
    ("\u00e91", "e\u03011"),  # NFC and NFD spellings of the same name
    ("AbC", "aBc"),
]


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5", "gwf"])
@pytest.mark.parametrize(("gps_start", "duration"), _GRIDS)
def test_a_name_that_fits_one_epoch_is_measured_against_the_epoch_used(
    tmp_path: Path, artifact_format: str, gps_start: float, duration: float
) -> None:
    """The epoch and the duration are part of the name, so the limit moves with them.

    Not covered when this file was written, and named as a gap in the round-14 brief rather than left
    silent: a detector whose name fits at epoch `0` can overflow at a ten-digit GPS time, which is exactly
    how the frame-segment case escaped notice.
    """
    broken: list[str] = []
    for index, length in enumerate((200, 230, 234, 238, 240, 250)):
        # Not created: a refusal that creates the output directory is the round-16 defect, and this
        # cannot see it if the harness makes the directory first.
        directory = tmp_path / f"{artifact_format}-{index}"
        detector = "d" * length
        try:
            config = NoiseConfig(
                detectors=[detector],
                duration=duration,
                sampling_frequency=4.0,
                seed=1,
                components=["white"],
                output=OutputConfig(directory=directory, format=artifact_format, prefix="noise", gps_start=gps_start),
            )
            path = _run(config).output_paths[detector]
        except ValueError:
            if directory.exists():
                leftover = sorted(p.name for p in directory.iterdir())
                broken.append(f"length {length}: refused yet left {leftover or 'the output directory'}")
            continue
        except OSError as error:
            broken.append(f"length {length}: OSError rather than a refusal ({error.errno})")
            continue
        if len(path.name.encode("utf-8")) > 255:
            broken.append(f"length {length}: wrote a {len(path.name.encode())}-byte name")

    assert not broken, f"at gps_start={gps_start}, duration={duration}:\n" + "\n".join(broken)


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5", "gwf"])
@pytest.mark.parametrize(("first", "second"), _COLLIDING_PAIRS)
def test_two_names_never_share_one_file(tmp_path: Path, artifact_format: str, first: str, second: str) -> None:
    """Distinct detectors must not end up writing the same artifact.

    The round-14 defect: the config accepted `["H1", "h1"]`, reported two paths, and wrote one file. The
    assertion is on the filesystem, not on the exception -- a refusal is fine, and so is writing two
    genuinely distinct files, but reporting two paths while one artifact exists is not.
    """
    target = tmp_path / "out"
    try:
        config = NoiseConfig(
            detectors=[first, second],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=target, format=artifact_format, prefix="noise", gps_start=0.0),
        )
        result = _run(config)
    except ValueError:
        assert not target.exists(), "a refusal must not leave an artifact, or the directory, behind"
        return

    reported = {str(path) for path in result.output_paths.values()}
    suffix = _SUFFIXES[artifact_format]
    written = [path for path in target.iterdir() if path.suffix == suffix]
    assert len(written) == len(reported), (
        f"reported {len(reported)} path(s) for {first!r} and {second!r} but {len(written)} file(s) exist"
    )


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5", "gwf"])
@pytest.mark.parametrize("field", ["detector", "prefix", "channel"])
def test_an_accepted_name_is_a_writable_name(tmp_path: Path, field: str, artifact_format: str) -> None:
    """Nothing the config accepts may fail in a writer, and nothing may be left behind by a refusal."""
    if field == "channel" and artifact_format == "npy":
        pytest.skip("npy carries no channel, so its channel names are deliberately unchecked")

    broken: list[str] = []
    for index, name in enumerate([*_CORPUS, *_generated()]):
        directory = tmp_path / f"{field}-{artifact_format}-{index}"
        directory.mkdir()
        verdict, detail = _attempt(directory, artifact_format, **{field: name})
        if verdict == "broken":
            broken.append(f"{field}={name[:30]!r}: {detail}")

    assert not broken, "names the config accepted but the writer could not write:\n" + "\n".join(broken)
