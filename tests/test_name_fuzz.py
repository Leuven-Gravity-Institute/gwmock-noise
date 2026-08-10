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
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import NoiseConfig, OutputConfig
from gwmock_noise.simulators.default import DefaultNoiseSimulator

pytestmark = pytest.mark.unit

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


def _attempt(directory: Path, artifact_format: str, **name: str) -> tuple[str, str]:  # noqa: PLR0911
    """Return ``("accepted", "")``, ``("rejected", reason)``, or ``("broken", what went wrong)``.

    One early return per way an artifact can be wrong. Collapsing them into a single exit would mean
    losing which check failed, and the detail is what makes a fuzz failure actionable rather than a
    report that something, somewhere, is broken.
    """
    detector = name.get("detector", "H1")
    settings = {
        "directory": directory,
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
        result = DefaultNoiseSimulator().run(config)
    except ValueError as error:
        # A refusal is allowed -- but it must be a refusal, not a partial write with an exception on top.
        written = sorted(path.name for path in directory.rglob("*") if path.is_file())
        if written:
            return "broken", f"raised {str(error).splitlines()[0][:40]!r} yet left {written}"
        return "rejected", str(error).splitlines()[0][:60]
    except Exception as error:  # noqa: BLE001
        written = sorted(path.name for path in directory.rglob("*") if path.is_file())
        return "broken", f"{type(error).__name__}: {str(error).splitlines()[0][:40]} left={written}"

    path = result.output_paths[detector]
    if directory.resolve() not in path.resolve().parents:
        return "broken", f"escaped the output directory: {path}"
    if not path.exists():
        return "broken", f"reported a path that was never written: {path}"
    try:
        if artifact_format == "npy":
            np.load(path)
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


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5"])
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
        directory = tmp_path / f"{artifact_format}-{index}"
        directory.mkdir()
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
            path = DefaultNoiseSimulator().run(config).output_paths[detector]
        except ValueError:
            leftover = sorted(p.name for p in directory.iterdir())
            if leftover:
                broken.append(f"length {length}: refused yet left {leftover}")
            continue
        except OSError as error:
            broken.append(f"length {length}: OSError rather than a refusal ({error.errno})")
            continue
        if len(path.name.encode("utf-8")) > 255:
            broken.append(f"length {length}: wrote a {len(path.name.encode())}-byte name")

    assert not broken, f"at gps_start={gps_start}, duration={duration}:\n" + "\n".join(broken)


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5"])
@pytest.mark.parametrize(("first", "second"), _COLLIDING_PAIRS)
def test_two_names_never_share_one_file(tmp_path: Path, artifact_format: str, first: str, second: str) -> None:
    """Distinct detectors must not end up writing the same artifact.

    The round-14 defect: the config accepted `["H1", "h1"]`, reported two paths, and wrote one file. The
    assertion is on the filesystem, not on the exception -- a refusal is fine, and so is writing two
    genuinely distinct files, but reporting two paths while one artifact exists is not.
    """
    try:
        config = NoiseConfig(
            detectors=[first, second],
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(directory=tmp_path, format=artifact_format, prefix="noise", gps_start=0.0),
        )
        result = DefaultNoiseSimulator().run(config)
    except ValueError:
        assert list(tmp_path.iterdir()) == [], "a refusal must not leave an artifact behind"
        return

    reported = {str(path) for path in result.output_paths.values()}
    suffix = ".npy" if artifact_format == "npy" else ".hdf5"
    written = [path for path in tmp_path.iterdir() if path.suffix == suffix]
    assert len(written) == len(reported), (
        f"reported {len(reported)} path(s) for {first!r} and {second!r} but {len(written)} file(s) exist"
    )


@pytest.mark.parametrize("artifact_format", ["npy", "hdf5"])
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
