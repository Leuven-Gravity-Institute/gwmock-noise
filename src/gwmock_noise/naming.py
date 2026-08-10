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

"""What a detector, channel, or prefix may contain, in one place.

The rule exists because these names become parts of artifacts: a channel is an HDF5 dataset path, where
`/` opens a group, and a detector becomes a file name, where `:` opens an alternate data stream on NTFS
and `/` opens a directory anywhere.

**It lives here rather than in the config models because three modules enforce it.** The config rejects
a bad name when it is built; the simulator rejects one that reached it anyway, since `model_construct`
skips validators and this repo's own tests use it; and `FrameWriter`, which is public API, rejects the
names a caller hands it directly, having gone through no config at all.

They drifted twice while the rule was still being written -- first when it was written out twice (the
config checked three characters, the writer two, so a bypassed detector still produced a colon in a file
name), then when the surviving copy sat in the HDF5 branch alone and the numpy and frame writers kept
their bypass. One module, imported by all three and applied once for every format, is what makes "the
rule is re-asserted" mean the same rule everywhere.

The detector and prefix rules are universal because every format names its artifact and its JSON sidecar
from both. The channel rule applies only to the formats that carry a channel: an `npy` artifact is a bare
array, and rejecting its channel would refuse configurations that never use the name.

Emptiness is part of the rule, not a separate concern: a rule written only as a character test passes the
empty string, which is not a name.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Mapping

#: Characters Windows reserves in a file name, minus `:`, which the channel needs for `IFO:name` and the
#: detector rule already forbids outright. Rejected on **every** platform, not only Windows: a config is
#: written on one machine and run on another, and a rule that consulted the host would make the same
#: configuration valid in one place and invalid in the next -- the same reason `reject_colliding_names`
#: folds case rather than asking the filesystem. The price is refusing a `*` in a channel on Linux, where
#: it would have worked.
#:
#: Measured, not guessed: the first CI run on `windows-latest` failed on every one of these, in `npy`,
#: `hdf5` and `gwf` alike, with `OSError [Errno 22] Invalid argument`.
WINDOWS_RESERVED = ("<", ">", '"', "|", "?", "*")

#: Characters a channel cannot contain. `/` is an HDF5 group separator, so a channel carrying one is
#: written into a nested group rather than the dataset a reader looks for -- silently, since the file is
#: created and its path returned. `\` is the same story on a Windows path. The Windows set belongs here
#: too, because a channel reaches a *file* name: a frame is `H-H1_MOCK_NOISE_100-2.gwf`.
#:
#: `:` is deliberately absent. A resolved channel is `IFO:name` by convention, and `compose_frame_name`
#: drops that prefix before the channel enters a file name. A *second* colon is refused separately, since
#: only the first one is dropped and the rest would survive into the name.
UNSAFE_FOR_CHANNEL = ("/", "\\", *WINDOWS_RESERVED)

#: Characters a detector cannot contain. A superset: a detector becomes a file name, so `:` matters too
#: -- on NTFS it opens an alternate data stream, and the artifact would not exist as a file at all.
UNSAFE_FOR_DETECTOR = ("/", "\\", ":", *WINDOWS_RESERVED)

#: Which rule each kind of name takes. A prefix is a file-name component like a detector, so it takes the
#: detector rule; it is never written inside an artifact, which is what distinguishes it from a channel.
_RULES = {
    "channel": UNSAFE_FOR_CHANNEL,
    "detector": UNSAFE_FOR_DETECTOR,
    "prefix": UNSAFE_FOR_DETECTOR,
}


def reject_unsafe(value: str, *, field: str) -> str:
    """Return *value* unchanged, or raise if it cannot survive becoming part of an artifact.

    Args:
        value: The detector, channel, or prefix.
        field: ``"detector"``, ``"channel"``, or ``"prefix"``; selects the rule and names the field in
            the message. A prefix takes the detector rule: it is a file-name component and nothing else,
            and unlike the other two it may be empty, which means "no prefix".

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value is empty when it must name something, or contains a character the
            artifact cannot carry.
        KeyError: If *field* is not one of the three known kinds.
    """
    # Looked up rather than defaulted. `UNSAFE_FOR_CHANNEL if field == "channel" else UNSAFE_FOR_DETECTOR`
    # gives a misspelled field the detector rule in silence, which is how a caller ends up believing a
    # name was checked under a rule that never ran.
    forbidden = _RULES[field]
    # An empty name contains no forbidden character, so a rule written only as a character test passes it
    # -- the same vacuous-truth trap as checking values across an empty array. A detector or channel that
    # is the empty string is not a name: `npy` wrote `noise_.npy`, HDF5 raised `IndexError` from
    # `detector[0]`, and an empty channel override raised `TypeError` from h5py *after* creating the
    # file, leaving a partial artifact behind. A reviewer found all three. The prefix is exempt because
    # empty is its default and its documented meaning: no prefix.
    if field != "prefix" and not value:
        raise ValueError(
            f"{field} is empty, which cannot name an artifact. Give it a name, or omit the field if the "
            f"caller meant to leave it unset."
        )
    # `.` is HDF5's own name for the current group, so no dataset can be created with it: h5py raises
    # "name already exists" *after* opening the file, leaving a partial artifact -- the same shape as the
    # empty-channel case, and found the same way. Only the channel is affected: as a file-name component
    # `.` merely produces `noise_..npy`, which is odd and harmless.
    #
    # `..` is deliberately NOT rejected. It looks like the same class and is not: a dataset named `..` is
    # created, round-trips through GWpy, and is addressable as both `handle[".."]` and `handle["/.."]`.
    # That was checked rather than assumed -- the expectation was that it would resolve to the parent
    # group. Refusing it would be the over-rejection this rule has already had to walk back twice.
    # NUL breaks every artifact this package writes, and it is the only control character that does.
    # HDF5 stores names as VLEN strings, which cannot embed one -- h5py raises *after* opening the file,
    # leaving a partial artifact -- and a POSIX path cannot contain one either, so `np.save` refuses it
    # too. A reviewer found it. Applies to all three fields: each becomes either a dataset name or a
    # file name, and NUL is invalid in both.
    #
    # This comment used to say the rule was deliberately NOT extended to control characters generally,
    # because newline, tab, CR, DEL and bell had been measured to round-trip through HDF5 and through a
    # `.npy` file name. That measurement was right and its conclusion was wrong: it was taken on POSIX
    # only. The first CI run on `windows-latest` refused `a\nb`, `a\tb`, `a\rb` and `\x07` outright --
    # `OSError [Errno 22] Invalid argument` -- for `npy`, `hdf5` and `gwf`. Windows reserves every
    # character below 0x20 in a file name.
    #
    # So NUL keeps its own message, because it is the one that also breaks HDF5 (VLEN strings cannot
    # embed it, and h5py raises *after* opening the file, leaving a partial artifact) and POSIX paths,
    # and the rest are refused as a class. DEL (0x7f) is left alone: it is not in the reserved range and
    # it was measured to work on all three platforms.
    if "\x00" in value:
        raise ValueError(
            f"{field} {value!r} contains a NUL byte, which cannot appear in an HDF5 name or a file path. Remove it."
        )
    control = sorted({character for character in value if character < " "})
    if control:
        raise ValueError(
            f"{field} {value!r} contains {''.join(repr(character) for character in control)}, which Windows "
            f"reserves in a file name. Remove it."
        )
    # One colon, and only in a channel. `compose_frame_name` drops everything up to the first colon as the
    # `IFO:` prefix, so a second colon would survive into a GWF file name, where NTFS reads it as an
    # alternate data stream. The rule is here rather than in the composer because the composer's job is to
    # name a file, not to judge one.
    if field == "channel" and value.count(":") > 1:
        raise ValueError(
            f"channel {value!r} carries more than one ':'. Only the leading 'IFO:' is dropped when the "
            f"channel enters a frame name, so the rest would remain in the file name. Use one."
        )
    if field == "channel" and value == ".":
        raise ValueError(
            "channel '.' is HDF5's name for the current group, so no dataset can be created with it. "
            "Rename the channel."
        )
    found = [character for character in forbidden if character in value]
    if not found:
        return value
    raise ValueError(
        f"{field} {value!r} contains {', '.join(repr(character) for character in found)}, which cannot "
        f"appear in an artifact name: '/' is an HDF5 group separator, '\\' and ':' are path syntax, and "
        f"'<', '>', '\"', '|', '?' and '*' are reserved by Windows in a file name -- refused on every "
        f"platform so the same configuration stays valid wherever it runs. Rename it, or the artifact "
        f"would be written somewhere other than where it is reported, or not at all."
    )


#: The longest a single path component may be, in bytes. APFS and ext4 stop at 255 bytes; NTFS stops at
#: 255 UTF-16 units, which is characters rather than bytes, so counting bytes over-rejects some names NTFS
#: would hold. That direction is the safe one and the limit stays a single constant. It is not universal:
#: eCryptfs stops at 143, and network filesystems vary. A reviewer checked the NTFS claim, which this
#: comment previously got wrong by lumping all three together.
MAX_NAME_BYTES = 255


def reject_overlong(name: str, *, described_as: str) -> str:
    """Return *name* unchanged, or raise if the filesystem cannot hold a component that long.

    Checked on the **composed** artifact name rather than on any one field, because that is what the
    filesystem sees: a 250-character detector is fine alone and not once a prefix, an epoch, a duration
    and a suffix are around it. Found by fuzzing names against both writers rather than by review, and it
    matters for the same reason the other name rules do: with several detectors, the over-long one failed
    *after* the earlier detectors' artifacts were written, leaving a partial set on disk and an `OSError`
    from inside h5py or numpy instead of a statement about the name.

    Args:
        name: The composed artifact name, including any prefix and suffix.
        described_as: What to call it in the message, e.g. ``"HDF5 artifact name"``.

    Returns:
        The name, unchanged.

    Raises:
        ValueError: If the encoded name exceeds :data:`MAX_NAME_BYTES`.
    """
    encoded = len(name.encode("utf-8"))
    if encoded <= MAX_NAME_BYTES:
        return name
    raise ValueError(
        f"the {described_as} would be {encoded} bytes ({name[:40]!r}...), over the {MAX_NAME_BYTES}-byte "
        f"limit for one path component. Shorten the detector name or the prefix."
    )


def reject_repeated(detectors: Iterable[str]) -> list[str]:
    """Return the detectors unchanged, or raise if one appears more than once.

    Separate from :func:`reject_colliding_names` because it must run *before* the composed names are
    collected. Every layer that collects names does so into a dict keyed by detector, so an exact repeat
    collapses to one entry and the collision check then sees nothing wrong -- the run reports one artifact
    for two requested detectors, writes one file, and raises nothing. Both reviewers found that on the
    bypass path after the config validator alone had been thought sufficient.

    Only exact repeats. `["H1", "h1"]` survives this and is caught by the collision check, which is where
    the filesystem's opinion about case belongs.

    Args:
        detectors: The detectors a run or a writer was asked for.

    Returns:
        The detectors, unchanged.

    Raises:
        ValueError: If any detector appears more than once.
    """
    collected = list(detectors)
    repeated = sorted({detector for detector in collected if collected.count(detector) > 1})
    if repeated:
        raise ValueError(
            f"detectors contains {repeated!r} more than once; each detector writes one artifact, so a "
            f"repeat would silently produce fewer files than detectors requested."
        )
    return collected


def reject_colliding_names(names: Mapping[str, str], *, described_as: str) -> None:
    """Raise if two owners would write to what the filesystem may treat as one file.

    `_hdf5_name` claimed the name was "unique by construction" because one detector writes one file. It
    is injective in the detector *as a Python string*, which is not the same as unique on disk: APFS and
    NTFS compare names case-insensitively by default, so ``detectors=["H1", "h1"]`` produced two distinct
    reported paths and one file, with one detector's samples overwriting the other's and both paths
    pointing at the survivor. A reviewer found the stale claim; the collision is the same silent data
    loss that made this writer name artifacts after the detector rather than the channel in the first
    place.

    Compared under case-folding **and** NFC normalisation, both measured on the filesystem rather than
    argued about. `H1` and `h1` collapse to one file; so do the NFC and NFD spellings of the same
    detector, where the second write wins and the first detector's samples are gone.

    Normalisation was briefly dropped from this key on the strength of a probe that appeared to show two
    files surviving. Writing the two names directly, with no package code involved, showed one file
    holding the second payload. The probe was wrong and the direct measurement is what this rests on --
    which is the reason the test alongside it asserts the collision rather than the shape of the rule.

    Both comparisons are **stricter than a case-sensitive, non-normalising filesystem needs**. On ext4 both those names work, so this refuses a configuration that would have
    run. That trade is the opposite of the ones this module has walked back twice, and for a reason: the
    earlier over-rejections refused sensible configurations to prevent a loud error, while this refuses a
    pathological one to prevent silent loss. The alternative -- probing the filesystem's collation --
    makes the behaviour depend on where the output happens to be pointed.

    Args:
        names: The composed name each owner (usually a detector) will write.
        described_as: What to call the names in the message, e.g. ``"HDF5 artifact names"``.

    Raises:
        ValueError: If two owners map onto the same name under that comparison.
    """
    seen: dict[str, str] = {}
    for owner, name in names.items():
        folded = unicodedata.normalize("NFC", name).casefold()
        if folded in seen:
            raise ValueError(
                f"{described_as} collide: {seen[folded]!r} and {owner!r} both write {name!r} on a "
                f"filesystem that ignores case, so one would silently overwrite the other. Rename one."
            )
        seen[folded] = owner


def check_artifact_names(*, detectors: Iterable[str], channels: Mapping[str, str], prefix: str = "") -> None:
    """Check every name a run is about to use, before anything is generated or written.

    Checked up front rather than per detector as the writing proceeds. Writing detector by detector meant
    a run with one bad name wrote the good detectors' files and then raised, leaving a partial set on
    disk -- and doing the whole simulation first, only to refuse afterwards. Both were demonstrated by a
    reviewer; the second is merely wasteful, the first leaves the caller with output they did not get to
    keep or discard as a whole.

    Args:
        detectors: The detectors this run will write.
        channels: The resolved channel for each detector, or empty for a format that carries no channel.
        prefix: The artifact name prefix, if the caller uses one.

    Raises:
        ValueError: If any name cannot survive becoming part of an artifact.
    """
    # The prefix is checked under the detector rule because it is a file-name component and nothing else:
    # every format prepends it, and no format writes it inside the artifact. It went unchecked for nine
    # review rounds while the detector and channel rules were built around it, and unlike those two it
    # was not even a bypass -- a validated `OutputConfig(prefix="sub/run")` wrote `sub/run_H1.npy`,
    # below the directory the caller named. Found by a reviewer.
    reject_unsafe(prefix, field="prefix")
    for detector in detectors:
        reject_unsafe(detector, field="detector")
    for channel in channels.values():
        reject_unsafe(channel, field="channel")
