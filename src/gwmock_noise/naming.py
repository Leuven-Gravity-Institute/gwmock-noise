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

from collections.abc import Iterable, Mapping

#: Characters a channel cannot contain. `/` is an HDF5 group separator, so a channel carrying one is
#: written into a nested group rather than the dataset a reader looks for -- silently, since the file is
#: created and its path returned. `\` is the same story on a Windows path.
UNSAFE_FOR_CHANNEL = ("/", "\\")

#: Characters a detector cannot contain. A superset: a detector becomes a file name, so `:` matters too
#: -- on NTFS it opens an alternate data stream, and the artifact would not exist as a file at all.
UNSAFE_FOR_DETECTOR = ("/", "\\", ":")

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
    # Deliberately NOT extended to control characters generally. Newline, tab, CR, DEL and bell were all
    # measured: every one round-trips through HDF5 and through a `.npy` file name. They look worse than
    # they behave, and refusing them would be the over-rejection this rule has had to walk back twice.
    if "\x00" in value:
        raise ValueError(
            f"{field} {value!r} contains a NUL byte, which cannot appear in an HDF5 name or a file path. Remove it."
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
        f"appear in an artifact name: '/' is an HDF5 group separator, and '\\' and ':' are path syntax. "
        f"Rename it, or the artifact would be written somewhere other than where it is reported."
    )


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
