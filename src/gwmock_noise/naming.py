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

"""What a detector or channel name may contain, in one place.

The rule exists because these names become parts of artifacts: a channel is an HDF5 dataset path, where
`/` opens a group, and a detector becomes a file name, where `:` opens an alternate data stream on NTFS
and `/` opens a directory anywhere.

**It lives here rather than in the config models because two layers enforce it.** The config rejects a
bad name when it is built; the simulator rejects one that reached it anyway, since `model_construct`
skips validators and this repo's own tests use it. Those two drifted twice while the rule was still
being written -- first when it was written out twice (the config checked three characters, the writer
two, so a bypassed detector still produced a colon in a file name), then when the surviving copy sat in
the HDF5 branch alone and the numpy and frame writers kept their bypass. One module, imported by both
layers and applied once for every format, is what makes "the simulator re-asserts the rule" mean the
same rule everywhere.

The detector rule is universal because every format names its artifact and its JSON sidecar after the
detector. The channel rule applies only to the formats that carry a channel: an `npy` artifact is a bare
array, and rejecting its channel would refuse configurations that never use the name.
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
            the message. A prefix takes the detector rule: it is a file-name component and nothing else.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value contains a character the artifact cannot carry.
        KeyError: If *field* is not one of the three known kinds.
    """
    # Looked up rather than defaulted. `UNSAFE_FOR_CHANNEL if field == "channel" else UNSAFE_FOR_DETECTOR`
    # gives a misspelled field the detector rule in silence, which is how a caller ends up believing a
    # name was checked under a rule that never ran.
    forbidden = _RULES[field]
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
