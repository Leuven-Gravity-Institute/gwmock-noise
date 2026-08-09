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
bad name when it is built; the writer rejects one that reached it anyway, since `model_construct` skips
validators and this repo's own tests use it. Those two drifted almost immediately when the rule was
written twice: the config checked detectors and channels for three characters, and the writer checked
channels for two, so a bypassed detector still produced a colon in a file name. A reviewer demonstrated
it. One module, imported by both, is what makes "the writer re-asserts the rule" mean the same rule.
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


def reject_unsafe(value: str, *, field: str) -> str:
    """Return *value* unchanged, or raise if it cannot survive becoming part of an artifact.

    Args:
        value: The detector or channel name.
        field: ``"detector"`` or ``"channel"``; selects the rule and names the field in the message.

    Returns:
        The value, unchanged.

    Raises:
        ValueError: If the value contains a character the artifact cannot carry.
    """
    forbidden = UNSAFE_FOR_CHANNEL if field == "channel" else UNSAFE_FOR_DETECTOR
    found = [character for character in forbidden if character in value]
    if not found:
        return value
    raise ValueError(
        f"{field} {value!r} contains {', '.join(repr(character) for character in found)}, which cannot "
        f"appear in an artifact name: '/' is an HDF5 group separator, and '\\' and ':' are path syntax. "
        f"Rename it, or the artifact would be written somewhere other than where it is reported."
    )


def check_artifact_names(*, detectors: Iterable[str], channels: Mapping[str, str]) -> None:
    """Check every name a run is about to use, before anything is generated or written.

    Checked up front rather than per detector as the writing proceeds. Writing detector by detector meant
    a run with one bad name wrote the good detectors' files and then raised, leaving a partial set on
    disk -- and doing the whole simulation first, only to refuse afterwards. Both were demonstrated by a
    reviewer; the second is merely wasteful, the first leaves the caller with output they did not get to
    keep or discard as a whole.

    Args:
        detectors: The detectors this run will write.
        channels: The resolved channel for each detector.

    Raises:
        ValueError: If any name cannot survive becoming part of an artifact.
    """
    for detector in detectors:
        reject_unsafe(detector, field="detector")
    for channel in channels.values():
        reject_unsafe(channel, field="channel")
