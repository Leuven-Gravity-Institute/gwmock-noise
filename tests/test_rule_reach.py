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

"""Every name rule, at every layer that can be reached.

The last two defects on this branch were not wrong rules. They were rules with too little *reach*: the
prefix rule ran for one format, the repeat rule ran in the config only. In both cases the rule itself was
right, the demonstrated case was fixed, and a reviewer then found the same rule missing from a layer.
`test_name_fuzz.py` cannot see that -- it drives the validated path, so a rule the config enforces looks
enforced.

So this is a matrix rather than a list: each bad name is pushed through the config *and* through a
bypassed config into the simulator, for **every output format**, and the writers must refuse it in both.
A rule that only the config enforces fails here, which is what neither the unit tests nor the fuzz could
say.

The format axis was missing when this file was first written, and a reviewer said so: the docstring
claimed every layer while the rows drove `hdf5` alone. That mattered -- the defect found in the same round
was `gwf`-only, where the simulator composed no names at all and created the output directory before the
frame writer could object.

Adding a rule means adding a row. If a row cannot be written for some layer, that is worth knowing too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gwmock_noise import NoiseConfig, OutputConfig
from gwmock_noise.simulators.default import DefaultNoiseSimulator

pytestmark = pytest.mark.unit

#: Each row: the bad value, and the layers that must refuse it.
#:
#: Not every rule belongs at every layer, and writing the matrix out is what made that precise:
#:
#: * ``config`` only -- the defect cannot survive resolution, so the writer never sees it. An empty
#:   ``channel`` resolves to ``H1:`` and a ``channel`` of ``.`` to ``H1:.``, both of which are valid HDF5
#:   dataset names that write without complaint (measured). The config refuses them because a caller who
#:   wrote them meant something else, not because an artifact would break.
#: * ``simulator`` only -- the rule needs the *composed* name, which the config cannot build: it does not
#:   know that HDF5 adds an epoch and a duration while ``npy`` adds neither. Length and collision live
#:   there for that reason.
#: * both -- the value survives into the composed name unchanged, so the config's refusal is a
#:   convenience and the writer's is the guarantee. This is the majority, and it is where the last two
#:   defects were: a rule present in one and missing from the other.
_BAD_NAMES = [
    ("detector with a slash", ["H1/A"], {}, "path syntax", {"config", "simulator"}),
    ("detector with a colon", ["H1:A"], {}, "path syntax", {"config", "simulator"}),
    ("empty detector", [""], {}, "empty", {"config", "simulator"}),
    ("detector with a NUL", ["H\x001"], {}, "NUL", {"config", "simulator"}),
    ("over-long detector", ["a" * 300], {}, "over the 255-byte limit", {"simulator"}),
    ("repeated detector", ["H1", "H1"], {}, "more than once", {"config", "simulator"}),
    ("detectors colliding on case", ["H1", "h1"], {}, "collide", {"simulator"}),
    ("prefix with a slash", ["H1"], {"prefix": "sub/run"}, "path syntax", {"config", "simulator"}),
    ("prefix with a NUL", ["H1"], {"prefix": "a\x00b"}, "NUL", {"config", "simulator"}),
    ("over-long prefix", ["H1"], {"prefix": "p" * 300}, "over the 255-byte limit", {"simulator"}),
    ("channel with a slash", ["H1"], {"channel": "MOCK/NOISE"}, "group separator", {"config", "simulator"}),
    ("empty channel", ["H1"], {"channel": ""}, "empty", {"config"}),
    ("channel named dot", ["H1"], {"channel": "."}, "current group", {"config"}),
    ("channel with a NUL", ["H1"], {"channel": "a\x00b"}, "NUL", {"config", "simulator"}),
    (
        "override channel with a slash",
        ["H1"],
        {"channels": {"H1": "H1:A/B"}},
        "group separator",
        {"config", "simulator"},
    ),
]


def _settings(directory: Path, overrides: dict[str, object], artifact_format: str = "hdf5") -> dict[str, object]:
    settings: dict[str, object] = {
        "directory": directory,
        "format": artifact_format,
        "prefix": "noise",
        "gps_start": 0.0,
        "channel": "MOCK_NOISE",
        "channels": None,
    }
    settings.update(overrides)
    return settings


@pytest.mark.parametrize(("label", "detectors", "overrides", "expected", "layers"), _BAD_NAMES, ids=str)
def test_the_config_refuses_it(
    label: str, detectors: list[str], overrides: dict[str, object], expected: str, layers: set[str]
) -> None:
    """Layer one: the name never gets built."""
    if "config" not in layers:
        pytest.skip("needs a composed name, which the config cannot build")

    with pytest.raises(ValueError, match=expected):
        NoiseConfig(
            detectors=detectors,
            duration=1.0,
            sampling_frequency=4.0,
            seed=1,
            components=["white"],
            output=OutputConfig(**_settings(Path("."), overrides)),
        )


@pytest.mark.parametrize("artifact_format", ["npy", "gwf", "hdf5"])
@pytest.mark.parametrize(("label", "detectors", "overrides", "expected", "layers"), _BAD_NAMES, ids=str)
def test_the_simulator_refuses_it_when_validation_was_skipped(  # noqa: PLR0913, PLR0917
    tmp_path: Path,
    label: str,
    detectors: list[str],
    overrides: dict[str, object],
    expected: str,
    layers: set[str],
    artifact_format: str,
) -> None:
    """Layer two, which is where reach goes missing.

    `model_construct` skips validators, and this repo's own tests use it, so the writers cannot rely on
    the config having run. The assertion on the directory matters as much as the exception: a refusal that
    leaves an artifact behind is the failure this branch has fixed five times.
    """
    if "simulator" not in layers:
        pytest.skip("the defect does not survive channel resolution, so the writer never sees it")
    if artifact_format == "npy" and ("channel" in overrides or "channels" in overrides):
        pytest.skip("npy carries no channel, so its channel names are deliberately unchecked")

    target = tmp_path / "not-created-yet"
    config = NoiseConfig.model_construct(
        detectors=detectors,
        duration=1.0,
        sampling_frequency=4.0,
        seed=1,
        components=[],
        output=OutputConfig.model_construct(**_settings(target, overrides, artifact_format)),
    )

    with pytest.raises(ValueError, match=expected):
        DefaultNoiseSimulator().run(config)

    assert not target.exists(), "a refused run must not create the output directory"
