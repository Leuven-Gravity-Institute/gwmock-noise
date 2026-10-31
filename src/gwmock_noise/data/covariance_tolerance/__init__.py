"""Evaluated covariance errors of the crossfade, from a published data release.

``crossfade_errors.json`` holds, for each bundled detector curve the release evaluated, the
total covariance error of :class:`~gwmock_noise.simulators.ColoredNoiseSimulator`'s crossfade
at each evaluated window and sampling frequency, and whether that evaluation converged. The
errors were evaluated against the paper's continued target (held flat below 5 Hz, and between
5 Hz and a first tabulated point above it continued by a power law), over a fixed band per
curve; the table records both. It is what
``ColoredNoiseSimulator(covariance_tolerance=...)`` chooses a window from.

Source: I. C. F. Wong, "Choosing a generator for streaming Gaussian noise in
gravitational-wave mock data: accuracy, cost and reproducibility", data release
doi:10.5281/zenodo.23243286.

Regenerate it with ``scripts/build_covariance_tolerance_table.py --write``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

#: File name of the bundled table.
CROSSFADE_ERRORS = "crossfade_errors.json"


def load_crossfade_errors() -> dict[str, Any]:
    """Return the bundled crossfade covariance-error table.

    Returns:
        The table in the shape ``scripts/build_covariance_tolerance_table.py`` writes.
    """
    payload = resources.files(__name__).joinpath(CROSSFADE_ERRORS).read_text(encoding="utf-8")
    return json.loads(payload)
