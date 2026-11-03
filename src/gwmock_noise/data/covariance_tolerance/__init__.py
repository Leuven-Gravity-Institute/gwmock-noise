"""Evaluated covariance errors of the crossfade, from a published data release and merged evaluations.

``crossfade_errors.json`` holds, for each bundled detector curve the release evaluated, the
total covariance error of :class:`~gwmock_noise.simulators.ColoredNoiseSimulator`'s crossfade
at each evaluated window and sampling frequency, whether that evaluation converged, and where
it comes from: the data release, or a ``gwmock-noise evaluate-covariance-error`` result
committed under ``evaluations/`` (see :mod:`gwmock_noise.data.covariance_tolerance._merge`). The
errors were evaluated against the paper's continued target (held flat below 5 Hz, and between
5 Hz and a first tabulated point above it continued by a power law), over a fixed band per
curve; the table records both. It is what
``ColoredNoiseSimulator(covariance_tolerance=...)`` chooses a window from.

Source: I. C. F. Wong, "Choosing a generator for streaming Gaussian noise in
gravitational-wave mock data: accuracy, cost and reproducibility", arXiv:2611.01624;
data release doi:10.5281/zenodo.23243286.

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


def describe_source(table: dict[str, Any], source: str) -> str:
    """Return a citation for a row's source: the release's DOI, or the evaluation file and version."""
    entry = table["sources"][source]
    if entry["kind"] == "release":
        return f"doi:{entry['doi']}"
    return f"gwmock-noise {entry['gwmock_noise_version']} evaluation {entry['file']} (SHA-256 {entry['sha256']})"
