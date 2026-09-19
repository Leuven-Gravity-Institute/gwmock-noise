"""Measured reference statistics the registered glitch populations are anchored to.

``o3_reference_summary.json`` is the small summary of a measurement over two published data
products -- the Gravity Spy machine-learning classifications of LIGO glitches and the
GWOSC O3 observing segments. It ships with the package so a user can see what the
registered rates and amplitude laws were derived from without re-running the measurement,
and so the tests can assert the registered constants against it.

Regenerate it with ``scripts/measure_glitch_population_anchors.py --write-summary``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

#: File name of the bundled summary.
O3_REFERENCE_SUMMARY = "o3_reference_summary.json"


def load_o3_reference_summary() -> dict[str, Any]:
    """Return the bundled O3 reference summary.

    Returns:
        The measured counts, observing times, rates and distribution statistics, in the
        shape the measurement harness writes.
    """
    payload = resources.files(__name__).joinpath(O3_REFERENCE_SUMMARY).read_text(encoding="utf-8")
    return json.loads(payload)
