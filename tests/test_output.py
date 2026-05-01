"""Tests for optional output adapters."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from gwmock_noise import output
from gwmock_noise.output.gwpy import GWpyAdapter


def test_output_module_rejects_unknown_export() -> None:
    """Unknown lazy exports raise AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = output.DoesNotExist


def test_gwpy_adapter_metadata_includes_adapter_fields() -> None:
    """GWpyAdapter metadata overlays adapter-specific fields."""

    class DummyBase:
        duration = 1.0
        sampling_frequency = 16.0
        detectors: ClassVar[list[str]] = ["H1"]
        seed = 1
        metadata: ClassVar[dict[str, Any]] = {"implementation": "dummy"}

        def generate(self, duration: float, sampling_frequency: float, detectors: list[str], seed: int | None = None):
            _ = (duration, sampling_frequency, detectors, seed)
            return {"H1": [0.0]}

    base = DummyBase()
    adapter = GWpyAdapter.__new__(GWpyAdapter)
    adapter.base = base
    adapter.gps_start = 123.0
    assert adapter.metadata["output_adapter"] == "gwpy"
    assert adapter.metadata["gps_start"] == 123.0
