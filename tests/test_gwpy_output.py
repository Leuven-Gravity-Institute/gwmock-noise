"""Tests for the optional GWpy output adapter."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.output import GWpyAdapter


class FixedNoiseSimulator:
    """Minimal simulator that returns deterministic detector arrays."""

    def __init__(self) -> None:
        """Set protocol-compatible state."""
        self.duration = 0.0
        self.sampling_frequency = 0.0
        self.detectors: list[str] = []
        self.seed: int | None = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return a fixed ramp for each detector."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        sample_count = int(duration * sampling_frequency)
        return {
            detector: np.linspace(index, index + sample_count - 1, sample_count, dtype=float)
            for index, detector in enumerate(detectors)
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose placeholder metadata."""
        return {"implementation": "fixed"}


def test_gwpy_adapter_is_importable_from_top_level_package() -> None:
    """GWpyAdapter is re-exported lazily from the top-level package."""
    assert gwmock_noise.GWpyAdapter is GWpyAdapter


def test_gwpy_adapter_raises_clear_error_when_gwpy_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instantiating the adapter without gwpy raises a helpful error."""
    gwpy_output = import_module("gwmock_noise.output.gwpy")
    original_import_module = gwpy_output.import_module

    def fake_import_module(name: str):
        if name == "gwpy.timeseries":
            raise ImportError("No module named 'gwpy'")
        return original_import_module(name)

    monkeypatch.setattr(gwpy_output, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"pip install gwmock-noise\[gwpy\]"):
        GWpyAdapter(FixedNoiseSimulator())


def test_gwpy_adapter_wraps_timeseries_when_gwpy_is_available() -> None:
    """The adapter returns gwpy.TimeSeries with the expected metadata."""
    gwpy = pytest.importorskip("gwpy")
    adapter = GWpyAdapter(FixedNoiseSimulator(), gps_start=100.5)

    first = adapter.generate(duration=2.0, sampling_frequency=4.0, detectors=["H1", "L1"])
    second = adapter.generate(duration=2.0, sampling_frequency=4.0, detectors=["H1"])

    assert isinstance(first["H1"], gwpy.timeseries.TimeSeries)
    assert float(first["H1"].t0.value) == pytest.approx(100.5)
    assert float(first["H1"].sample_rate.value) == pytest.approx(4.0)
    assert first["H1"].channel.name == "H1"
    assert np.allclose(first["H1"].value, np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))
    assert float(second["H1"].t0.value) == pytest.approx(102.5)
    assert adapter.gps_start == pytest.approx(104.5)
