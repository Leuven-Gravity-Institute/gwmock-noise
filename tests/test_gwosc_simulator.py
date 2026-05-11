"""Tests for GwoscNoiseSimulator."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.gwosc.models import FilterType, GwoscFilterConfig, GwoscNoiseConfig


class FakeTimeSeries:
    """Fake gwpy.TimeSeries for testing."""

    def __init__(
        self,
        data: np.ndarray,
        *,
        t0: float = 0.0,
        sample_rate: float = 1.0,
        channel: str = "",
        name: str = "",
    ) -> None:
        """Initialize with fake data."""
        self.value = data
        self.t0 = t0
        self.sample_rate = sample_rate
        self.channel = channel
        self.name = name or channel
        self._data = data
        self._t0 = t0
        self._sample_rate = sample_rate

    def crop(self, start: float, end: float) -> FakeTimeSeries:
        """Fake crop."""
        start_idx = int((start - self._t0) * self._sample_rate)
        end_idx = int((end - self._t0) * self._sample_rate)
        start_idx = max(0, start_idx)
        end_idx = min(len(self._data), end_idx)
        if end_idx <= start_idx:
            raise ValueError("no data in range")
        return FakeTimeSeries(
            self._data[start_idx:end_idx],
            t0=start,
            sample_rate=self._sample_rate,
            channel=self.channel,
            name=self.name,
        )

    def append(self, other: FakeTimeSeries) -> FakeTimeSeries:
        """Fake append."""
        return FakeTimeSeries(
            np.concatenate([self._data, other._data]),
            t0=self._t0,
            sample_rate=self._sample_rate,
            channel=self.channel,
            name=self.name,
        )

    @staticmethod
    def fetch_open_data(  # noqa: PLR0913
        detector: str,
        start: float,
        end: float,
        sample_rate: float = 4096.0,
        host: str = "https://gwosc.org",
        cache: bool = False,
    ) -> FakeTimeSeries:
        """Fake fetch_open_data."""
        duration = end - start
        n_samples = int(duration * sample_rate)
        data = np.arange(float(n_samples))
        return FakeTimeSeries(data, t0=start, sample_rate=sample_rate, name=detector)

    @staticmethod
    def read(source: str, format: str = "") -> FakeTimeSeries:  # noqa: A002
        """Fake read from file."""
        data = np.arange(1000.0)
        return FakeTimeSeries(data, t0=0.0, sample_rate=4096.0, name="cached")


class FakeTimeSeriesModule:
    """Fake gwpy.timeseries module."""

    TimeSeries = FakeTimeSeries


class FakeGwoscLocateModule:
    """Fake gwosc.locate module."""

    @staticmethod
    def get_urls(
        detector: str,
        start: int,
        end: int,
        sample_rate: int = 4096,
        host: str = "https://gwosc.org",
    ) -> list[str]:
        """Return fake URLs."""
        return [
            f"https://gwosc.org/archive/data/O3b_4KHZ_R1/{start}/{detector[0]}-{detector}_GWOSC_O3b_4KHZ_R1-{start}-4096.hdf5"
        ]


def _patch_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch import_module in both fetcher and simulator modules."""
    fetcher_mod = import_module("gwmock_noise.gwosc.fetcher")

    def fake_import(name: str) -> Any:
        if name == "gwpy.timeseries":
            return FakeTimeSeriesModule()
        if name == "gwosc.locate":
            return FakeGwoscLocateModule()
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(fetcher_mod, "import_module", fake_import)

    # Also patch urlretrieve
    def fake_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_text("fake hdf5 content")

    monkeypatch.setattr(fetcher_mod, "urlretrieve", fake_urlretrieve)


class TestGwoscNoiseSimulator:
    """Tests for GwoscNoiseSimulator."""

    def test_protocol_attributes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Simulator exposes NoiseSimulator protocol attributes."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=100.0,
            gps_end=200.0,
            detectors=["H1", "L1"],
            sample_rate=4096.0,
        )
        sim = GwoscNoiseSimulator(config)

        assert sim.duration == 100.0
        assert sim.sampling_frequency == 4096.0
        assert sim.detectors == ["H1", "L1"]
        assert sim.seed is None

    def test_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Metadata reflects the gwosc configuration."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=100.0,
            gps_end=200.0,
            detectors=["H1"],
            sample_rate=4096.0,
            filters=GwoscFilterConfig(
                filter_types=[FilterType.HIGH_CONFIDENCE_GW],
                far_threshold=1.0,
            ),
        )
        sim = GwoscNoiseSimulator(config)
        meta = sim.metadata

        assert meta["implementation"] == "gwosc_real_noise"
        assert meta["gps_start"] == 100.0
        assert meta["gps_end"] == 200.0
        assert meta["sample_rate"] == 4096.0
        assert meta["detectors"] == ["H1"]
        assert meta["filters"]["far_threshold"] == 1.0

    def test_metadata_with_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Metadata includes cache_dir when configured."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            cache_dir=tmp_path / "gwosc_cache",
        )
        sim = GwoscNoiseSimulator(config)
        meta = sim.metadata

        assert meta["cache_dir"] == str(tmp_path / "gwosc_cache")

    def test_metadata_without_cache_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Metadata shows None for cache_dir when not configured."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(gps_start=0.0, gps_end=100.0)
        sim = GwoscNoiseSimulator(config)

        assert sim.metadata["cache_dir"] is None

    def test_generate_returns_arrays(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generate returns numpy arrays for each detector."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=100.0,
            gps_end=104.0,
            detectors=["H1"],
            sample_rate=4096.0,
            filters=GwoscFilterConfig(filter_types=[]),
        )
        sim = GwoscNoiseSimulator(config)
        result = sim.generate(
            duration=4.0,
            sampling_frequency=4096.0,
            detectors=["H1"],
        )

        assert "H1" in result
        assert isinstance(result["H1"], np.ndarray)
        assert len(result["H1"]) > 0

    def test_generate_mismatched_sampling_frequency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generate raises if sampling_frequency doesn't match config."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            sample_rate=4096.0,
        )
        sim = GwoscNoiseSimulator(config)

        with pytest.raises(ValueError, match="does not match"):
            sim.generate(duration=100.0, sampling_frequency=2048.0, detectors=["H1"])

    def test_generate_non_subset_detectors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generate raises if requested detectors are not a subset."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            detectors=["H1"],
        )
        sim = GwoscNoiseSimulator(config)

        with pytest.raises(ValueError, match="not a subset"):
            sim.generate(duration=100.0, sampling_frequency=4096.0, detectors=["L1"])

    def test_generate_stream(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_stream yields chunks of the requested duration."""
        _patch_imports(monkeypatch)

        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        config = GwoscNoiseConfig(
            gps_start=100.0,
            gps_end=104.0,
            detectors=["H1"],
            sample_rate=4096.0,
            filters=GwoscFilterConfig(filter_types=[]),
        )
        sim = GwoscNoiseSimulator(config)
        chunks = list(
            sim.generate_stream(
                chunk_duration=2.0,
                sampling_frequency=4096.0,
                detectors=["H1"],
            )
        )

        assert len(chunks) == 2
        for chunk in chunks:
            assert "H1" in chunk
            assert isinstance(chunk["H1"], np.ndarray)
            assert len(chunk["H1"]) == int(2.0 * 4096.0)

    def test_lazy_export_from_top_level(self) -> None:
        """GwoscNoiseSimulator is lazily exportable from the top-level package."""
        import gwmock_noise
        from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator

        assert gwmock_noise.GwoscNoiseSimulator is GwoscNoiseSimulator
