"""Tests for GWOSC data fetching."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
import pytest

from gwmock_noise.gwosc.filters import GwoscSegmentFilter
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
        self.name = name
        self._data = data
        self._t0 = t0
        self._sample_rate = sample_rate

    def crop(self, start: float, end: float) -> FakeTimeSeries:
        """Fake crop that returns a slice of the data."""
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


class FakeTimeSeriesModule:
    """Fake gwpy.timeseries module."""

    TimeSeries = FakeTimeSeries


def _make_fake_gwpy_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch import_module to return a fake gwpy.timeseries module."""
    fetcher_mod = import_module("gwmock_noise.gwosc.fetcher")

    def fake_import(name: str) -> Any:
        if name == "gwpy.timeseries":
            return FakeTimeSeriesModule()
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(fetcher_mod, "import_module", fake_import)


class TestGwoscNoiseFetcher:
    """Tests for GwoscNoiseFetcher."""

    def test_fetch_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_raw returns TimeSeries for all detectors."""
        _make_fake_gwpy_module(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(
            gps_start=1261875618,
            gps_end=1261877618,
            detectors=["H1", "L1"],
        )
        fetcher = GwoscNoiseFetcher(config)
        result = fetcher.fetch_raw()

        assert "H1" in result
        assert "L1" in result
        assert isinstance(result["H1"], FakeTimeSeries)
        assert result["H1"].name == "H1"

    def test_fetch_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_clean returns cropped clean segments."""
        _make_fake_gwpy_module(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        # Use a filter config with no filters, so the full interval is clean
        config = GwoscNoiseConfig(
            gps_start=1261875618,
            gps_end=1261877618,
            detectors=["H1"],
            filters=GwoscFilterConfig(filter_types=[]),
        )
        fetcher = GwoscNoiseFetcher(config)
        result = fetcher.fetch_clean()

        assert "H1" in result
        assert len(result["H1"]) == 1
        assert isinstance(result["H1"][0], FakeTimeSeries)

    def test_clean_segments_property(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """clean_segments property returns computed segments."""
        _make_fake_gwpy_module(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            detectors=["H1"],
            filters=GwoscFilterConfig(filter_types=[]),
        )
        fetcher = GwoscNoiseFetcher(config)
        segments = fetcher.clean_segments

        assert "H1" in segments
        assert segments["H1"] == [(0.0, 100.0)]

    def test_fetch_clean_no_segments_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """fetch_clean raises when no clean segments are found."""
        _make_fake_gwpy_module(monkeypatch)

        # We need to make the segment filter return no clean segments.
        # Monkeypatch compute_clean_segments on the instance.
        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            detectors=["H1"],
            filters=GwoscFilterConfig(filter_types=[]),
        )
        fetcher = GwoscNoiseFetcher(config)

        # Override the segment filter's compute_clean_segments
        fetcher._segment_filter.compute_clean_segments = lambda *a, **kw: {"H1": []}

        with pytest.raises(ValueError, match="No clean segments found"):
            fetcher.fetch_clean()

    def test_import_error_when_gwpy_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear ImportError when gwpy is not installed."""
        fetcher_mod = import_module("gwmock_noise.gwosc.fetcher")

        def fake_import(name: str) -> None:
            raise ImportError("No module named 'gwpy'")

        monkeypatch.setattr(fetcher_mod, "import_module", fake_import)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(gps_start=0.0, gps_end=100.0)
        with pytest.raises(ImportError, match="pip install gwmock-noise\\[gwpy\\]"):
            GwoscNoiseFetcher(config)

    def test_lazy_export_from_top_level(self) -> None:
        """GwoscNoiseFetcher is exportable from the top-level package."""
        import gwmock_noise
        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        assert gwmock_noise.GwoscNoiseFetcher is GwoscNoiseFetcher

    def test_lazy_export_gwosc_models(self) -> None:
        """GwoscNoiseConfig is exportable from the top-level package."""
        import gwmock_noise
        from gwmock_noise.gwosc.models import GwoscNoiseConfig

        assert gwmock_noise.GwoscNoiseConfig is GwoscNoiseConfig

    def test_lazy_export_gwosc_filter_config(self) -> None:
        """GwoscFilterConfig is exportable from the top-level package."""
        import gwmock_noise
        from gwmock_noise.gwosc.models import GwoscFilterConfig

        assert gwmock_noise.GwoscFilterConfig is GwoscFilterConfig

    def test_lazy_export_filter_type(self) -> None:
        """FilterType is exportable from the top-level package."""
        import gwmock_noise

        assert gwmock_noise.FilterType is FilterType

    def test_lazy_export_segment_filter(self) -> None:
        """GwoscSegmentFilter is exportable from the top-level package."""
        import gwmock_noise

        assert gwmock_noise.GwoscSegmentFilter is GwoscSegmentFilter
