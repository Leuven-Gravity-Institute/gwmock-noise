"""Tests for GWOSC data fetching."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher
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
        self.name = name or channel
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

    def append(self, other: FakeTimeSeries) -> FakeTimeSeries:
        """Fake append of two time series."""
        combined = np.concatenate([self._data, other._data])
        return FakeTimeSeries(
            combined,
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
        """Fake read from file. Extracts GPS start from URL-style filename."""
        import re

        source_str = str(source)
        # Try to extract GPS start from filename like "...-1261875618-4096.hdf5"
        match = re.search(r"-(\d{10})-(\d+)\.hdf5", source_str)
        if match:
            gps_start = float(match.group(1))
            duration = float(match.group(2))
        else:
            gps_start = 0.0
            duration = 1.0
        n_samples = int(duration * 4096.0)
        data = np.arange(float(n_samples))
        return FakeTimeSeries(data, t0=gps_start, sample_rate=4096.0, name="cached")


class FakeTimeSeriesModule:
    """Fake gwpy.timeseries module."""

    TimeSeries = FakeTimeSeries


class FakeGwoscLocateModule:
    """Fake gwosc.locate module for testing."""

    @staticmethod
    def get_urls(
        detector: str,
        start: int,
        end: int,
        sample_rate: int = 4096,
        host: str = "https://gwosc.org",
    ) -> list[str]:
        """Return fake GWOSC URLs."""
        return [
            f"https://gwosc.org/archive/data/O3b_4KHZ_R1/{start}/{detector[0]}-{detector}_GWOSC_O3b_4KHZ_R1-{start}-4096.hdf5"
        ]


def _make_fake_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch import_module to return fake gwpy and gwosc modules."""
    fetcher_mod = import_module("gwmock_noise.gwosc.fetcher")

    def fake_import(name: str) -> Any:
        if name == "gwpy.timeseries":
            return FakeTimeSeriesModule()
        if name == "gwosc.locate":
            return FakeGwoscLocateModule()
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(fetcher_mod, "import_module", fake_import)

    # Also patch urlretrieve to avoid actual downloads
    def fake_urlretrieve(url: str, filename: str) -> None:
        Path(filename).write_text("fake hdf5 content")

    monkeypatch.setattr(fetcher_mod, "urlretrieve", fake_urlretrieve)


class TestGwoscNoiseFetcher:
    """Tests for GwoscNoiseFetcher."""

    def test_fetch_raw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """fetch_raw returns TimeSeries for all detectors."""
        _make_fake_modules(monkeypatch)

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
        _make_fake_modules(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

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
        _make_fake_modules(monkeypatch)

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
        _make_fake_modules(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(
            gps_start=0.0,
            gps_end=100.0,
            detectors=["H1"],
            filters=GwoscFilterConfig(filter_types=[]),
        )
        fetcher = GwoscNoiseFetcher(config)

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

    def test_cache_dir_is_accepted(self, tmp_path: Path) -> None:
        """cache_dir can be set in config."""
        cache = tmp_path / "gwosc_cache"
        config = GwoscNoiseConfig(
            gps_start=1261875618,
            gps_end=1261877618,
            cache_dir=cache,
        )
        assert config.cache_dir == cache

    def test_cache_dir_defaults_to_none(self) -> None:
        """cache_dir defaults to None (no caching)."""
        config = GwoscNoiseConfig(gps_start=0.0, gps_end=100.0)
        assert config.cache_dir is None

    def test_fetch_raw_with_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """fetch_raw uses cache when cache_dir is set."""
        _make_fake_modules(monkeypatch)

        from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher

        config = GwoscNoiseConfig(
            gps_start=1261875618,
            gps_end=1261877618,
            detectors=["H1"],
            cache_dir=tmp_path / "cache",
        )
        fetcher = GwoscNoiseFetcher(config)
        result = fetcher.fetch_raw()

        assert "H1" in result
        assert isinstance(result["H1"], FakeTimeSeries)
        # Cache directory should have been created
        assert (tmp_path / "cache").exists()

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

    def test_fetch_clean_with_cache_dir(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """fetch_clean passes format='hdf5.gwosc' when reading from cache."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Pre-create a fake cached file matching the expected filename pattern
        fake_file = cache_dir / "H-H1_GWOSC_O3b_4KHZ_R1-1261875618-4096.hdf5"
        fake_file.write_text("dummy")

        captured_formats: list[str] = []
        original_read = FakeTimeSeries.read

        @staticmethod
        def read_spy(source, format=""):  # noqa: A002
            captured_formats.append(format)
            return original_read(source, format=format)

        config = GwoscNoiseConfig(
            detectors=["H1"],
            gps_start=1261875618.0,
            gps_end=1261879714.0,
            sample_rate=4096,
            cache_dir=cache_dir,
        )

        class _FakeLoc:
            @staticmethod
            def get_urls(**kw) -> list[str]:
                return ["https://gwosc.org/H-H1_GWOSC_O3b_4KHZ_R1-1261875618-4096.hdf5"]

        monkeypatch.setattr("gwmock_noise.gwosc.fetcher._import_gwosc_locate", _FakeLoc)
        monkeypatch.setattr("gwmock_noise.gwosc.fetcher._load_timeseries", lambda: FakeTimeSeries)
        monkeypatch.setattr(FakeTimeSeries, "read", read_spy)

        monkeypatch.setattr(
            "gwmock_noise.gwosc.fetcher.GwoscSegmentFilter.compute_clean_segments",
            lambda self, start, end, detectors: {"H1": [(1261875618.0, 1261876000.0)]},
        )

        fetcher = GwoscNoiseFetcher(config)
        result = fetcher.fetch_clean()

        assert "H1" in result
        assert len(result["H1"]) == 1
        assert "hdf5.gwosc" in captured_formats, f"Expected format='hdf5.gwosc' but got: {captured_formats}"
