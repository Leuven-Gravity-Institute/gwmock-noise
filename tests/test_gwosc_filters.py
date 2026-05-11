"""Tests for GWOSC segment filtering logic."""

from __future__ import annotations

from importlib import import_module
from typing import Any

import pytest

from gwmock_noise.gwosc.filters import (
    GwoscSegmentFilter,
    _invert_segments,
    _merge_segments,
)
from gwmock_noise.gwosc.models import FilterType, GwoscFilterConfig


class TestMergeSegments:
    """Tests for the _merge_segments helper."""

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        assert _merge_segments([]) == []

    def test_single_segment(self) -> None:
        """Single segment returned unchanged."""
        assert _merge_segments([(0.0, 10.0)]) == [(0.0, 10.0)]

    def test_non_overlapping(self) -> None:
        """Non-overlapping segments are returned sorted."""
        segments = [(20.0, 30.0), (0.0, 10.0)]
        assert _merge_segments(segments) == [(0.0, 10.0), (20.0, 30.0)]

    def test_overlapping(self) -> None:
        """Overlapping segments are merged."""
        segments = [(0.0, 10.0), (5.0, 15.0)]
        assert _merge_segments(segments) == [(0.0, 15.0)]

    def test_contained(self) -> None:
        """Contained segment is absorbed."""
        segments = [(0.0, 20.0), (5.0, 10.0)]
        assert _merge_segments(segments) == [(0.0, 20.0)]

    def test_adjacent(self) -> None:
        """Adjacent segments are merged."""
        segments = [(0.0, 10.0), (10.0, 20.0)]
        assert _merge_segments(segments) == [(0.0, 20.0)]

    def test_multiple_overlaps(self) -> None:
        """Multiple overlapping segments are merged correctly."""
        segments = [(0.0, 10.0), (5.0, 15.0), (12.0, 25.0), (30.0, 40.0)]
        assert _merge_segments(segments) == [(0.0, 25.0), (30.0, 40.0)]


class TestInvertSegments:
    """Tests for the _invert_segments helper."""

    def test_empty_vetosegments(self) -> None:
        """No vetosegments returns the full interval."""
        assert _invert_segments([], 0.0, 100.0) == [(0.0, 100.0)]

    def test_single_veto(self) -> None:
        """Single veto yields two clean segments."""
        result = _invert_segments([(30.0, 50.0)], 0.0, 100.0)
        assert result == [(0.0, 30.0), (50.0, 100.0)]

    def test_veto_at_start(self) -> None:
        """Veto at start yields one clean segment after."""
        result = _invert_segments([(0.0, 20.0)], 0.0, 100.0)
        assert result == [(20.0, 100.0)]

    def test_veto_at_end(self) -> None:
        """Veto at end yields one clean segment before."""
        result = _invert_segments([(80.0, 100.0)], 0.0, 100.0)
        assert result == [(0.0, 80.0)]

    def test_full_coverage(self) -> None:
        """Full veto coverage returns no clean segments."""
        result = _invert_segments([(0.0, 100.0)], 0.0, 100.0)
        assert result == []

    def test_multiple_vetos(self) -> None:
        """Multiple vetos yield correct clean segments."""
        result = _invert_segments([(10.0, 20.0), (40.0, 60.0), (80.0, 90.0)], 0.0, 100.0)
        assert result == [(0.0, 10.0), (20.0, 40.0), (60.0, 80.0), (90.0, 100.0)]

    def test_veto_outside_range(self) -> None:
        """Vetos outside the range are clipped."""
        result = _invert_segments([(30.0, 50.0), (110.0, 120.0)], 0.0, 100.0)
        assert result == [(0.0, 30.0), (50.0, 100.0)]


class FakeGwoscDatasets:
    """Fake gwosc.datasets module for testing."""

    def __init__(self, events: dict[str, float], far_events: dict[str, float]) -> None:
        """Initialize with event name -> GPS time mappings."""
        self._events = events
        self._far_events = far_events

    def query_events(self, *, select: list[str], host: str = "") -> list[str]:
        """Return event names matching the FAR filter."""
        # Parse the select clauses
        far_limit: float | None = None
        for clause in select:
            clause_stripped = clause.strip()
            if clause_stripped.startswith("far <="):
                far_limit = float(clause_stripped.split("<=")[1].strip())
        if far_limit is not None:
            return list(self._far_events.keys())
        return list(self._events.keys())

    def event_gps(self, event: str) -> float:
        """Return GPS time for an event name."""
        combined = {**self._events, **self._far_events}
        return combined[event]

    def event_segment(self, event: str) -> tuple[float, float]:
        """Return a dummy event segment."""
        gps = self.event_gps(event)
        return (gps - 4, gps + 4)


class FakeGwoscTimeline:
    """Fake gwosc.timeline module for testing."""

    def __init__(self, segments: dict[str, list[tuple[int, int]]]) -> None:
        """Initialize with flag_name -> segment list."""
        self._segments = segments

    def get_segments(
        self,
        flag: str,
        start: int,
        end: int,
        host: str = "",
    ) -> list[tuple[int, int]]:
        """Return segments for a flag."""
        return self._segments.get(flag, [])


class FakeGwoscApi:
    """Fake gwosc.api module for testing."""

    def __init__(self, dataset_json: dict[str, Any]) -> None:
        """Initialize with the dataset JSON response."""
        self._dataset_json = dataset_json

    def fetch_dataset_json(self, gpsstart: int, gpsend: int, host: str = "") -> dict[str, Any]:
        """Return the fake dataset JSON."""
        return self._dataset_json


class FakeGwoscModule:
    """Fake gwosc module for testing."""

    def __init__(
        self,
        datasets: FakeGwoscDatasets,
        timeline: FakeGwoscTimeline,
        api: FakeGwoscApi,
    ) -> None:
        """Initialize with fake submodules."""
        self.datasets = datasets
        self.timeline = timeline
        self.api = api


def _make_fake_gwosc_module(monkeypatch: pytest.MonkeyPatch, fake_module: Any) -> None:
    """Patch import_module to return fake gwosc submodules."""
    filters_mod = import_module("gwmock_noise.gwosc.filters")

    def fake_import(name: str) -> Any:
        if name == "gwosc.datasets":
            return fake_module.datasets
        if name == "gwosc.timeline":
            return fake_module.timeline
        if name == "gwosc.api":
            return fake_module.api
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(filters_mod, "import_module", fake_import)


@pytest.fixture
def gw_events() -> dict[str, float]:
    """Sample GW events for testing."""
    return {
        "GW150914-v3": 1126259462.4,
        "GW151226-v2": 1135136350.6,
    }


@pytest.fixture
def far_events() -> dict[str, float]:
    """Sample high-confidence (low FAR) events."""
    return {
        "GW150914-v3": 1126259462.4,
    }


@pytest.fixture
def dq_segments() -> dict[str, list[tuple[int, int]]]:
    """Sample DQ vetosegments."""
    return {
        "H1_CBC_CAT1": [(1126260000, 1126260100)],
        "L1_CBC_CAT1": [(1126260000, 1126260200)],
        "H1_CBC_CAT2": [(1126259500, 1126259550)],
        "L1_CBC_CAT2": [],
    }


@pytest.fixture
def dataset_json() -> dict[str, Any]:
    """Sample dataset JSON response."""
    return {
        "events": {
            "GW150914-v3": {
                "GPStime": 1126259462.4,
                "detectors": ["H1", "L1"],
                "DQbits": 7,
                "INJbits": 0,
            },
            "blind_injection-v1": {
                "GPStime": 1126259500.0,
                "detectors": ["H1", "L1"],
                "DQbits": 1,
                "INJbits": 1,
            },
        },
        "runs": {},
    }


class TestGwoscSegmentFilter:
    """Tests for GwoscSegmentFilter."""

    def test_get_gw_vetosegments_high_confidence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """High-confidence filter returns only low-FAR events."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(
            filter_types=[FilterType.HIGH_CONFIDENCE_GW],
            far_threshold=1.0,
            event_padding=10.0,
        )
        filt = GwoscSegmentFilter(config)

        segments = filt.get_gw_vetosegments(1126259000, 1126260000)
        # Only GW150914-v3 has FAR <= 1
        assert len(segments) == 1
        gps = far_events["GW150914-v3"]
        assert segments[0] == (gps - 10.0, gps + 10.0)

    def test_get_gw_vetosegments_all_signals(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """All-signals filter returns all events (no FAR filter)."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(
            filter_types=[FilterType.ALL_GW_SIGNALS],
            event_padding=5.0,
        )
        filt = GwoscSegmentFilter(config)

        segments = filt.get_gw_vetosegments(1126259000, 1135150000)
        # All events in range (no FAR filter for ALL_GW_SIGNALS)
        assert len(segments) == 2
        expected_gps = sorted(gw_events.values())
        for i, gps in enumerate(expected_gps):
            assert segments[i] == (gps - 5.0, gps + 5.0)

    def test_get_dq_vetosegments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """DQ vetosegments are returned for a detector."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(
            filter_types=[FilterType.DATA_QUALITY],
            dq_flags=["CBC_CAT1", "CBC_CAT2"],
        )
        filt = GwoscSegmentFilter(config)

        segments = filt.get_dq_vetosegments(1126259000, 1126261000, "H1")
        # H1 has both CAT1 and CAT2 segments in range
        assert len(segments) == 2
        assert segments[0] == (1126259500, 1126259550)  # CAT2
        assert segments[1] == (1126260000, 1126260100)  # CAT1

    def test_get_dq_vetosegments_no_flags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """Empty DQ flags returns empty list."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(dq_flags=[])
        filt = GwoscSegmentFilter(config)

        segments = filt.get_dq_vetosegments(1126259000, 1126261000, "H1")
        assert segments == []

    def test_get_hardware_injection_segments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """Hardware injection segments are identified by INJbits."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(
            filter_types=[FilterType.HIGH_CONFIDENCE_GW],
            event_padding=10.0,
        )
        filt = GwoscSegmentFilter(config)

        segments = filt.get_hardware_injection_segments(1126259000, 1126260000)
        # blind_injection-v1 has INJbits=1
        assert len(segments) == 1
        assert segments[0] == (1126259500.0 - 10.0, 1126259500.0 + 10.0)

    def test_compute_clean_segments(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """Clean segments combine GW and DQ vetosegments."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(
            filter_types=[FilterType.HIGH_CONFIDENCE_GW, FilterType.DATA_QUALITY],
            far_threshold=1.0,
            event_padding=10.0,
            dq_flags=["CBC_CAT1", "CBC_CAT2"],
        )
        filt = GwoscSegmentFilter(config)

        clean = filt.compute_clean_segments(1126259000, 1126261000, ["H1", "L1"])

        assert "H1" in clean
        assert "L1" in clean
        # Both detectors should have some clean segments
        assert len(clean["H1"]) > 0
        assert len(clean["L1"]) > 0

    def test_compute_clean_segments_no_filters(
        self,
        monkeypatch: pytest.MonkeyPatch,
        gw_events: dict[str, float],
        far_events: dict[str, float],
        dq_segments: dict[str, list[tuple[int, int]]],
        dataset_json: dict[str, Any],
    ) -> None:
        """No filters returns the full interval as clean."""
        fake_datasets = FakeGwoscDatasets(gw_events, far_events)
        fake_timeline = FakeGwoscTimeline(dq_segments)
        fake_api = FakeGwoscApi(dataset_json)
        fake_gwosc = FakeGwoscModule(fake_datasets, fake_timeline, fake_api)
        _make_fake_gwosc_module(monkeypatch, fake_gwosc)

        config = GwoscFilterConfig(filter_types=[])
        filt = GwoscSegmentFilter(config)

        clean = filt.compute_clean_segments(0.0, 100.0, ["H1"])
        assert clean == {"H1": [(0.0, 100.0)]}

    def test_import_error_when_gwosc_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Clear ImportError is raised when gwosc is not installed."""
        filters_mod = import_module("gwmock_noise.gwosc.filters")

        def fake_import(name: str) -> None:
            raise ImportError("No module named 'gwosc'")

        monkeypatch.setattr(filters_mod, "import_module", fake_import)

        config = GwoscFilterConfig()
        filt = GwoscSegmentFilter(config)

        with pytest.raises(ImportError, match="pip install gwmock-noise\\[gwosc\\]"):
            filt.get_gw_vetosegments(0, 100)
