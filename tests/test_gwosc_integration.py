"""Integration tests for GWOSC data fetching (requires network access)."""

from __future__ import annotations

import pytest

gwosc = pytest.importorskip("gwosc")
gwpy = pytest.importorskip("gwpy")

from gwmock_noise.gwosc.filters import GwoscSegmentFilter  # noqa: E402
from gwmock_noise.gwosc.models import FilterType, GwoscFilterConfig  # noqa: E402

pytestmark = pytest.mark.integration

# Use a small time window around GW150914 to test
GW150914_GPS = 1126259462.4
TEST_START = GW150914_GPS - 100
TEST_END = GW150914_GPS + 100


class TestGwoscSegmentFilterIntegration:
    """Integration tests for GwoscSegmentFilter with real GWOSC API."""

    def test_get_gw_vetosegments_high_confidence(self) -> None:
        """High-confidence GW events are found around GW150914."""
        config = GwoscFilterConfig(
            filter_types=[FilterType.HIGH_CONFIDENCE_GW],
            far_threshold=1.0,
            event_padding=10.0,
        )
        filt = GwoscSegmentFilter(config)
        segments = filt.get_gw_vetosegments(TEST_START, TEST_END)

        assert len(segments) >= 1
        # GW150914 should be in the vetosegments
        found = False
        for seg_start, seg_end in segments:
            if seg_start <= GW150914_GPS <= seg_end:
                found = True
                assert seg_start == pytest.approx(GW150914_GPS - 10.0)
                assert seg_end == pytest.approx(GW150914_GPS + 10.0)
                break
        assert found, f"GW150914 not found in vetosegments: {segments}"

    def test_get_gw_vetosegments_all_signals(self) -> None:
        """All-signals filter finds GW150914 without FAR filter."""
        config = GwoscFilterConfig(
            filter_types=[FilterType.ALL_GW_SIGNALS],
            event_padding=5.0,
        )
        filt = GwoscSegmentFilter(config)
        segments = filt.get_gw_vetosegments(TEST_START, TEST_END)

        assert len(segments) >= 1

    def test_get_dq_vetosegments_h1(self) -> None:
        """DQ vetosegments can be queried for H1."""
        config = GwoscFilterConfig(
            filter_types=[FilterType.DATA_QUALITY],
            dq_flags=["CBC_CAT1"],
        )
        filt = GwoscSegmentFilter(config)
        segments = filt.get_dq_vetosegments(TEST_START, TEST_END, "H1")
        # Just checking the API works (may or may not have DQ segments)
        assert isinstance(segments, list)

    def test_compute_clean_segments(self) -> None:
        """Clean segments are computed for GW150914 region."""
        config = GwoscFilterConfig(
            filter_types=[FilterType.HIGH_CONFIDENCE_GW],
            far_threshold=1.0,
            event_padding=10.0,
        )
        filt = GwoscSegmentFilter(config)
        clean = filt.compute_clean_segments(TEST_START, TEST_END, ["H1", "L1"])

        assert "H1" in clean
        assert "L1" in clean
        # Both detectors should have clean segments around GW150914
        for det in ["H1", "L1"]:
            total_clean = sum(end - start for start, end in clean[det])
            # Most of the interval should be clean (just the event ± padding is excluded)
            assert total_clean > 0

    def test_no_filters_returns_full_interval(self) -> None:
        """No filter types returns the full interval as clean."""
        config = GwoscFilterConfig(filter_types=[])
        filt = GwoscSegmentFilter(config)
        clean = filt.compute_clean_segments(TEST_START, TEST_END, ["H1"])

        assert clean["H1"] == [(TEST_START, TEST_END)]
