"""Tests for GWOSC configuration models."""

from __future__ import annotations

import pytest

from gwmock_noise.gwosc.models import FilterType, GwoscFilterConfig, GwoscNoiseConfig


class TestFilterType:
    """Tests for the FilterType enum."""

    def test_enum_values(self) -> None:
        """FilterType has the expected members."""
        assert FilterType.HIGH_CONFIDENCE_GW.value == "high_confidence_gw"
        assert FilterType.ALL_GW_SIGNALS.value == "all_gw_signals"
        assert FilterType.DATA_QUALITY.value == "data_quality"

    def test_enum_from_string(self) -> None:
        """FilterType can be constructed from a string."""
        assert FilterType("high_confidence_gw") == FilterType.HIGH_CONFIDENCE_GW


class TestGwoscFilterConfig:
    """Tests for GwoscFilterConfig."""

    def test_default_values(self) -> None:
        """Default config has sensible values."""
        config = GwoscFilterConfig()
        assert config.filter_types == [FilterType.HIGH_CONFIDENCE_GW, FilterType.DATA_QUALITY]
        assert config.far_threshold == 1.0
        assert config.event_padding == 16.0
        assert config.dq_flags == ["CBC_CAT1", "CBC_CAT2"]
        assert config.exclude_hardware_injections is True

    def test_has_gw_filter_true(self) -> None:
        """has_gw_filter is True when GW filter types are present."""
        config = GwoscFilterConfig(filter_types=[FilterType.HIGH_CONFIDENCE_GW])
        assert config.has_gw_filter is True

    def test_has_gw_filter_false(self) -> None:
        """has_gw_filter is False when only DQ filter is present."""
        config = GwoscFilterConfig(filter_types=[FilterType.DATA_QUALITY])
        assert config.has_gw_filter is False

    def test_has_dq_filter(self) -> None:
        """has_dq_filter is True when DATA_QUALITY is in filter_types."""
        config = GwoscFilterConfig(filter_types=[FilterType.DATA_QUALITY])
        assert config.has_dq_filter is True

    def test_include_marginal_events(self) -> None:
        """include_marginal_events is True only for ALL_GW_SIGNALS."""
        high = GwoscFilterConfig(filter_types=[FilterType.HIGH_CONFIDENCE_GW])
        assert high.include_marginal_events is False

        all_gw = GwoscFilterConfig(filter_types=[FilterType.ALL_GW_SIGNALS])
        assert all_gw.include_marginal_events is True

    def test_custom_far_threshold(self) -> None:
        """Custom FAR threshold is accepted."""
        config = GwoscFilterConfig(far_threshold=0.1)
        assert config.far_threshold == 0.1

    def test_custom_dq_flags(self) -> None:
        """Custom DQ flags are accepted."""
        config = GwoscFilterConfig(dq_flags=["CBC_CAT1", "CBC_CAT2", "CBC_CAT3"])
        assert config.dq_flags == ["CBC_CAT1", "CBC_CAT2", "CBC_CAT3"]

    def test_empty_filter_types_allowed(self) -> None:
        """Empty filter_types list is allowed (no filtering)."""
        config = GwoscFilterConfig(filter_types=[])
        assert config.filter_types == []
        assert config.has_gw_filter is False
        assert config.has_dq_filter is False

    def test_zero_far_threshold_raises(self) -> None:
        """Zero FAR threshold raises validation error."""
        with pytest.raises(ValueError, match="greater than 0"):
            GwoscFilterConfig(far_threshold=0)

    def test_negative_event_padding_is_allowed(self) -> None:
        """Negative event padding is NOT allowed (ge=0)."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            GwoscFilterConfig(event_padding=-1.0)

    def test_empty_dq_flags_allowed(self) -> None:
        """Empty dq_flags list is allowed (no DQ filtering)."""
        config = GwoscFilterConfig(dq_flags=[])
        assert config.dq_flags == []

    def test_extra_fields_ignored(self) -> None:
        """Extra fields in config are ignored."""
        config = GwoscFilterConfig(unknown_field="value")  # type: ignore[call-arg]
        assert not hasattr(config, "unknown_field")


class TestGwoscNoiseConfig:
    """Tests for GwoscNoiseConfig."""

    def test_minimal_config(self) -> None:
        """Minimal config requires gps_start and gps_end."""
        config = GwoscNoiseConfig(gps_start=1261875618, gps_end=1261876618)
        assert config.gps_start == 1261875618
        assert config.gps_end == 1261876618
        assert config.duration == 1000.0
        assert config.detectors == ["H1", "L1"]
        assert config.sample_rate == 4096.0
        assert config.host == "https://gwosc.org"
        assert config.cache_dir is None

    def test_duration_property(self) -> None:
        """Duration is gps_end - gps_start."""
        config = GwoscNoiseConfig(gps_start=100, gps_end=200)
        assert config.duration == 100.0

    def test_gps_end_before_start_raises(self) -> None:
        """gps_end must be after gps_start."""
        with pytest.raises(ValueError, match="gps_end must be greater than gps_start"):
            GwoscNoiseConfig(gps_start=200, gps_end=100)

    def test_gps_end_equal_start_raises(self) -> None:
        """gps_end equal to gps_start raises."""
        with pytest.raises(ValueError, match="gps_end must be greater than gps_start"):
            GwoscNoiseConfig(gps_start=100, gps_end=100)

    def test_custom_detectors(self) -> None:
        """Custom detector list is accepted."""
        config = GwoscNoiseConfig(gps_start=100, gps_end=200, detectors=["H1", "L1", "V1"])
        assert config.detectors == ["H1", "L1", "V1"]

    def test_empty_detectors_raises(self) -> None:
        """Empty detectors list raises validation error."""
        with pytest.raises(ValueError, match="at least 1 item"):
            GwoscNoiseConfig(gps_start=100, gps_end=200, detectors=[])

    def test_custom_sample_rate(self) -> None:
        """Custom sample rate is accepted."""
        config = GwoscNoiseConfig(gps_start=100, gps_end=200, sample_rate=16384)
        assert config.sample_rate == 16384

    def test_filters_config_default(self) -> None:
        """Default filter config is created."""
        config = GwoscNoiseConfig(gps_start=100, gps_end=200)
        assert isinstance(config.filters, GwoscFilterConfig)

    def test_extra_fields_ignored(self) -> None:
        """Extra fields in config are ignored."""
        config = GwoscNoiseConfig(gps_start=100, gps_end=200, unknown="value")  # type: ignore[call-arg]
        assert not hasattr(config, "unknown")
