"""Pydantic models for GWOSC real-noise fetching configuration."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, Field, model_validator


class FilterType(StrEnum):
    """Types of segments that can be filtered out from GWOSC data."""

    HIGH_CONFIDENCE_GW = "high_confidence_gw"
    ALL_GW_SIGNALS = "all_gw_signals"
    DATA_QUALITY = "data_quality"


DEFAULT_DQ_FLAGS = ["CBC_CAT1", "CBC_CAT2"]
DEFAULT_EVENT_PADDING = 16.0
DEFAULT_FAR_THRESHOLD = 1.0


class GwoscFilterConfig(BaseModel):
    """Configuration for filtering GWOSC strain segments.

    Attributes:
        filter_types: Which categories of segments to exclude.
        far_threshold: Maximum false-alarm rate (events/year) for GW event
            filtering. Events with FAR above this threshold are excluded from
            filtering (treated as noise). Default 1.0 (one per year).
        event_padding: Padding in seconds to apply around each GW event GPS
            time when building vetosegments.
        dq_flags: DQ flag basenames (without detector prefix) to query for
            data-quality vetosegments. E.g. ``["CBC_CAT1", "CBC_CAT2"]``.
            The detector prefix is prepended automatically.
        exclude_hardware_injections: Whether to also exclude segments
            containing hardware injections.
    """

    filter_types: list[FilterType] = Field(
        default_factory=lambda: [FilterType.HIGH_CONFIDENCE_GW, FilterType.DATA_QUALITY],
        description="Filter categories to apply.",
    )
    far_threshold: float = Field(
        default=DEFAULT_FAR_THRESHOLD,
        gt=0,
        description="FAR threshold in events/year for GW event filtering.",
    )
    event_padding: float = Field(
        default=DEFAULT_EVENT_PADDING,
        ge=0,
        description="Padding in seconds around GW events.",
    )
    dq_flags: list[str] = Field(
        default_factory=lambda: list(DEFAULT_DQ_FLAGS),
        description="DQ flag basenames (without detector prefix).",
    )
    exclude_hardware_injections: bool = Field(
        default=True,
        description="Exclude segments containing hardware injections.",
    )

    model_config = {"frozen": False, "extra": "ignore"}

    @property
    def has_gw_filter(self) -> bool:
        """Return whether any GW signal filter is active."""
        return FilterType.HIGH_CONFIDENCE_GW in self.filter_types or FilterType.ALL_GW_SIGNALS in self.filter_types

    @property
    def has_dq_filter(self) -> bool:
        """Return whether the data-quality filter is active."""
        return FilterType.DATA_QUALITY in self.filter_types

    @property
    def include_marginal_events(self) -> bool:
        """Return whether to include marginal (low-confidence) GW events."""
        return FilterType.ALL_GW_SIGNALS in self.filter_types


MIN_DETECTOR_PREFIX_LENGTH = 2


class GwoscNoiseConfig(BaseModel):
    """Configuration for fetching real detector noise from GWOSC.

    Attributes:
        detectors: List of detector prefixes (e.g. ``["H1", "L1"]``).
        gps_start: GPS start time of the requested data interval.
        gps_end: GPS end time of the requested data interval.
        sample_rate: Desired sampling rate in Hz. GWOSC typically provides
            4096 Hz, with some event datasets at 16384 Hz.
        filters: Filter configuration for excluding segments.
        host: GWOSC host URL.
        cache_dir: Optional directory for file-level caching. When set,
            downloaded HDF5 frame files are saved locally and reused on
            subsequent requests for the same GPS interval. When ``None``,
            data is downloaded on every call without caching.
    """

    detectors: list[str] = Field(
        default=["H1", "L1"],
        description="Detector prefixes to fetch data for.",
        min_length=1,
    )
    gps_start: float = Field(
        ...,
        description="GPS start time of the requested interval.",
    )
    gps_end: float = Field(
        ...,
        description="GPS end time of the requested interval.",
    )
    sample_rate: float = Field(
        default=4096.0,
        gt=0,
        description="Sampling rate in Hz.",
    )
    filters: GwoscFilterConfig = Field(
        default_factory=GwoscFilterConfig,
        description="Filter configuration for excluding segments.",
    )
    host: str = Field(
        default="https://gwosc.org",
        description="GWOSC host URL.",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Optional directory for file-level caching of downloaded HDF5 files.",
    )

    model_config = {"frozen": False, "extra": "ignore"}

    @property
    def duration(self) -> float:
        """Return the duration of the requested interval in seconds."""
        return self.gps_end - self.gps_start

    @model_validator(mode="after")
    def validate_gps_range(self) -> Self:
        """Validate that gps_end is after gps_start."""
        if self.gps_end <= self.gps_start:
            raise ValueError("gps_end must be greater than gps_start.")
        return self

    @staticmethod
    def _parse_filter_type(value: object) -> FilterType:
        """Normalize a filter-type input to a FilterType enum member."""
        if isinstance(value, FilterType):
            return value
        if isinstance(value, str):
            return FilterType(value)
        raise ValueError(f"Invalid filter type: {value!r}")

    @staticmethod
    def _normalize_detector(detector: str) -> str:
        """Normalize a detector name to the standard two-character prefix."""
        cleaned = detector.strip().upper()
        if len(cleaned) < MIN_DETECTOR_PREFIX_LENGTH:
            raise ValueError(f"Detector name must be at least 2 characters: {detector!r}")
        return cleaned[:2]
