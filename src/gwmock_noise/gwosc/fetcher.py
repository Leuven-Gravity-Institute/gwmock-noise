"""Fetch real detector strain data from GWOSC.

Uses ``gwpy.timeseries.TimeSeries.fetch_open_data()`` to download
HDF5 strain files and applies user-configured filters to return
clean noise segments.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from gwmock_noise.gwosc.filters import GwoscSegmentFilter
from gwmock_noise.gwosc.models import GwoscNoiseConfig

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries

_GWPY_IMPORT_ERROR = "gwpy is required to use GwoscNoiseFetcher. Install it with `pip install gwmock-noise[gwpy]`."


def _load_timeseries() -> type[TimeSeries]:
    """Import and return gwpy.TimeSeries on demand."""
    try:
        module = import_module("gwpy.timeseries")
    except ImportError as exc:
        raise ImportError(_GWPY_IMPORT_ERROR) from exc
    return module.TimeSeries


class GwoscNoiseFetcher:
    """Fetch real detector noise data from GWOSC with optional filtering.

    Downloads strain data via ``gwpy.TimeSeries.fetch_open_data()`` and
    applies user-configured filters to exclude segments containing GW
    signals and data-quality issues.

    Attributes:
        config: The GWOSC noise fetching configuration.
    """

    def __init__(self, config: GwoscNoiseConfig) -> None:
        """Initialize the fetcher.

        Args:
            config: Configuration specifying detectors, GPS range,
                sample rate, and filtering options.
        """
        _load_timeseries()
        self.config = config
        self._segment_filter = GwoscSegmentFilter(config.filters)

    def fetch_raw(self) -> dict[str, TimeSeries]:
        """Fetch raw strain data for all detectors without filtering.

        Returns:
            A dictionary mapping each detector to a full-interval
            ``gwpy.TimeSeries``.

        Raises:
            ValueError: If no data is available for any detector.
        """
        timeseries_cls = _load_timeseries()
        result: dict[str, TimeSeries] = {}
        for detector in self.config.detectors:
            try:
                series = timeseries_cls.fetch_open_data(
                    detector,
                    self.config.gps_start,
                    self.config.gps_end,
                    sample_rate=self.config.sample_rate,
                    host=self.config.host,
                    cache=self.config.cache,
                )
                result[detector] = series
            except Exception as exc:
                raise ValueError(
                    f"Failed to fetch data for {detector} [{self.config.gps_start}, {self.config.gps_end}): {exc}"
                ) from exc
        return result

    def fetch_clean(self) -> dict[str, list[TimeSeries]]:
        """Fetch clean noise segments for all detectors.

        Clean segments are computed by excluding GW events and
        data-quality issues according to the filter configuration.

        Returns:
            A dictionary mapping each detector to a list of
            ``gwpy.TimeSeries``, one per clean segment.

        Raises:
            ValueError: If no data is available for any detector or
                no clean segments are found.
        """
        timeseries_cls = _load_timeseries()

        clean_segments = self._segment_filter.compute_clean_segments(
            self.config.gps_start,
            self.config.gps_end,
            self.config.detectors,
        )

        result: dict[str, list[TimeSeries]] = {}
        for detector in self.config.detectors:
            segments = clean_segments.get(detector, [])
            if not segments:
                raise ValueError(
                    f"No clean segments found for {detector} "
                    f"in [{self.config.gps_start}, {self.config.gps_end}). "
                    f"Try relaxing the filter criteria."
                )

            full_series = timeseries_cls.fetch_open_data(
                detector,
                self.config.gps_start,
                self.config.gps_end,
                sample_rate=self.config.sample_rate,
                host=self.config.host,
                cache=self.config.cache,
            )

            clean_list: list[TimeSeries] = []
            for seg_start, seg_end in segments:
                try:
                    cropped = full_series.crop(seg_start, seg_end)
                    cropped.name = detector
                    clean_list.append(cropped)
                except (ValueError, IndexError):
                    continue

            if not clean_list:
                raise ValueError(
                    f"Failed to crop clean segments for {detector}. The data may not cover the requested interval."
                )
            result[detector] = clean_list

        return result

    @property
    def clean_segments(self) -> dict[str, list[tuple[float, float]]]:
        """Return the computed clean segments per detector.

        Returns:
            A dictionary mapping each detector to a list of
            ``(start, end)`` clean segment tuples.
        """
        return self._segment_filter.compute_clean_segments(
            self.config.gps_start,
            self.config.gps_end,
            self.config.detectors,
        )
