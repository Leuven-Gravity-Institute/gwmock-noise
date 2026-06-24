"""Fetch real detector strain data from GWOSC.

Downloads HDF5 strain files from GWOSC, supports file-level caching to
a local directory, and applies user-configured filters to return clean
noise segments.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.request import urlretrieve

from gwmock_noise.gwosc.filters import GwoscSegmentFilter
from gwmock_noise.gwosc.models import GwoscNoiseConfig

if TYPE_CHECKING:
    from gwpy.timeseries import TimeSeries

_GWPY_IMPORT_ERROR = "gwpy is required to use GwoscNoiseFetcher. Install it with `pip install gwmock-noise[gwpy]`."
_GWOSC_IMPORT_ERROR = "gwosc is required to use GwoscNoiseFetcher. Install it with `pip install gwmock-noise[gwosc]`."


def _load_timeseries() -> type[TimeSeries]:
    """Import and return gwpy.TimeSeries on demand."""
    try:
        module = import_module("gwpy.timeseries")
    except ImportError as exc:
        raise ImportError(_GWPY_IMPORT_ERROR) from exc
    return module.TimeSeries


def _import_gwosc_locate():
    """Import and return gwosc.locate on demand."""
    try:
        return import_module("gwosc.locate")
    except ImportError as exc:
        raise ImportError(_GWOSC_IMPORT_ERROR) from exc


def _fetch_via_cache(  # noqa: PLR0913
    detector: str,
    gps_start: float,
    gps_end: float,
    sample_rate: int,
    cache_dir: Path,
    host: str,
) -> TimeSeries:
    """Fetch strain data using a local file cache.

    Downloads HDF5 files from GWOSC to ``cache_dir``, reusing cached
    files on subsequent calls for the same GPS interval.

    Args:
        detector: Detector prefix (e.g. ``"H1"``).
        gps_start: GPS start time.
        gps_end: GPS end time.
        sample_rate: Sampling rate in Hz.
        cache_dir: Local directory for cached HDF5 files.
        host: GWOSC host URL.

    Returns:
        A ``gwpy.TimeSeries`` for the full GPS interval.

    Raises:
        ValueError: If no GWOSC data URLs are found or download fails.
    """
    locate = _import_gwosc_locate()
    timeseries_cls = _load_timeseries()

    cache_dir.mkdir(parents=True, exist_ok=True)

    urls = locate.get_urls(
        detector=detector,
        start=int(gps_start),
        end=int(gps_end),
        sample_rate=sample_rate,
        host=host,
    )

    if not urls:
        raise ValueError(f"No GWOSC data URLs found for {detector} [{gps_start}, {gps_end}) at {sample_rate} Hz.")

    series_parts: list[TimeSeries] = []
    for url in urls:
        filename = url.rstrip("/").rsplit("/", 1)[-1]
        cache_path = cache_dir / filename

        if not cache_path.exists():
            urlretrieve(url, cache_path)  # noqa: S310

        series_part = timeseries_cls.read(cache_path, format="hdf5.gwosc")
        series_parts.append(series_part)

    if len(series_parts) == 1:
        full_series = series_parts[0]
    else:
        full_series = series_parts[0]
        for part in series_parts[1:]:
            full_series = full_series.append(part)

    cropped = full_series.crop(gps_start, gps_end)
    cropped.name = detector
    return cropped


class GwoscNoiseFetcher:
    """Fetch real detector noise data from GWOSC with optional filtering.

    Downloads strain data from GWOSC and applies user-configured filters
    to exclude segments containing GW signals and data-quality issues.

    When ``cache_dir`` is configured, HDF5 files are saved locally and
    reused on subsequent requests — avoiding repeated downloads for the
    same GPS interval.

    Attributes:
        config: The GWOSC noise fetching configuration.
    """

    def __init__(self, config: GwoscNoiseConfig) -> None:
        """Initialize the fetcher.

        Args:
            config: Configuration specifying detectors, GPS range,
                sample rate, filtering options, and optional cache
                directory.
        """
        _load_timeseries()
        self.config = config
        self._segment_filter = GwoscSegmentFilter(config.filters)

    def check_availability(self, detectors: list[str] | None = None) -> dict[str, bool]:
        """Probe GWOSC for per-detector strain-data availability.

        Unlike :attr:`clean_segments`, which only reflects *vetoes* (GW
        events and data-quality flags), this checks whether the strain
        data itself has been published and is downloadable for the full
        configured GPS interval. A detector can have a fully "clean"
        interval yet have no published strain — e.g. the data has not
        been released yet for the observing run — in which case a fetch
        would fail. Use this for a pre-flight check before fetching.

        Args:
            detectors: Detectors to probe. Defaults to all configured
                detectors when ``None``.

        Returns:
            A dictionary mapping each requested detector to ``True`` if
            GWOSC has data URLs covering the interval, ``False`` otherwise.
        """
        locate = _import_gwosc_locate()
        sample_rate = int(self.config.sample_rate)
        requested = self.config.detectors if detectors is None else detectors

        availability: dict[str, bool] = {}
        for detector in requested:
            try:
                urls = locate.get_urls(
                    detector=detector,
                    start=int(self.config.gps_start),
                    end=int(self.config.gps_end),
                    sample_rate=sample_rate,
                    host=self.config.host,
                )
            except (ValueError, OSError):
                urls = []
            availability[detector] = bool(urls)
        return availability

    def _require_available_detectors(self, detectors: list[str] | None = None) -> None:
        """Raise a clear error if any requested detector lacks published data.

        Args:
            detectors: Detectors to check. Defaults to all configured
                detectors when ``None``.

        Raises:
            ValueError: Listing every detector for which GWOSC has no
                strain data covering the configured interval.
        """
        availability = self.check_availability(detectors)
        unavailable = [detector for detector, ok in availability.items() if not ok]
        if unavailable:
            raise ValueError(
                f"GWOSC has no published strain data covering "
                f"[{self.config.gps_start}, {self.config.gps_end}) at "
                f"{int(self.config.sample_rate)} Hz for detector(s): {', '.join(unavailable)}. "
                f"The strain may not be released yet for this observing run. "
                f"Note that clean-segment computation only reflects vetoes (GW events "
                f"and data-quality flags), not whether the data is actually available."
            )

    def _fetch_detector(self, detector: str) -> TimeSeries:
        """Fetch strain data for a single detector, using cache if configured.

        Args:
            detector: Detector prefix.

        Returns:
            A ``gwpy.TimeSeries`` for the full GPS interval.
        """
        timeseries_cls = _load_timeseries()
        # gwpy 4.x expects int sample_rate, not float
        sample_rate = int(self.config.sample_rate)

        if self.config.cache_dir is not None:
            return _fetch_via_cache(
                detector=detector,
                gps_start=self.config.gps_start,
                gps_end=self.config.gps_end,
                sample_rate=sample_rate,
                cache_dir=self.config.cache_dir,
                host=self.config.host,
            )

        return timeseries_cls.fetch_open_data(
            detector,
            self.config.gps_start,
            self.config.gps_end,
            sample_rate=sample_rate,
            host=self.config.host,
        )

    def fetch_raw(self, detectors: list[str] | None = None) -> dict[str, TimeSeries]:
        """Fetch raw strain data without filtering.

        Args:
            detectors: Detectors to fetch. Defaults to all configured
                detectors when ``None``.

        Returns:
            A dictionary mapping each requested detector to a full-interval
            ``gwpy.TimeSeries``.

        Raises:
            ValueError: If no data is available for any requested detector.
        """
        requested = self.config.detectors if detectors is None else detectors
        self._require_available_detectors(requested)

        result: dict[str, TimeSeries] = {}
        for detector in requested:
            try:
                result[detector] = self._fetch_detector(detector)
            except Exception as exc:
                raise ValueError(
                    f"Failed to fetch data for {detector} [{self.config.gps_start}, {self.config.gps_end}): {exc}"
                ) from exc
        return result

    def fetch_clean(self, detectors: list[str] | None = None) -> dict[str, list[TimeSeries]]:
        """Fetch clean noise segments.

        Clean segments are computed by excluding GW events and
        data-quality issues according to the filter configuration.

        Args:
            detectors: Detectors to fetch. Defaults to all configured
                detectors when ``None``.

        Returns:
            A dictionary mapping each requested detector to a list of
            ``gwpy.TimeSeries``, one per clean segment.

        Raises:
            ValueError: If no data is available for any requested detector
                or no clean segments are found.
        """
        requested = self.config.detectors if detectors is None else detectors
        self._require_available_detectors(requested)

        clean_segments = self._segment_filter.compute_clean_segments(
            self.config.gps_start,
            self.config.gps_end,
            requested,
        )

        result: dict[str, list[TimeSeries]] = {}
        for detector in requested:
            segments = clean_segments.get(detector, [])
            if not segments:
                raise ValueError(
                    f"No clean segments found for {detector} "
                    f"in [{self.config.gps_start}, {self.config.gps_end}). "
                    f"Try relaxing the filter criteria."
                )

            full_series = self._fetch_detector(detector)

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
