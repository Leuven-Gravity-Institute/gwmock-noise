"""Segment filtering for GWOSC data.

Computes vetosegments (segments to exclude) based on GW events and
data-quality flags, returning the clean (analysis-ready) time intervals.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from gwmock_noise.gwosc.models import GwoscFilterConfig

if TYPE_CHECKING:
    pass

_GWOSC_IMPORT_ERROR = "gwosc is required to use GwoscSegmentFilter. Install it with `pip install gwmock-noise[gwosc]`."


def _import_gwosc():
    """Import and return the gwosc submodules on demand."""
    try:
        datasets = import_module("gwosc.datasets")
        timeline = import_module("gwosc.timeline")
        api = import_module("gwosc.api")
    except ImportError as exc:
        raise ImportError(_GWOSC_IMPORT_ERROR) from exc
    return _GwoscNamespace(datasets=datasets, timeline=timeline, api=api)


Segment = tuple[float, float]
SegmentList = list[Segment]


def _merge_segments(segments: SegmentList) -> SegmentList:
    """Merge overlapping or adjacent segments.

    Args:
        segments: A list of ``(start, end)`` segment tuples.

    Returns:
        A sorted, merged list of non-overlapping segments.
    """
    if not segments:
        return []
    sorted_segments = sorted(segments, key=lambda s: s[0])
    merged: SegmentList = [sorted_segments[0]]
    for current in sorted_segments[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def _invert_segments(segments: SegmentList, gps_start: float, gps_end: float) -> SegmentList:
    """Return the complement of *segments* within ``[gps_start, gps_end]``.

    Args:
        segments: Non-overlapping segments to invert.
        gps_start: Start of the enclosing interval.
        gps_end: End of the enclosing interval.

    Returns:
        The segments within ``[gps_start, gps_end]`` that are not covered
        by *segments*.
    """
    merged = _merge_segments(segments)
    inverted: SegmentList = []
    cursor = gps_start
    for seg_start, seg_end in merged:
        clipped_start = max(seg_start, gps_start)
        clipped_end = min(seg_end, gps_end)
        if clipped_start >= clipped_end:
            continue
        if cursor < clipped_start:
            inverted.append((cursor, clipped_start))
        cursor = max(cursor, clipped_end)
    if cursor < gps_end:
        inverted.append((cursor, gps_end))
    return inverted


class _GwoscNamespace:
    """Namespace holding gwosc submodules."""

    __slots__ = ("api", "datasets", "timeline")

    def __init__(self, *, datasets: Any, timeline: Any, api: Any) -> None:
        """Initialize with gwosc submodules."""
        self.datasets = datasets
        self.timeline = timeline
        self.api = api


class GwoscSegmentFilter:
    """Compute clean analysis segments from GWOSC metadata.

    Queries GWTC event catalogs and data-quality flags to build
    vetosegments (time windows to exclude) and returns the remaining
    clean intervals suitable for noise analysis.

    Attributes:
        config: The filter configuration.
    """

    def __init__(self, config: GwoscFilterConfig) -> None:
        """Initialize the segment filter.

        Args:
            config: Filter configuration specifying which segment
                categories to exclude and their parameters.
        """
        self.config = config

    def get_gw_vetosegments(self, gps_start: float, gps_end: float) -> SegmentList:
        """Query GWTC events in the GPS range and build vetosegments.

        For ``HIGH_CONFIDENCE_GW``, only events with FAR <= *far_threshold*
        are included. For ``ALL_GW_SIGNALS``, all events in the range are
        included.

        Args:
            gps_start: GPS start of the query interval.
            gps_end: GPS end of the query interval.

        Returns:
            A list of ``(start, end)`` vetosegment tuples, each centred
            on a GW event GPS time with *event_padding* on both sides.
        """
        gwosc = _import_gwosc()
        far_threshold = self.config.far_threshold
        include_marginal = self.config.include_marginal_events
        padding = self.config.event_padding

        select = [
            f"gps-time >= {gps_start}",
            f"gps-time <= {gps_end}",
        ]
        if not include_marginal:
            select.append(f"far <= {far_threshold}")

        event_names = gwosc.datasets.query_events(select=select, host=self._host)

        vetosegments: SegmentList = []
        for event_name in event_names:
            try:
                gps_time = gwosc.datasets.event_gps(event_name)
            except (ValueError, KeyError, IndexError):
                continue
            vetosegments.append((gps_time - padding, gps_time + padding))

        return _merge_segments(vetosegments)

    def get_dq_vetosegments(
        self,
        gps_start: float,
        gps_end: float,
        detector: str,
    ) -> SegmentList:
        """Query data-quality flags for *detector* and build vetosegments.

        Uses ``gwosc.timeline.get_segments()`` to fetch pre-computed
        DQ veto segments for each flag in *dq_flags*.

        Args:
            gps_start: GPS start of the query interval.
            gps_end: GPS end of the query interval.
            detector: Detector prefix (e.g. ``"H1"``).

        Returns:
            A list of ``(start, end)`` vetosegment tuples covering
            time windows with data-quality issues.
        """
        gwosc = _import_gwosc()

        vetosegments: SegmentList = []
        for flag_base in self.config.dq_flags:
            flag_name = f"{detector}_{flag_base}"
            try:
                segments_iter = gwosc.timeline.get_segments(
                    flag_name,
                    int(gps_start),
                    int(gps_end),
                    host=self._host,
                )
                for seg_start, seg_end in segments_iter:
                    vetosegments.append((float(seg_start), float(seg_end)))
            except (ValueError, KeyError, TypeError, OSError):
                continue

        return _merge_segments(vetosegments)

    def get_hardware_injection_segments(
        self,
        gps_start: float,
        gps_end: float,
    ) -> SegmentList:
        """Query hardware injection events and build vetosegments.

        Args:
            gps_start: GPS start of the query interval.
            gps_end: GPS end of the query interval.

        Returns:
            A list of ``(start, end)`` vetosegment tuples around
            hardware injection times.
        """
        gwosc = _import_gwosc()
        padding = self.config.event_padding

        try:
            dataset = gwosc.api.fetch_dataset_json(
                int(gps_start),
                int(gps_end + 1),
                host=self._host,
            )
        except (ValueError, OSError):
            return []

        events = dataset.get("events", {})
        vetosegments: SegmentList = []
        for _event_name, event_data in events.items():
            inj_bits = event_data.get("INJbits", 0)
            if not inj_bits:
                continue
            gps_time = event_data.get("GPStime")
            if gps_time is None:
                continue
            vetosegments.append((float(gps_time) - padding, float(gps_time) + padding))

        return _merge_segments(vetosegments)

    def compute_clean_segments(
        self,
        gps_start: float,
        gps_end: float,
        detectors: list[str],
    ) -> dict[str, SegmentList]:
        """Compute clean (analysis-ready) segments for each detector.

        Clean segments are the requested GPS range minus the union of:
        - GW event vetosegments (if any GW filter is active)
        - DQ vetosegments (if the data-quality filter is active)
        - Hardware injection segments (if enabled)

        Args:
            gps_start: GPS start of the requested interval.
            gps_end: GPS end of the requested interval.
            detectors: List of detector prefixes.

        Returns:
            A dictionary mapping each detector to a list of clean
            ``(start, end)`` segments.
        """
        gw_vetosegments: SegmentList = []
        if self.config.has_gw_filter:
            gw_vetosegments = self.get_gw_vetosegments(gps_start, gps_end)

        hw_inj_segments: SegmentList = []
        if self.config.exclude_hardware_injections and self.config.has_gw_filter:
            hw_inj_segments = self.get_hardware_injection_segments(gps_start, gps_end)

        combined_gw = _merge_segments(gw_vetosegments + hw_inj_segments)

        clean_by_detector: dict[str, SegmentList] = {}
        for detector in detectors:
            all_vetosegments = list(combined_gw)
            if self.config.has_dq_filter:
                dq_vetosegments = self.get_dq_vetosegments(gps_start, gps_end, detector)
                all_vetosegments.extend(dq_vetosegments)

            merged_vetos = _merge_segments(all_vetosegments)
            clean_by_detector[detector] = _invert_segments(merged_vetos, gps_start, gps_end)

        return clean_by_detector

    @property
    def _host(self) -> str:
        """Return the GWOSC host, defaulting to the public server."""
        return "https://gwosc.org"
