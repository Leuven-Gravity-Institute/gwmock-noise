"""Real-noise simulator backed by GWOSC strain data.

Provides a :class:`GwoscNoiseSimulator` that satisfies the
``NoiseSimulator`` protocol, allowing GWOSC-fetched noise to be used
interchangeably with the built-in synthetic simulators.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher
from gwmock_noise.gwosc.models import GwoscNoiseConfig


class GwoscNoiseSimulator:
    """Real-noise simulator that fetches strain data from GWOSC.

    Implements the ``NoiseSimulator`` protocol so it can be used
    interchangeably with synthetic simulators like
    ``ColoredNoiseSimulator`` and ``CorrelatedNoiseSimulator``.

    Fetches real detector strain from the Gravitational-Wave Open
    Science Centre and applies user-configured filters to exclude
    segments containing GW signals or data-quality issues.

    When ``cache_dir`` is set in the config, downloaded HDF5 files
    are saved locally and reused — avoiding repeated downloads for
    the same GPS interval.

    Attributes:
        duration: Duration of the configured GPS interval (seconds).
        sampling_frequency: Sampling frequency in Hz.
        detectors: List of detector prefixes.
        seed: Always ``None`` (real noise has no random seed).
        config: The underlying GWOSC configuration.
    """

    def __init__(self, config: GwoscNoiseConfig) -> None:
        """Initialize the real-noise simulator.

        Args:
            config: Configuration specifying GPS range, detectors,
                sample rate, filtering options, and optional cache
                directory.
        """
        self.config = config
        self._fetcher = GwoscNoiseFetcher(config)

    @property
    def duration(self) -> float:
        """Return the total duration of the configured GPS interval."""
        return self.config.duration

    @property
    def sampling_frequency(self) -> float:
        """Return the sampling frequency in Hz."""
        return self.config.sample_rate

    @property
    def detectors(self) -> list[str]:
        """Return the list of detector prefixes."""
        return list(self.config.detectors)

    @property
    def seed(self) -> None:
        """Return ``None`` — real noise has no controllable random seed."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the simulator and its configuration.

        Returns:
            A dictionary with the simulator implementation name,
            GPS range, sample rate, detectors, filter configuration,
            and cache status.
        """
        return {
            "implementation": "gwosc_real_noise",
            "gps_start": self.config.gps_start,
            "gps_end": self.config.gps_end,
            "sample_rate": self.config.sample_rate,
            "detectors": list(self.config.detectors),
            "filters": {
                "filter_types": [ft.value for ft in self.config.filters.filter_types],
                "far_threshold": self.config.filters.far_threshold,
                "event_padding": self.config.filters.event_padding,
                "dq_flags": list(self.config.filters.dq_flags),
            },
            "cache_dir": str(self.config.cache_dir) if self.config.cache_dir else None,
            "host": self.config.host,
        }

    def check_availability(self, detectors: list[str] | None = None) -> dict[str, bool]:
        """Probe GWOSC for per-detector strain-data availability.

        Convenience passthrough to :meth:`GwoscNoiseFetcher.check_availability`.
        Use this as a pre-flight check: a detector may have a fully clean
        interval (no vetoes) yet have no published strain data, in which
        case :meth:`generate` would fail with a clear error.

        Args:
            detectors: Detectors to probe. Defaults to all configured
                detectors when ``None``.

        Returns:
            A dictionary mapping each requested detector to ``True`` if
            GWOSC has data covering the interval, ``False`` otherwise.
        """
        return self._fetcher.check_availability(detectors)

    def _validate_request(self, sampling_frequency: float, detectors: list[str]) -> None:
        """Validate a generation request against the configured values.

        Args:
            sampling_frequency: Requested sampling frequency.
            detectors: Requested detector list.

        Raises:
            ValueError: If ``sampling_frequency`` does not match the
                configured value or if ``detectors`` are not a subset
                of the configured detectors.
        """
        if sampling_frequency != self.config.sample_rate:
            raise ValueError(
                f"sampling_frequency {sampling_frequency} does not match "
                f"configured sample_rate {self.config.sample_rate}."
            )
        if not set(detectors).issubset(set(self.config.detectors)):
            raise ValueError(
                f"Requested detectors {detectors} are not a subset of configured detectors {self.config.detectors}."
            )

    def generate_segments(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Fetch clean noise and return per-detector lists of contiguous segments.

        Unlike :meth:`generate`, this preserves the gap structure of the
        real data: each clean segment — the spans left after excluding GW
        events and data-quality vetoes — is returned as its own array, in
        GPS-time order. Adjacent segments are **not** contiguous in time,
        so callers must treat each segment independently; concatenating
        them would splice non-adjacent samples together and introduce
        discontinuities at the joins (which manifest as sharp transients
        after whitening).

        Args:
            duration: Requested duration (ignored; the GPS interval from
                the config determines the available data).
            sampling_frequency: Requested sampling frequency (must match
                the configured ``sample_rate``).
            detectors: Requested detector list (must be a subset of the
                configured detectors).
            seed: Ignored for real noise.

        Returns:
            A dictionary mapping each detector to a list of 1-D numpy
            arrays, one per clean segment, ordered by GPS time.

        Raises:
            ValueError: If ``sampling_frequency`` does not match the
                configured value, if ``detectors`` are not a subset, or
                if a detector has no clean data.
        """
        self._validate_request(sampling_frequency, detectors)

        clean_data = self._fetcher.fetch_clean(detectors)

        result: dict[str, list[np.ndarray]] = {}
        for detector in detectors:
            segments = clean_data.get(detector, [])
            if not segments:
                raise ValueError(f"No clean data for detector {detector}.")
            result[detector] = [np.asarray(ts.value) for ts in segments]

        return result

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Fetch clean noise and return the longest contiguous segment per detector.

        Real GWOSC noise is fragmented by the excision of GW events and
        data-quality vetoes, so the clean data generally consists of
        several non-contiguous spans. To honour the ``NoiseSimulator``
        contract — a single, gap-free array per detector — this returns
        the **longest** contiguous clean segment for each detector.

        Concatenating the fragments instead (the historical behaviour)
        splices non-adjacent samples together, creating discontinuities
        that show up as sharp transients after whitening. Use
        :meth:`generate_segments` to obtain every clean segment
        separately when the full clean data is required.

        Args:
            duration: Requested duration (ignored; the longest available
                contiguous clean segment determines the output length).
            sampling_frequency: Requested sampling frequency (must match
                the configured ``sample_rate``).
            detectors: Requested detector list (must be a subset of the
                configured detectors).
            seed: Ignored for real noise.

        Returns:
            A dictionary mapping each detector to a 1-D numpy array
            holding its longest contiguous clean segment.

        Raises:
            ValueError: If ``sampling_frequency`` does not match the
                configured value, if ``detectors`` are not a subset, or
                if a detector has no clean data.
        """
        segments_by_detector = self.generate_segments(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )
        return {detector: max(segments, key=len) for detector, segments in segments_by_detector.items()}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield clean-noise chunks lazily.

        Fetches the longest contiguous clean segment once (see
        :meth:`generate`) and yields it in chunks of ``chunk_duration``
        seconds. Chunking the longest contiguous span keeps every chunk
        free of the discontinuities that splicing fragments would
        introduce.

        Args:
            chunk_duration: Duration of each yielded chunk in seconds.
            sampling_frequency: Requested sampling frequency (must
                match the configured ``sample_rate``).
            detectors: Requested detector list (must be a subset of
                the configured detectors).
            seed: Ignored for real noise.

        Yields:
            Per-detector strain arrays for each chunk.
        """
        full_data = self.generate(
            duration=self.config.duration,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
            seed=seed,
        )

        chunk_samples = round(chunk_duration * sampling_frequency)
        if chunk_samples <= 0:
            raise ValueError(f"chunk_duration {chunk_duration} yields zero samples.")

        total_samples = min(len(data) for data in full_data.values())
        offset = 0
        while offset < total_samples:
            end = min(offset + chunk_samples, total_samples)
            yield {detector: data[offset:end] for detector, data in full_data.items()}
            offset = end
