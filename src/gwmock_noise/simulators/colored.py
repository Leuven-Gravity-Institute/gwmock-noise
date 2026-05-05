"""Colored noise simulator with overlap-add stitching."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.simulators._spectral import load_spectral_series, normalize_spectral_reference
from gwmock_noise.simulators._stitching import OVERLAP_SIZE, WINDOW_SIZE, OverlapAddStitcher

PSD_WINDOW_ALPHA = 1e-3
MIN_TAPER_BINS = 2
FRAME_STEP = WINDOW_SIZE - OVERLAP_SIZE


def _tukey_window(length: int, alpha: float = PSD_WINDOW_ALPHA) -> np.ndarray:
    """Return a Tukey window without depending on SciPy."""
    if length < 1:
        raise ValueError("length must be positive.")
    if alpha <= 0:
        return np.ones(length, dtype=float)
    if alpha >= 1:
        return np.hanning(length)
    if length <= MIN_TAPER_BINS or alpha * length <= MIN_TAPER_BINS:
        return np.ones(length, dtype=float)

    x = np.linspace(0.0, 1.0, length)
    window = np.ones(length, dtype=float)
    leading = x < (alpha / 2.0)
    trailing = x >= (1.0 - alpha / 2.0)
    window[leading] = 0.5 * (1.0 + np.cos((2.0 * np.pi / alpha) * (x[leading] - alpha / 2.0)))
    window[trailing] = 0.5 * (1.0 + np.cos((2.0 * np.pi / alpha) * (x[trailing] - 1.0 + alpha / 2.0)))
    return window


class ColoredNoiseSimulator:
    """Generate colored detector noise from an input PSD."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
    ) -> None:
        """Initialize the simulator."""
        if psd_file is None and psd_schedule is None:
            raise ValueError("Either psd_file or psd_schedule must be provided.")
        if psd_file is not None and psd_schedule is not None:
            raise ValueError("psd_file and psd_schedule are mutually exclusive.")
        if psd_schedule is not None and not psd_schedule:
            raise ValueError("psd_schedule must contain at least one anchor.")
        if psd_schedule is not None:
            offsets = [float(gps_offset_seconds) for gps_offset_seconds, _ in psd_schedule]
            if offsets != sorted(offsets):
                raise ValueError("psd_schedule entries must be sorted by GPS offset.")
            if len(offsets) != len(set(offsets)):
                raise ValueError("psd_schedule entries must use distinct GPS offsets.")

        self.psd_file = normalize_spectral_reference(psd_file) if psd_file is not None else None
        self.psd_schedule = (
            [
                (float(gps_offset_seconds), normalize_spectral_reference(path))
                for gps_offset_seconds, path in psd_schedule
            ]
            if psd_schedule is not None
            else None
        )
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff

        self._rngs: dict[str, np.random.Generator] = {}
        self._stitcher = OverlapAddStitcher(self.detectors)
        self._generated_samples = 0
        self._psd_anchors: list[tuple[float, np.ndarray]] = []

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._configure_frequency_grid()

    @property
    def previous_strain(self) -> dict[str, np.ndarray]:
        """Expose continuity buffers for protocol-compatible state inspection."""
        return self._stitcher.previous_strain

    def _validate_runtime(
        self,
        *,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
    ) -> None:
        """Validate runtime arguments shared by initialization and generation."""
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if self.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be non-negative.")

        nyquist = sampling_frequency / 2.0
        high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist
        if high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")
        if high_frequency_cutoff > nyquist:
            raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

    def _configure_frequency_grid(self) -> None:
        """Configure the FFT grid and interpolated PSD."""
        self._delta_frequency = self.sampling_frequency / WINDOW_SIZE
        self._frequency_grid = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / self.sampling_frequency)
        self._high_frequency_cutoff = (
            self.high_frequency_cutoff if self.high_frequency_cutoff is not None else self.sampling_frequency / 2.0
        )
        self._frequency_mask = (self._frequency_grid >= self.low_frequency_cutoff) & (
            self._frequency_grid <= self._high_frequency_cutoff
        )
        if not np.any(self._frequency_mask):
            raise ValueError("The requested frequency range contains no simulation bins.")

        self._psd_anchors = self._load_psd_anchors()
        self._psd = self._interpolate_psd(0.0)

    def _load_psd_on_grid(self, psd_file: str | Path) -> np.ndarray:
        """Load one PSD file onto the simulator frequency grid."""
        psd = np.zeros_like(self._frequency_grid, dtype=float)
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        psd_frequencies, psd_values = load_spectral_series(psd_file, kind="PSD")
        interpolated_psd = np.interp(masked_frequencies, psd_frequencies, psd_values, left=0.0, right=0.0)
        psd[self._frequency_mask] = np.clip(interpolated_psd, a_min=0.0, a_max=None)
        psd[self._frequency_mask] *= _tukey_window(masked_frequencies.size)
        return psd

    def _load_psd_anchors(self) -> list[tuple[float, np.ndarray]]:
        """Load all configured PSD anchors onto the current simulator grid."""
        anchors = self.psd_schedule or [(0.0, self.psd_file)]
        return [
            (gps_offset_seconds, self._load_psd_on_grid(psd_path))
            for gps_offset_seconds, psd_path in anchors
            if psd_path is not None
        ]

    def _interpolate_psd(self, t: float) -> np.ndarray:
        """Interpolate the PSD schedule log-linearly at frame midpoint time ``t``."""
        if len(self._psd_anchors) == 1:
            return self._psd_anchors[0][1].copy()

        anchor_times = np.array([time for time, _ in self._psd_anchors], dtype=float)
        if t <= anchor_times[0]:
            return self._psd_anchors[0][1].copy()
        if t >= anchor_times[-1]:
            return self._psd_anchors[-1][1].copy()

        upper_index = int(np.searchsorted(anchor_times, t, side="right"))
        lower_time, lower_psd = self._psd_anchors[upper_index - 1]
        upper_time, upper_psd = self._psd_anchors[upper_index]
        weight = (t - lower_time) / (upper_time - lower_time)

        interpolated = np.zeros_like(self._frequency_grid, dtype=float)
        lower_masked = np.maximum(lower_psd[self._frequency_mask], np.finfo(float).tiny)
        upper_masked = np.maximum(upper_psd[self._frequency_mask], np.finfo(float).tiny)
        log_psd = ((1.0 - weight) * np.log(lower_masked)) + (weight * np.log(upper_masked))
        interpolated[self._frequency_mask] = np.exp(log_psd)
        return interpolated

    def _simulate(self, *, n_samples: int) -> dict[str, np.ndarray]:
        """Generate a prefix-consistent realization while updating the PSD per frame."""
        next_frame_midpoint = self._generated_samples if self.previous_strain else -FRAME_STEP

        def chunk_generator() -> dict[str, np.ndarray]:
            nonlocal next_frame_midpoint
            self._psd = self._interpolate_psd(next_frame_midpoint / self.sampling_frequency)
            next_frame_midpoint += FRAME_STEP
            return self._generate_realization_chunk()

        history = self.previous_strain
        if history:
            self._stitcher._validate_chunk_map(history)
            current_raw = {detector: history[detector].copy() for detector in self.detectors}
        else:
            warmup = chunk_generator()
            self._stitcher._validate_chunk_map(warmup)
            current_raw = {detector: warmup[detector].copy() for detector in self.detectors}

        raw_buffers = {detector: current_raw[detector].copy() for detector in self.detectors}
        emitted_segments = {detector: [] for detector in self.detectors}
        produced_samples = 0

        while produced_samples < n_samples:
            next_raw = chunk_generator()
            self._stitcher._validate_chunk_map(next_raw)

            for detector in self.detectors:
                blended_overlap = (
                    (current_raw[detector][-OVERLAP_SIZE:] * self._stitcher._window_out)
                    + (next_raw[detector][:OVERLAP_SIZE] * self._stitcher._window_in)
                ) / self._stitcher._blend_norm
                emitted_segments[detector].append(blended_overlap)
                raw_buffers[detector] = np.concatenate((raw_buffers[detector], next_raw[detector][OVERLAP_SIZE:]))
                current_raw[detector] = next_raw[detector].copy()

            produced_samples += FRAME_STEP

        realization = {
            detector: np.concatenate(segments)[:n_samples] for detector, segments in emitted_segments.items()
        }
        self.previous_strain.clear()
        self.previous_strain.update(
            {detector: raw_buffers[detector][n_samples : n_samples + WINDOW_SIZE].copy() for detector in self.detectors}
        )
        return realization

    def _initialize_generators(self, seed: int | None) -> None:
        """Initialize per-detector random-number generators."""
        seed_sequence = np.random.SeedSequence() if seed is None else np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(len(self.detectors))
        self._rngs = {
            detector: np.random.default_rng(child_sequence)
            for detector, child_sequence in zip(self.detectors, child_sequences, strict=True)
        }

    def _generate_single_realization(self, detector: str) -> np.ndarray:
        """Generate one colored-noise chunk for a single detector."""
        rng = self._rngs[detector]
        n_frequencies = int(np.count_nonzero(self._frequency_mask))
        white_noise = (rng.standard_normal(n_frequencies) + 1j * rng.standard_normal(n_frequencies)) / np.sqrt(2.0)

        frequency_series = np.zeros(self._frequency_grid.size, dtype=np.complex128)
        frequency_series[self._frequency_mask] = white_noise * np.sqrt(
            self._psd[self._frequency_mask] * 0.5 / self._delta_frequency
        )
        return np.fft.irfft(frequency_series, n=WINDOW_SIZE) * self._delta_frequency * WINDOW_SIZE

    def _generate_realization_chunk(self) -> dict[str, np.ndarray]:
        """Generate one colored-noise chunk for every configured detector."""
        return {detector: self._generate_single_realization(detector) for detector in self.detectors}

    def reset(self) -> None:
        """Clear continuity and RNG state."""
        self._stitcher.reset()
        self._rngs = {}
        self._generated_samples = 0
        if self._psd_anchors:
            self._psd = self._psd_anchors[0][1].copy()

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector colored noise."""
        runtime_detectors = list(detectors)
        self._validate_runtime(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=runtime_detectors,
        )

        runtime_changed = (
            sampling_frequency != self.sampling_frequency
            or runtime_detectors != self.detectors
            or self._high_frequency_cutoff
            != (self.high_frequency_cutoff if self.high_frequency_cutoff is not None else sampling_frequency / 2.0)
        )

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors

        if runtime_changed:
            self._stitcher.configure_detectors(self.detectors)
            self.reset()
            self._configure_frequency_grid()

        if seed is not None:
            self.seed = seed
            self.reset()

        if not self._rngs:
            self._initialize_generators(self.seed)

        n_samples = round(duration * sampling_frequency)
        realization = self._simulate(n_samples=n_samples)
        self._generated_samples += n_samples
        return realization

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield colored-noise chunks lazily while preserving simulator state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
        return {
            "implementation": "colored",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "colored_noise": {
                "psd_file": str(self.psd_file) if self.psd_file is not None else None,
                "psd_schedule": [
                    {"gps_offset_seconds": gps_offset_seconds, "psd_file": str(psd_path)}
                    for gps_offset_seconds, psd_path in (self.psd_schedule or [])
                ],
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "window_size": WINDOW_SIZE,
                "overlap_size": OVERLAP_SIZE,
            },
        }


TimeVaryingColoredNoiseSimulator = ColoredNoiseSimulator
