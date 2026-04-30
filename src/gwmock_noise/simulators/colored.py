"""Colored noise simulator with overlap-add stitching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators._stitching import OVERLAP_SIZE, WINDOW_SIZE, OverlapAddStitcher

PSD_WINDOW_ALPHA = 1e-3
MIN_TAPER_BINS = 2


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
        psd_file: str | Path,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
    ) -> None:
        """Initialize the simulator."""
        self.psd_file = Path(psd_file)
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff

        self._rngs: dict[str, np.random.Generator] = {}
        self._stitcher = OverlapAddStitcher(self.detectors)

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

        self._psd = np.zeros_like(self._frequency_grid, dtype=float)
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        psd_frequencies, psd_values = load_spectral_series(self.psd_file, kind="PSD")
        interpolated_psd = np.interp(masked_frequencies, psd_frequencies, psd_values, left=0.0, right=0.0)
        self._psd[self._frequency_mask] = np.clip(interpolated_psd, a_min=0.0, a_max=None)
        self._psd[self._frequency_mask] *= _tukey_window(masked_frequencies.size)

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
        return self._stitcher.stitch(n_samples=n_samples, chunk_generator=self._generate_realization_chunk)

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
                "psd_file": str(self.psd_file),
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "window_size": WINDOW_SIZE,
                "overlap_size": OVERLAP_SIZE,
            },
        }
