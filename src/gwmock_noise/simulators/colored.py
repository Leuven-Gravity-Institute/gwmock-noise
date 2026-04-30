"""Colored noise simulator with overlap-add stitching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

WINDOW_SIZE = 2048
OVERLAP_SIZE = WINDOW_SIZE // 2
PSD_WINDOW_ALPHA = 1e-3
PSD_COLUMNS = 2
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

        self.previous_strain: dict[str, np.ndarray] = {}
        self._rngs: dict[str, np.random.Generator] = {}
        self._window_out = np.cos(np.linspace(0.0, np.pi / 2.0, OVERLAP_SIZE))
        self._window_in = np.sin(np.linspace(0.0, np.pi / 2.0, OVERLAP_SIZE))
        self._blend_norm = np.sqrt(self._window_out**2 + self._window_in**2)

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._configure_frequency_grid()

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
        psd_data = self._load_spectral_data(self.psd_file)
        interpolated_psd = np.interp(masked_frequencies, psd_data[:, 0], psd_data[:, 1], left=0.0, right=0.0)
        self._psd[self._frequency_mask] = np.clip(interpolated_psd, a_min=0.0, a_max=None)
        self._psd[self._frequency_mask] *= _tukey_window(masked_frequencies.size)

    def _load_spectral_data(self, file_path: str | Path) -> np.ndarray:
        """Load a PSD array from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PSD file not found: {path}")

        if path.suffix == ".npy":
            data = np.load(path)
        elif path.suffix == ".txt":
            data = np.loadtxt(path)
        elif path.suffix == ".csv":
            data = np.loadtxt(path, delimiter=",")
        else:
            raise ValueError(f"Unsupported PSD file format: {path.suffix}. Use .npy, .txt, or .csv.")

        if data.ndim != PSD_COLUMNS or data.shape[1] != PSD_COLUMNS:
            raise ValueError("PSD file must have shape (N, 2).")

        order = np.argsort(data[:, 0])
        return np.asarray(data[order], dtype=float)

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

    def reset(self) -> None:
        """Clear continuity and RNG state."""
        self.previous_strain = {}
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
            self.reset()
            self._configure_frequency_grid()

        if seed is not None:
            self.seed = seed
            self.reset()

        if not self._rngs:
            self._initialize_generators(self.seed)

        n_samples = round(duration * sampling_frequency)
        realizations: dict[str, np.ndarray] = {}

        for detector in self.detectors:
            history = self.previous_strain.get(detector)
            if history is None:
                history = self._generate_single_realization(detector)
            elif history.shape != (WINDOW_SIZE,):
                raise ValueError(f"previous_strain for {detector} must have shape ({WINDOW_SIZE},).")

            raw_strain_buffer = history.copy()
            strain_buffer = raw_strain_buffer.copy()
            strain_buffer[-OVERLAP_SIZE:] *= self._window_out

            while strain_buffer.size - WINDOW_SIZE < n_samples:
                raw_new_strain = self._generate_single_realization(detector)
                new_strain = raw_new_strain.copy()
                new_strain[:OVERLAP_SIZE] *= self._window_in
                new_strain[-OVERLAP_SIZE:] *= self._window_out
                strain_buffer[-OVERLAP_SIZE:] = (
                    strain_buffer[-OVERLAP_SIZE:] + new_strain[:OVERLAP_SIZE]
                ) / self._blend_norm
                strain_buffer = np.concatenate((strain_buffer, new_strain[OVERLAP_SIZE:]))
                raw_strain_buffer = np.concatenate((raw_strain_buffer, raw_new_strain[OVERLAP_SIZE:]))

            realizations[detector] = strain_buffer[WINDOW_SIZE : WINDOW_SIZE + n_samples].copy()
            self.previous_strain[detector] = raw_strain_buffer[n_samples : n_samples + WINDOW_SIZE].copy()

        return realizations

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
