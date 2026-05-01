"""Correlated noise simulator with shared overlap-add stitching."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators._stitching import OVERLAP_SIZE, WINDOW_SIZE, OverlapAddStitcher
from gwmock_noise.simulators.colored import _tukey_window

DETECTOR_PAIR_SIZE = 2


def parse_csd_file_map(csd_files: dict[str, Path] | None) -> dict[tuple[str, str], Path]:
    """Convert config-string detector pairs into tuple keys."""
    if not csd_files:
        return {}

    parsed: dict[tuple[str, str], Path] = {}
    for pair_key, file_path in csd_files.items():
        detectors = pair_key.split("-")
        if len(detectors) != DETECTOR_PAIR_SIZE or not all(detectors):
            raise ValueError("csd_files keys must use the 'DET1-DET2' format.")

        detector_a, detector_b = tuple(sorted(detectors))
        if detector_a == detector_b:
            raise ValueError("csd_files keys must reference two distinct detectors.")

        normalized_key = (detector_a, detector_b)
        if normalized_key in parsed:
            raise ValueError(f"Duplicate CSD file mapping for detector pair {detector_a}-{detector_b}.")

        parsed[normalized_key] = Path(file_path)

    return parsed


class CorrelatedNoiseSimulator:
    """Generate correlated detector noise from PSD and CSD inputs."""

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_files: dict[str, str | Path],
        csd_files: dict[tuple[str, str], str | Path] | None = None,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        regularization_epsilon: float = 1.0e-12,
    ) -> None:
        """Initialize the simulator."""
        self.psd_files = {detector: Path(path) for detector, path in psd_files.items()}
        self.detectors = list(detectors) if detectors is not None else list(self.psd_files)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.regularization_epsilon = regularization_epsilon

        self._stitcher = OverlapAddStitcher(self.detectors)
        self._rng: np.random.Generator | None = None

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._normalized_csd_files = self._normalize_csd_files(csd_files or {})
        self._configure_frequency_grid()
        self._configure_spectral_factors()

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
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicates.")
        if self.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be non-negative.")
        if self.regularization_epsilon <= 0:
            raise ValueError("regularization_epsilon must be greater than zero.")

        nyquist = sampling_frequency / 2.0
        high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist
        if high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")
        if high_frequency_cutoff > nyquist:
            raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

    def _normalize_csd_files(
        self,
        csd_files: dict[tuple[str, str], str | Path],
    ) -> dict[tuple[str, str], Path]:
        """Validate and normalize pairwise CSD inputs."""
        detector_set = set(self.detectors)
        if set(self.psd_files) != detector_set:
            raise ValueError("psd_files keys must exactly match detectors.")

        normalized: dict[tuple[str, str], Path] = {}
        for pair, file_path in csd_files.items():
            if len(pair) != DETECTOR_PAIR_SIZE:
                raise ValueError("Each csd_files key must contain exactly two detector names.")

            detector_a, detector_b = tuple(sorted(pair))
            if detector_a == detector_b:
                raise ValueError("CSD detector pairs must reference two distinct detectors.")
            if detector_a not in detector_set or detector_b not in detector_set:
                raise ValueError("CSD detector pairs must reference configured detectors.")

            normalized_key = (detector_a, detector_b)
            if normalized_key in normalized:
                raise ValueError(f"Duplicate CSD file mapping for detector pair {detector_a}-{detector_b}.")
            normalized[normalized_key] = Path(file_path)

        return normalized

    def _configure_frequency_grid(self) -> None:
        """Configure the FFT grid and band mask."""
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

    def _interpolate_psd(self, detector: str, masked_frequencies: np.ndarray, taper: np.ndarray) -> np.ndarray:
        """Interpolate a detector PSD onto the FFT grid."""
        frequencies, values = load_spectral_series(self.psd_files[detector], kind="PSD")
        interpolated = np.interp(masked_frequencies, frequencies, values, left=0.0, right=0.0)
        return np.clip(interpolated, a_min=0.0, a_max=None) * taper

    def _interpolate_csd(
        self,
        pair: tuple[str, str],
        masked_frequencies: np.ndarray,
        taper: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a detector-pair CSD onto the FFT grid."""
        frequencies, values = load_spectral_series(self._normalized_csd_files[pair], kind="CSD", complex_values=True)
        real = np.interp(masked_frequencies, frequencies, values.real, left=0.0, right=0.0)
        imag = np.interp(masked_frequencies, frequencies, values.imag, left=0.0, right=0.0)
        return (real + 1j * imag) * taper

    def _configure_spectral_factors(self) -> None:
        """Construct regularized spectral factors on the FFT grid."""
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        taper = _tukey_window(masked_frequencies.size)

        self._detector_index = {detector: index for index, detector in enumerate(self.detectors)}
        self._psd = {
            detector: self._interpolate_psd(detector, masked_frequencies, taper) for detector in self.detectors
        }
        self._csd = {
            pair: self._interpolate_csd(pair, masked_frequencies, taper)
            for pair in combinations(sorted(self.detectors), 2)
            if pair in self._normalized_csd_files
        }

        n_detectors = len(self.detectors)
        n_frequencies = masked_frequencies.size
        self._cholesky_factors = np.zeros((n_frequencies, n_detectors, n_detectors), dtype=np.complex128)

        for frequency_index in range(n_frequencies):
            spectral_matrix = np.zeros((n_detectors, n_detectors), dtype=np.complex128)
            for detector, psd in self._psd.items():
                detector_index = self._detector_index[detector]
                spectral_matrix[detector_index, detector_index] = psd[frequency_index]

            for pair, csd in self._csd.items():
                detector_a, detector_b = pair
                index_a = self._detector_index[detector_a]
                index_b = self._detector_index[detector_b]
                spectral_matrix[index_a, index_b] = csd[frequency_index]
                spectral_matrix[index_b, index_a] = np.conj(csd[frequency_index])

            self._cholesky_factors[frequency_index] = self._regularized_cholesky(
                spectral_matrix * (0.5 / self._delta_frequency)
            )

    def _regularized_cholesky(self, spectral_matrix: np.ndarray) -> np.ndarray:
        """Return a numerically stable Cholesky-like factor."""
        hermitian_matrix = 0.5 * (spectral_matrix + spectral_matrix.conj().T)
        diagonal_scale = max(float(np.max(np.real(np.diag(hermitian_matrix)))), 1.0)
        epsilon = self.regularization_epsilon * diagonal_scale
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_matrix)))

        regularized = hermitian_matrix
        if minimum_eigenvalue < epsilon:
            regularized = regularized + np.eye(hermitian_matrix.shape[0]) * (epsilon - minimum_eigenvalue)

        try:
            return np.linalg.cholesky(regularized)
        except np.linalg.LinAlgError:
            diagonal = np.clip(np.real(np.diag(hermitian_matrix)), a_min=0.0, a_max=None)
            return np.diag(np.sqrt(diagonal + epsilon))

    def _initialize_generator(self, seed: int | None) -> None:
        """Initialize the shared random-number generator."""
        self._rng = np.random.default_rng(seed)

    def _generate_realization_chunk(self) -> dict[str, np.ndarray]:
        """Generate one correlated noise chunk for all detectors."""
        if self._rng is None:
            raise RuntimeError("Random number generator not initialized.")

        n_frequencies = int(np.count_nonzero(self._frequency_mask))
        white_noise = (
            self._rng.standard_normal((n_frequencies, len(self.detectors)))
            + 1j * self._rng.standard_normal((n_frequencies, len(self.detectors)))
        ) / np.sqrt(2.0)
        colored_noise = np.einsum("fij,fj->fi", self._cholesky_factors, white_noise)

        frequency_series = np.zeros((len(self.detectors), self._frequency_grid.size), dtype=np.complex128)
        frequency_series[:, self._frequency_mask] = colored_noise.T
        time_series = np.fft.irfft(frequency_series, n=WINDOW_SIZE, axis=1) * self._delta_frequency * WINDOW_SIZE
        return {detector: time_series[self._detector_index[detector]].copy() for detector in self.detectors}

    def reset(self) -> None:
        """Clear continuity and RNG state."""
        self._stitcher.reset()
        self._rng = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate correlated per-detector noise."""
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
            self._normalized_csd_files = self._normalize_csd_files(self._normalized_csd_files)
            self._configure_frequency_grid()
            self._configure_spectral_factors()

        if seed is not None:
            self.seed = seed
            self.reset()

        if self._rng is None:
            self._initialize_generator(self.seed)

        n_samples = round(duration * sampling_frequency)
        return self._stitcher.stitch(n_samples=n_samples, chunk_generator=self._generate_realization_chunk)

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield correlated-noise chunks lazily while preserving simulator state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
        return {
            "implementation": "correlated",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "correlated_noise": {
                "psd_files": {detector: str(path) for detector, path in self.psd_files.items()},
                "csd_files": {
                    f"{detector_a}-{detector_b}": str(path)
                    for (detector_a, detector_b), path in self._normalized_csd_files.items()
                },
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "window_size": WINDOW_SIZE,
                "overlap_size": OVERLAP_SIZE,
                "regularization_epsilon": self.regularization_epsilon,
            },
        }
