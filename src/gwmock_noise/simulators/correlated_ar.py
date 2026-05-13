"""Correlated moving-average noise simulator with persistent innovation state."""

from __future__ import annotations

import time
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.colored import _tukey_window
from gwmock_noise.simulators.correlated import parse_csd_file_map

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

DEFAULT_VMA_ORDER = 256
DEFAULT_BLOCK_SIZE = 65_536
DEFAULT_REGULARIZATION_EPSILON = 1.0e-12
DETECTOR_PAIR_SIZE = 2
MIN_SPECTRAL_POINTS = 2


class CorrelatedARNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate correlated detector noise with a truncated VMA representation."""

    simulator_name = "correlated_ar"

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_files: dict[str, str | Path],
        csd_files: dict[str, str | Path] | dict[tuple[str, str], str | Path] | None = None,
        order: int = DEFAULT_VMA_ORDER,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        regularization_epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
    ) -> None:
        """Initialize the simulator and fit truncated VMA taps once."""
        self.psd_files = {detector: Path(path) for detector, path in psd_files.items()}
        self.order = order
        self.detectors = list(detectors) if detectors is not None else list(self.psd_files)
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.block_size = block_size
        self.regularization_epsilon = regularization_epsilon

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)

        self._normalized_csd_files = self._normalize_csd_files(csd_files or {})
        self._rng: np.random.Generator | None = None
        self._state = np.zeros((self.order, len(self.detectors)), dtype=float)
        self._fit_time_seconds = 0.0
        self._fit_grid_size = 0
        self._high_frequency_cutoff = 0.0
        self._detector_index = {detector: index for index, detector in enumerate(self.detectors)}
        self._filter_taps = np.zeros((self.order + 1, len(self.detectors), len(self.detectors)), dtype=float)

        self._fit_model()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> CorrelatedARNoiseSimulator:
        """Construct a correlated-AR simulator from one component definition."""
        options = dict(component.options)
        psd_files = options.pop("psd_files", None)
        csd_files = options.pop("csd_files", None)
        return cls(
            psd_files=psd_files or {},
            csd_files=csd_files,
            detectors=config.detectors,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
            **options,
        )

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
        if self.order < 0:
            raise ValueError("order must be non-negative.")
        if self.block_size < 1:
            raise ValueError("block_size must be greater than zero.")
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
        csd_files: dict[str, str | Path] | dict[tuple[str, str], str | Path],
    ) -> dict[tuple[str, str], Path]:
        """Validate and normalize pairwise CSD inputs."""
        detector_set = set(self.detectors)
        if set(self.psd_files) != detector_set:
            raise ValueError("psd_files keys must exactly match detectors.")

        if not csd_files:
            return {}

        first_key = next(iter(csd_files))
        if isinstance(first_key, str):
            normalized = parse_csd_file_map(csd_files)  # type: ignore[arg-type]
        else:
            normalized = {}
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

        for detector_a, detector_b in normalized:
            if detector_a not in detector_set or detector_b not in detector_set:
                raise ValueError("CSD detector pairs must reference configured detectors.")

        return {pair: Path(path) for pair, path in normalized.items()}

    def _regularized_cholesky(self, spectral_matrix: np.ndarray) -> np.ndarray:
        """Return a numerically stable spectral factor."""
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

    def _load_fit_grid_size(self) -> int:
        """Determine a uniform FFT size covering all spectral inputs."""
        max_points = 0
        for psd_path in self.psd_files.values():
            frequencies, _ = load_spectral_series(psd_path, kind="PSD")
            max_points = max(max_points, frequencies.size)
        for csd_path in self._normalized_csd_files.values():
            frequencies, _ = load_spectral_series(csd_path, kind="CSD", complex_values=True)
            max_points = max(max_points, frequencies.size)

        if max_points < MIN_SPECTRAL_POINTS:
            raise ValueError("Spectral inputs must contain at least two frequency samples.")
        return max(2 * (max_points - 1), 8 * max(self.order, 1))

    def _interpolate_psd(self, detector: str, masked_frequencies: np.ndarray, taper: np.ndarray) -> np.ndarray:
        """Interpolate a detector PSD onto the fit grid."""
        frequencies, values = load_spectral_series(self.psd_files[detector], kind="PSD")
        interpolated = np.interp(masked_frequencies, frequencies, values, left=0.0, right=0.0)
        return np.clip(interpolated, a_min=0.0, a_max=None) * taper

    def _interpolate_csd(
        self,
        pair: tuple[str, str],
        masked_frequencies: np.ndarray,
        taper: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a detector-pair CSD onto the fit grid."""
        frequencies, values = load_spectral_series(self._normalized_csd_files[pair], kind="CSD", complex_values=True)
        real = np.interp(masked_frequencies, frequencies, values.real, left=0.0, right=0.0)
        imag = np.interp(masked_frequencies, frequencies, values.imag, left=0.0, right=0.0)
        return (real + 1j * imag) * taper

    def _fit_model(self) -> None:
        """Fit truncated VMA filter taps from PSD/CSD inputs."""
        fit_start = time.perf_counter()

        self._detector_index = {detector: index for index, detector in enumerate(self.detectors)}
        self._fit_grid_size = self._load_fit_grid_size()
        nyquist = self.sampling_frequency / 2.0
        self._high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist

        frequency_grid = np.fft.rfftfreq(self._fit_grid_size, d=1.0 / self.sampling_frequency)
        frequency_mask = (frequency_grid >= self.low_frequency_cutoff) & (frequency_grid <= self._high_frequency_cutoff)
        if not np.any(frequency_mask):
            raise ValueError("The requested frequency range contains no simulation bins.")

        masked_frequencies = frequency_grid[frequency_mask]
        taper = _tukey_window(masked_frequencies.size)
        n_detectors = len(self.detectors)
        spectral_factors = np.zeros((frequency_grid.size, n_detectors, n_detectors), dtype=np.complex128)

        psd = {detector: self._interpolate_psd(detector, masked_frequencies, taper) for detector in self.detectors}
        csd = {
            pair: self._interpolate_csd(pair, masked_frequencies, taper)
            for pair in combinations(sorted(self.detectors), 2)
            if pair in self._normalized_csd_files
        }

        for masked_index, _ in enumerate(masked_frequencies):
            spectral_matrix = np.zeros((n_detectors, n_detectors), dtype=np.complex128)
            for detector, values in psd.items():
                detector_index = self._detector_index[detector]
                spectral_matrix[detector_index, detector_index] = values[masked_index]
            for pair, values in csd.items():
                detector_a, detector_b = pair
                index_a = self._detector_index[detector_a]
                index_b = self._detector_index[detector_b]
                spectral_matrix[index_a, index_b] = values[masked_index]
                spectral_matrix[index_b, index_a] = np.conj(values[masked_index])

            grid_index = np.flatnonzero(frequency_mask)[masked_index]
            spectral_factors[grid_index] = self._regularized_cholesky(spectral_matrix * (self.sampling_frequency / 2.0))

        taps = np.fft.irfft(spectral_factors, n=self._fit_grid_size, axis=0)
        self._filter_taps = np.asarray(taps[: self.order + 1], dtype=float)
        self._fit_time_seconds = time.perf_counter() - fit_start
        self._state = np.zeros((self.order, n_detectors), dtype=float)

    def _initialize_generator(self, seed: int | None) -> None:
        """Initialize the innovation generator."""
        self._rng = np.random.default_rng(seed)

    def _generate_block(self, n_samples: int) -> np.ndarray:
        """Generate one multivariate correlated-noise block."""
        if self._rng is None:
            raise RuntimeError("Random number generator not initialized.")

        innovations = self._rng.standard_normal((n_samples, len(self.detectors)))
        full_innovations = np.concatenate((self._state, innovations), axis=0)
        windows = sliding_window_view(full_innovations, window_shape=self.order + 1, axis=0)
        windows = np.moveaxis(windows, -1, 1)
        block = np.einsum("tki,koi->to", windows, self._filter_taps[::-1], optimize=True)

        if self.order > 0:
            self._state = full_innovations[-self.order :].copy()
        else:
            self._state = np.zeros((0, len(self.detectors)), dtype=float)
        return block

    def reset(self) -> None:
        """Clear innovation state and RNG."""
        self._rng = None
        self._state = np.zeros((self.order, len(self.detectors)), dtype=float)

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate correlated per-detector noise with stateful continuity."""
        runtime_detectors = list(detectors)
        self._validate_runtime(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=runtime_detectors,
        )

        if set(runtime_detectors) != set(self.detectors):
            raise ValueError(
                "Changing the detector network to a subset or superset is unsupported; "
                "use the same detector names as at initialization (reordering is allowed)."
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
            self._normalized_csd_files = self._normalize_csd_files(self._normalized_csd_files)
            self._fit_model()
            self.reset()

        if seed is not None or self._rng is None:
            self.seed = seed if seed is not None else self.seed
            self._initialize_generator(self.seed)

        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        blocks = []
        remaining = n_samples
        while remaining > 0:
            chunk_size = min(self.block_size, remaining)
            blocks.append(self._generate_block(chunk_size))
            remaining -= chunk_size

        realization = np.concatenate(blocks, axis=0)
        return {detector: realization[:, self._detector_index[detector]].copy() for detector in self.detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield correlated AR chunks lazily while preserving innovation state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the fitted VMA model."""
        return {
            "implementation": "correlated_autoregressive",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "correlated_autoregressive_noise": {
                "psd_files": {detector: str(path) for detector, path in self.psd_files.items()},
                "csd_files": {
                    f"{detector_a}-{detector_b}": str(path)
                    for (detector_a, detector_b), path in self._normalized_csd_files.items()
                },
                "order": self.order,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "block_size": self.block_size,
                "fit_grid_size": self._fit_grid_size,
                "fit_time_seconds": self._fit_time_seconds,
                "regularization_epsilon": self.regularization_epsilon,
            },
        }
