"""Correlated noise simulator with shared overlap-add stitching."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators._spectral import load_spectral_series, median_frequency_spacing
from gwmock_noise.simulators._stitching import (
    DEFAULT_WINDOW_DURATION,
    OverlapAddStitcher,
    resolve_window_sizes,
    warn_if_underresolved,
)
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.colored import _resolve_taper_alpha, _tukey_window
from gwmock_noise.spectral import (
    build_spectral_covariance_from_files,
    regularized_cholesky,
    simulate_spectral_covariance_chunk,
)
from gwmock_noise.utils.log import LOGGER_NAME

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

logger = logging.getLogger(LOGGER_NAME)

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


class CorrelatedNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate correlated detector noise from PSD and CSD inputs."""

    simulator_name = "correlated"

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
        window_duration: float = DEFAULT_WINDOW_DURATION,
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
        self.window_duration = window_duration

        self._rng: np.random.Generator | None = None

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._window_size, self._overlap_size = resolve_window_sizes(window_duration, sampling_frequency)
        self._stitcher = OverlapAddStitcher(
            self.detectors,
            window_size=self._window_size,
            overlap_size=self._overlap_size,
        )
        self._normalized_csd_files = self._normalize_csd_files(csd_files or {})
        self._configure_frequency_grid()
        self._configure_spectral_factors()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> CorrelatedNoiseSimulator:
        """Construct a correlated-noise simulator from one component definition."""
        options = dict(component.options)
        psd_files = options.pop("psd_files", None)
        csd_files = options.pop("csd_files", None)
        normalized_csd_files = (
            parse_csd_file_map(csd_files) if isinstance(next(iter(csd_files or {}), None), str) else csd_files
        )
        return cls(
            psd_files=psd_files or {},
            csd_files=normalized_csd_files,
            detectors=config.detectors,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
            **options,
        )

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
        self._delta_frequency = self.sampling_frequency / self._window_size
        self._frequency_grid = np.fft.rfftfreq(self._window_size, d=1.0 / self.sampling_frequency)
        self._high_frequency_cutoff = (
            self.high_frequency_cutoff if self.high_frequency_cutoff is not None else self.sampling_frequency / 2.0
        )
        self._frequency_mask = (self._frequency_grid >= self.low_frequency_cutoff) & (
            self._frequency_grid <= self._high_frequency_cutoff
        )
        if not np.any(self._frequency_mask):
            raise ValueError("The requested frequency range contains no simulation bins.")

    def _configure_spectral_factors(self) -> None:
        """Construct regularized spectral factors on the FFT grid."""
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        taper = _tukey_window(masked_frequencies.size, alpha=_resolve_taper_alpha(masked_frequencies))
        covariance = build_spectral_covariance_from_files(
            detectors=self.detectors,
            psd_files=self.psd_files,
            csd_files=self._normalized_csd_files,
            frequencies=masked_frequencies,
            taper=taper,
            delta_frequency=self._delta_frequency,
            regularization_epsilon=self.regularization_epsilon,
        )
        self._detector_index = covariance.detector_index
        self._psd = covariance.psd
        self._csd = covariance.csd
        self._spectral_matrices = covariance.matrices
        self._cholesky_factors = covariance.cholesky_factors

        input_spacing = np.inf
        for path in self.psd_files.values():
            psd_frequencies, _ = load_spectral_series(path, kind="PSD")
            input_spacing = min(input_spacing, median_frequency_spacing(psd_frequencies))
        warn_if_underresolved(
            delta_frequency=self._delta_frequency,
            low_frequency_cutoff=self.low_frequency_cutoff,
            reference_spacing=input_spacing,
            logger=logger,
            context="correlated noise simulator",
        )

    def _regularized_cholesky(self, spectral_matrix: np.ndarray) -> np.ndarray:
        """Return a numerically stable Cholesky-like factor."""
        return regularized_cholesky(spectral_matrix, regularization_epsilon=self.regularization_epsilon)

    def _initialize_generator(self, seed: int | None) -> None:
        """Initialize the shared random-number generator."""
        self._rng = np.random.default_rng(seed)

    def _generate_realization_chunk(self) -> dict[str, np.ndarray]:
        """Generate one correlated noise chunk for all detectors."""
        if self._rng is None:
            raise RuntimeError("Random number generator not initialized.")

        return simulate_spectral_covariance_chunk(
            self._rng,
            self._cholesky_factors,
            detectors=self.detectors,
            frequency_grid_size=self._frequency_grid.size,
            frequency_mask=self._frequency_mask,
            delta_frequency=self._delta_frequency,
            window_size=self._window_size,
        )

    def _simulate(self, *, n_samples: int) -> dict[str, np.ndarray]:
        """Generate a prefix-consistent realization from shared correlated chunks."""
        history = self.previous_strain
        if history:
            self._stitcher._validate_chunk_map(history)
            current_raw = {detector: history[detector].copy() for detector in self.detectors}
        else:
            warmup = self._generate_realization_chunk()
            self._stitcher._validate_chunk_map(warmup)
            current_raw = self._generate_realization_chunk()
            self._stitcher._validate_chunk_map(current_raw)

        overlap_size = self._stitcher.overlap_size
        frame_step = self._stitcher.window_size - overlap_size
        emitted_segments = {detector: [] for detector in self.detectors}
        produced_samples = 0

        while produced_samples < n_samples:
            next_raw = self._generate_realization_chunk()
            self._stitcher._validate_chunk_map(next_raw)

            for detector in self.detectors:
                blended_overlap = (
                    (current_raw[detector][-overlap_size:] * self._stitcher._window_out)
                    + (next_raw[detector][:overlap_size] * self._stitcher._window_in)
                ) / self._stitcher._blend_norm
                emitted_segments[detector].append(blended_overlap)
                current_raw[detector] = next_raw[detector].copy()

            produced_samples += frame_step

        realization = {
            detector: np.concatenate(segments)[:n_samples] for detector, segments in emitted_segments.items()
        }
        self.previous_strain.clear()
        self.previous_strain.update({detector: current_raw[detector].copy() for detector in self.detectors})
        return realization

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
            self._window_size, self._overlap_size = resolve_window_sizes(self.window_duration, self.sampling_frequency)
            self._stitcher = OverlapAddStitcher(
                self.detectors,
                window_size=self._window_size,
                overlap_size=self._overlap_size,
            )
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
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")
        return self._simulate(n_samples=n_samples)

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
                "window_duration": self.window_duration,
                "window_size": self._window_size,
                "overlap_size": self._overlap_size,
                "regularization_epsilon": self.regularization_epsilon,
            },
        }
