"""Schumann-resonance correlated magnetic noise simulator."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.special import eval_legendre

from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators._stitching import OVERLAP_SIZE, WINDOW_SIZE, OverlapAddStitcher
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.colored import _tukey_window

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

EARTH_RADIUS_METERS = 6_371_000.0
SPEED_OF_LIGHT_METERS_PER_SECOND = 299_792_458.0


@dataclass(frozen=True, slots=True)
class SchumannParams:
    """Physical parameters for the isotropic Schumann-resonance model."""

    mode_frequencies_hz: tuple[float, ...] = (7.83, 14.0, 20.0, 26.0, 32.5)
    quality_factors: tuple[float, ...] = (4.0, 5.0, 6.0, 6.0, 7.0)
    amplitudes: tuple[float, ...] = (1.0, 0.6, 0.45, 0.35, 0.25)
    earth_radius_meters: float = EARTH_RADIUS_METERS
    light_speed_meters_per_second: float = SPEED_OF_LIGHT_METERS_PER_SECOND
    regularization_epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        """Validate the resonance parameter vectors."""
        n_modes = len(self.mode_frequencies_hz)
        if n_modes == 0:
            raise ValueError("SchumannParams must define at least one resonance mode.")
        if len(self.quality_factors) != n_modes or len(self.amplitudes) != n_modes:
            raise ValueError("Schumann parameter arrays must have the same length.")
        if any(frequency <= 0.0 for frequency in self.mode_frequencies_hz):
            raise ValueError("Schumann resonance frequencies must be positive.")
        if any(quality <= 0.0 for quality in self.quality_factors):
            raise ValueError("Schumann quality factors must be positive.")
        if any(amplitude < 0.0 for amplitude in self.amplitudes):
            raise ValueError("Schumann amplitudes must be non-negative.")
        if self.earth_radius_meters <= 0.0:
            raise ValueError("earth_radius_meters must be positive.")
        if self.light_speed_meters_per_second <= 0.0:
            raise ValueError("light_speed_meters_per_second must be positive.")
        if self.regularization_epsilon <= 0.0:
            raise ValueError("regularization_epsilon must be greater than zero.")


class SchumannNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate correlated strain noise from an isotropic Schumann-resonance model."""

    simulator_name = "schumann"

    def __init__(  # noqa: PLR0913
        self,
        *,
        positions: dict[str, tuple[float, float]],
        coupling_files: dict[str, str | Path],
        detectors: list[str] | None = None,
        schumann_params: SchumannParams | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
    ) -> None:
        """Initialize the Schumann simulator."""
        self.positions = {detector: (float(lat), float(lon)) for detector, (lat, lon) in positions.items()}
        self.coupling_files = {detector: Path(path) for detector, path in coupling_files.items()}
        self.detectors = list(detectors) if detectors is not None else list(self.positions)
        self.schumann_params = schumann_params or SchumannParams()
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff

        self._stitcher = OverlapAddStitcher(self.detectors)
        self._rng: np.random.Generator | None = None
        self._high_frequency_cutoff = 0.0
        self._coupling = {}
        self._schumann_psd = np.zeros(0, dtype=float)

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._validate_detector_inputs()
        self._configure_frequency_grid()
        self._configure_spectral_factors()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> SchumannNoiseSimulator:
        """Construct a Schumann-noise simulator from one component definition."""
        options = dict(component.options)
        positions = options.pop("positions", None)
        coupling_files = options.pop("coupling_files", None)
        if positions is None or coupling_files is None:
            raise ValueError("Schumann simulator requires 'positions' and 'coupling_files' in the component options.")
        schumann_params = options.pop("schumann_params", None)
        if isinstance(schumann_params, dict):
            schumann_params = SchumannParams(**schumann_params)
        return cls(
            positions=positions,
            coupling_files=coupling_files,
            detectors=config.detectors,
            schumann_params=schumann_params,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
            **options,
        )

    @property
    def previous_strain(self) -> dict[str, np.ndarray]:
        """Expose continuity buffers for protocol-compatible state inspection."""
        return self._stitcher.previous_strain

    def _validate_detector_inputs(self) -> None:
        """Validate detector positions and coupling coverage."""
        detector_set = set(self.detectors)
        if set(self.positions) != detector_set:
            raise ValueError("positions keys must exactly match detectors.")
        if set(self.coupling_files) != detector_set:
            raise ValueError("coupling_files keys must exactly match detectors.")

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
            raise ValueError("detectors must not contain duplicate names.")
        if self.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be non-negative.")

        nyquist = sampling_frequency / 2.0
        high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist
        if high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")
        if high_frequency_cutoff > nyquist:
            raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

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

    def _load_coupling_on_grid(self, coupling_file: Path) -> np.ndarray:
        """Load one detector coupling function onto the simulator grid."""
        coupling = np.zeros_like(self._frequency_grid, dtype=float)
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        frequencies, values = load_spectral_series(coupling_file, kind="magnetic coupling")
        interpolated = np.interp(masked_frequencies, frequencies, values, left=0.0, right=0.0)
        coupling[self._frequency_mask] = np.clip(interpolated, a_min=0.0, a_max=None)
        return coupling

    def spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Return the isotropic Schumann magnetic PSD on a frequency grid."""
        spectrum = np.zeros_like(frequencies, dtype=float)
        for center, quality, amplitude in zip(
            self.schumann_params.mode_frequencies_hz,
            self.schumann_params.quality_factors,
            self.schumann_params.amplitudes,
            strict=True,
        ):
            half_width = center / (2.0 * quality)
            spectrum += amplitude / (np.square(frequencies - center) + (half_width**2))
        return spectrum

    def _angular_separation(self, detector_a: str, detector_b: str) -> float:
        """Return the angular separation between two detectors in radians."""
        latitude_a, longitude_a = self.positions[detector_a]
        latitude_b, longitude_b = self.positions[detector_b]
        lat_a = np.deg2rad(latitude_a)
        lon_a = np.deg2rad(longitude_a)
        lat_b = np.deg2rad(latitude_b)
        lon_b = np.deg2rad(longitude_b)
        cos_theta = np.sin(lat_a) * np.sin(lat_b) + np.cos(lat_a) * np.cos(lat_b) * np.cos(lon_a - lon_b)
        return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def theoretical_coherence(self, frequency: float, detector_a: str, detector_b: str) -> float:
        """Return the isotropic Schumann coherence approximation."""
        if detector_a == detector_b:
            return 1.0
        theta = self._angular_separation(detector_a, detector_b)
        mode_order = round(
            (2.0 * np.pi * frequency * self.schumann_params.earth_radius_meters)
            / self.schumann_params.light_speed_meters_per_second
        )
        return float(eval_legendre(mode_order, np.cos(theta)))

    def _configure_spectral_factors(self) -> None:
        """Construct regularized strain spectral factors on the FFT grid."""
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        taper = _tukey_window(masked_frequencies.size)

        self._detector_index = {detector: index for index, detector in enumerate(self.detectors)}
        self._coupling = {
            detector: self._load_coupling_on_grid(self.coupling_files[detector]) for detector in self.detectors
        }
        self._schumann_psd = self.spectrum(masked_frequencies) * taper

        n_detectors = len(self.detectors)
        n_frequencies = masked_frequencies.size
        self._cholesky_factors = np.zeros((n_frequencies, n_detectors, n_detectors), dtype=np.complex128)

        for frequency_index, frequency in enumerate(masked_frequencies):
            spectral_matrix = np.zeros((n_detectors, n_detectors), dtype=np.complex128)
            for detector in self.detectors:
                detector_index = self._detector_index[detector]
                coupling = self._coupling[detector][self._frequency_mask][frequency_index]
                spectral_matrix[detector_index, detector_index] = self._schumann_psd[frequency_index] * (coupling**2)

            for detector_a, detector_b in combinations(self.detectors, 2):
                index_a = self._detector_index[detector_a]
                index_b = self._detector_index[detector_b]
                coupling_a = self._coupling[detector_a][self._frequency_mask][frequency_index]
                coupling_b = self._coupling[detector_b][self._frequency_mask][frequency_index]
                coherence = self.theoretical_coherence(frequency, detector_a, detector_b)
                cross_spectrum = self._schumann_psd[frequency_index] * coupling_a * coupling_b * coherence
                spectral_matrix[index_a, index_b] = cross_spectrum
                spectral_matrix[index_b, index_a] = np.conj(cross_spectrum)

            self._cholesky_factors[frequency_index] = self._regularized_cholesky(
                spectral_matrix * (0.5 / self._delta_frequency)
            )

    def _regularized_cholesky(self, spectral_matrix: np.ndarray) -> np.ndarray:
        """Return a numerically stable Cholesky-like factor."""
        hermitian_matrix = 0.5 * (spectral_matrix + spectral_matrix.conj().T)
        diagonal_scale = max(float(np.max(np.real(np.diag(hermitian_matrix)))), 1.0)
        epsilon = self.schumann_params.regularization_epsilon * diagonal_scale
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
        """Generate one correlated Schumann-noise chunk for all detectors."""
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
        """Generate correlated Schumann strain noise."""
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
            self._validate_detector_inputs()
            self._stitcher.configure_detectors(self.detectors)
            self.reset()
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
        """Yield Schumann-noise chunks lazily while preserving simulator state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
        return {
            "implementation": "schumann",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "schumann_noise": {
                "positions": {detector: list(position) for detector, position in self.positions.items()},
                "coupling_files": {detector: str(path) for detector, path in self.coupling_files.items()},
                "mode_frequencies_hz": list(self.schumann_params.mode_frequencies_hz),
                "quality_factors": list(self.schumann_params.quality_factors),
                "amplitudes": list(self.schumann_params.amplitudes),
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "window_size": WINDOW_SIZE,
                "overlap_size": OVERLAP_SIZE,
            },
        }
