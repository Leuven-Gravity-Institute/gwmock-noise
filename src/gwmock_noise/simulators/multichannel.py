"""Streaming multichannel noise from a cross-spectral matrix.

The simulator fits a causal minimum-phase matrix spectral factor to a tabulated
pair of power spectra and a cross-spectral density, then drives a bounded-state
streaming filter with white innovations.

The factor is the autoregressive form of
:func:`gwmock_noise.simulators.matrix_factorization.whittle_levinson_factorization`:
the block Levinson-Durbin (Whittle) recursion turns the matrix autocovariance of
the target into coefficient matrices ``A_1 .. A_p`` and an innovation covariance
``V``, and the process

    x_t = -A_1 x_{t-1} - ... - A_p x_{t-p} + V**(1/2) w_t

has the cross-spectral density ``A(f)**-1 V A(f)**-H``, an order-``p``
approximation of the target. The recursion guarantees that every zero of
``det A(z)`` lies strictly inside the unit circle, so the filter is stable and
minimum-phase by construction; the continuation state is the ``p`` samples of
history per channel, independent of the generated span.

The scalar case (one channel) reduces to the same all-pole model as
:class:`~gwmock_noise.simulators.autoregressive.ARNoiseSimulator`. Against
:class:`~gwmock_noise.simulators.correlated_ar.CorrelatedARNoiseSimulator`, which
takes the per-frequency Cholesky factor of the same target and truncates it, the
Whittle factor is causal by construction rather than by truncation, so at equal
order it carries more of the target and its state is the same bounded history.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators._fit import FitError
from gwmock_noise.simulators._spectral import load_spectral_series, median_frequency_spacing
from gwmock_noise.simulators._stitching import warn_if_underresolved
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.colored import _resolve_taper_alpha, _tukey_window
from gwmock_noise.simulators.correlated import parse_csd_file_map
from gwmock_noise.simulators.matrix_factorization import (
    matrix_autocovariance,
    matrix_band_fit_residual,
    whittle_levinson_factorization,
)
from gwmock_noise.utils.log import LOGGER_NAME

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

logger = logging.getLogger(LOGGER_NAME)

#: Default order of the autoregressive factor.
DEFAULT_ORDER = 256
#: Number of samples generated per recursion call before buffering.
DEFAULT_BLOCK_SIZE = 65_536
#: Relative ridge added to the target diagonal before factorizing.
DEFAULT_REGULARIZATION_EPSILON = 1e-8
DETECTOR_PAIR_SIZE = 2
MIN_SPECTRAL_POINTS = 2
#: Number of design-grid samples per autoregressive order.
DESIGN_OVERSAMPLING = 8
RESUME_STATE_FORMAT_VERSION = 1


class MultichannelNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate multichannel noise from a tabulated PSD/CSD matrix."""

    simulator_name = "multichannel"

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_files: dict[str, str | Path] | None = None,
        csd_files: dict[str, str | Path] | dict[tuple[str, str], str | Path] | None = None,
        target_matrices: np.ndarray | None = None,
        target_frequencies: np.ndarray | None = None,
        order: int = DEFAULT_ORDER,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        regularization_epsilon: float = DEFAULT_REGULARIZATION_EPSILON,
    ) -> None:
        """Initialize the simulator and fit the matrix spectral factor once.

        The cross-spectral matrix may be given either as files (``psd_files``
        plus ``csd_files``, absent off-diagonal pairs mean zero coherence) or as
        an in-memory ``target_matrices`` array on ``target_frequencies``, which
        lets an analytic pair bypass the tabulated-curve interpolation.

        Args:
            psd_files: Per-detector PSD table paths, keyed by detector.
            csd_files: Cross-spectral table paths, keyed by detector pair or by
                ``"DET1-DET2"`` strings.
            target_matrices: One-sided cross-spectral matrices of shape
                ``(n_frequencies, n_channels, n_channels)``, mutually exclusive
                with the file inputs.
            target_frequencies: Frequencies of ``target_matrices`` in hertz.
            order: Order ``p`` of the autoregressive factor.
            detectors: Detector names to generate.
            sampling_frequency: Sampling frequency in hertz.
            duration: Nominal generation duration in seconds.
            seed: Seed for the shared innovation generator.
            low_frequency_cutoff: Lower edge of the fit band in hertz.
            high_frequency_cutoff: Upper edge of the fit band in hertz.
            block_size: Number of samples generated per recursion call.
            regularization_epsilon: Relative ridge added to the target diagonal
                where it is not positive definite, as a fraction of the largest
                diagonal entry.
        """
        file_inputs = psd_files is not None
        if file_inputs == (target_matrices is not None):
            raise ValueError("Exactly one of psd_files or target_matrices must be provided.")

        self.psd_files = {detector: Path(path) for detector, path in (psd_files or {}).items()}
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
        self._target_matrices_input = (
            np.asarray(target_matrices, dtype=np.complex128) if target_matrices is not None else None
        )
        self._target_frequency_input = (
            np.asarray(target_frequencies, dtype=float) if target_frequencies is not None else None
        )
        if self._target_matrices_input is not None:
            if self._target_frequency_input is None:
                n_frequencies = self._target_matrices_input.shape[0]
                self._target_frequency_input = np.fft.rfftfreq(2 * (n_frequencies - 1), d=1.0 / self.sampling_frequency)
            if self._target_matrices_input.shape[-1] != len(self.detectors):
                raise ValueError("target_matrices channel count must match detectors.")
            if self._target_frequency_input.shape[0] != self._target_matrices_input.shape[0]:
                raise ValueError("target_frequencies and target_matrices must agree in length.")

        self._rng: np.random.Generator | None = None
        self._state = np.zeros((self.order, len(self.detectors)), dtype=float)
        self._pending = np.zeros((0, len(self.detectors)), dtype=float)
        self._ar_coefficients = np.zeros((self.order, len(self.detectors), len(self.detectors)), dtype=float)
        self._innovation_factor = np.zeros((len(self.detectors), len(self.detectors)), dtype=float)
        self._innovation_covariance = np.zeros((len(self.detectors), len(self.detectors)), dtype=float)
        self._fit_time_seconds = 0.0
        self._design_size = 0
        self._high_frequency_cutoff = 0.0
        self._target_matrices = np.zeros((0, 0, 0), dtype=np.complex128)
        self._model_matrices = np.zeros((0, 0, 0), dtype=np.complex128)
        self._fit_frequencies = np.zeros(0, dtype=float)
        self._fit_residual: dict[str, object] = {
            "band_count": 0,
            "worst_relative_error": 0.0,
            "median_relative_error": 0.0,
            "bands": [],
        }
        self._diagnostics: dict[str, float] = {}

        self._fit_model()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> MultichannelNoiseSimulator:
        """Construct a multichannel simulator from one component definition."""
        options = dict(component.options)
        psd_files = options.pop("psd_files", None)
        csd_files = options.pop("csd_files", None)
        return cls(
            psd_files=psd_files,
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
        if self.order < 1:
            raise ValueError("order must be greater than zero.")
        if self.block_size < 1:
            raise ValueError("block_size must be greater than zero.")
        if self.regularization_epsilon < 0:
            raise ValueError("regularization_epsilon must be non-negative.")
        if self.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be non-negative.")

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
        if self.psd_files and set(self.psd_files) != detector_set:
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

    def _input_grid_size(self) -> int:
        """Return a uniform FFT size covering every spectral input."""
        max_points = 0
        for psd_path in self.psd_files.values():
            frequencies, _ = load_spectral_series(psd_path, kind="PSD")
            max_points = max(max_points, frequencies.size)
        for csd_path in self._normalized_csd_files.values():
            frequencies, _ = load_spectral_series(csd_path, kind="CSD", complex_values=True)
            max_points = max(max_points, frequencies.size)
        if self._target_matrices_input is not None:
            max_points = max(max_points, self._target_frequency_input.size)
        if max_points < MIN_SPECTRAL_POINTS:
            raise ValueError("Spectral inputs must contain at least two frequency samples.")
        return max(2 * (max_points - 1), DESIGN_OVERSAMPLING * self.order)

    def _interpolate_real(self, values: np.ndarray, frequencies: np.ndarray, masked: np.ndarray) -> np.ndarray:
        """Interpolate a real spectral series onto the masked grid."""
        return np.interp(masked, frequencies, values, left=0.0, right=0.0)

    def _interpolate_complex(self, values: np.ndarray, frequencies: np.ndarray, masked: np.ndarray) -> np.ndarray:
        """Interpolate a complex spectral series onto the masked grid."""
        real = np.interp(masked, frequencies, values.real, left=0.0, right=0.0)
        imag = np.interp(masked, frequencies, values.imag, left=0.0, right=0.0)
        return real + 1j * imag

    def _build_target_matrices(self, frequency_grid: np.ndarray, frequency_mask: np.ndarray) -> np.ndarray:
        """Assemble the target cross-spectral matrix on the design grid."""
        n_channels = len(self.detectors)
        target = np.zeros((frequency_grid.size, n_channels, n_channels), dtype=np.complex128)
        masked = frequency_grid[frequency_mask]
        taper = _tukey_window(masked.size, alpha=_resolve_taper_alpha(masked))
        detector_index = {detector: index for index, detector in enumerate(self.detectors)}

        if self._target_matrices_input is not None:
            for index in range(n_channels):
                for other in range(n_channels):
                    interpolated = self._interpolate_complex(
                        self._target_matrices_input[:, index, other], self._target_frequency_input, masked
                    )
                    target[frequency_mask, index, other] = interpolated * taper
        else:
            for detector, psd_path in self.psd_files.items():
                frequencies, values = load_spectral_series(psd_path, kind="PSD")
                index = detector_index[detector]
                target[frequency_mask, index, index] = self._interpolate_real(values, frequencies, masked) * taper
            for pair, csd_path in self._normalized_csd_files.items():
                frequencies, values = load_spectral_series(csd_path, kind="CSD", complex_values=True)
                index_a = detector_index[pair[0]]
                index_b = detector_index[pair[1]]
                interpolated = self._interpolate_complex(values, frequencies, masked) * taper
                target[frequency_mask, index_a, index_b] = interpolated
                target[frequency_mask, index_b, index_a] = np.conj(interpolated)

        target = 0.5 * (target + target.conj().swapaxes(-1, -2)) * (self.sampling_frequency / 2.0)
        for index in (0, target.shape[0] - 1):
            target[index] = 0.5 * (target[index] + target[index].conj().T)
        input_spacing = self._reference_frequency_spacing()
        warn_if_underresolved(
            delta_frequency=self.sampling_frequency / self._design_size,
            low_frequency_cutoff=self.low_frequency_cutoff,
            reference_spacing=input_spacing,
            logger=logger,
            context="multichannel noise simulator",
        )
        return target

    def _reference_frequency_spacing(self) -> float:
        """Return the finest tabulated spacing among the spectral inputs."""
        spacing = np.inf
        for psd_path in self.psd_files.values():
            frequencies, _ = load_spectral_series(psd_path, kind="PSD")
            spacing = min(spacing, median_frequency_spacing(frequencies))
        for csd_path in self._normalized_csd_files.values():
            frequencies, _ = load_spectral_series(csd_path, kind="CSD", complex_values=True)
            spacing = min(spacing, median_frequency_spacing(frequencies))
        if self._target_matrices_input is not None:
            spacing = min(spacing, median_frequency_spacing(self._target_frequency_input))
        return spacing

    def _regularize_target(self, target: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Add a relative ridge where the target is not positive definite.

        Returns:
            The regularised target, its smallest eigenvalue before the ridge, and
            the absolute ridge added.
        """
        target = np.asarray(target, dtype=np.complex128)
        hermitian = 0.5 * (target + target.conj().swapaxes(-1, -2))
        eigenvalues = np.linalg.eigvalsh(hermitian)
        min_eigenvalue = float(np.min(eigenvalues))
        scale = max(
            float(np.max(np.real(np.diagonal(hermitian, axis1=-2, axis2=-1)), initial=0.0)),
            np.finfo(float).tiny,
        )
        threshold = self.regularization_epsilon * scale
        if min_eigenvalue >= threshold:
            return hermitian, min_eigenvalue, 0.0
        if self.regularization_epsilon == 0.0:
            raise FitError(
                "The target cross-spectral matrix is not positive definite and no ridge is allowed; "
                "the fit is not usable."
            )
        ridge = threshold - min_eigenvalue
        identity = np.eye(target.shape[-1], dtype=np.complex128)
        return hermitian + ridge * identity, min_eigenvalue, ridge

    def _fit_model(self) -> None:
        """Fit the Whittle autoregressive factor from the target matrix."""
        fit_start = time.perf_counter()
        self._design_size = self._input_grid_size()
        if self._design_size % 2 != 0:
            self._design_size += 1
        nyquist = self.sampling_frequency / 2.0
        self._high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist

        frequency_grid = np.fft.rfftfreq(self._design_size, d=1.0 / self.sampling_frequency)
        frequency_mask = (frequency_grid >= self.low_frequency_cutoff) & (frequency_grid <= self._high_frequency_cutoff)
        if not np.any(frequency_mask):
            raise ValueError("The requested frequency range contains no simulation bins.")

        target = self._build_target_matrices(frequency_grid, frequency_mask)
        regularized, min_eigenvalue, ridge = self._regularize_target(target)
        covariance = matrix_autocovariance(regularized, self._design_size)[: self.order + 1]

        factor = whittle_levinson_factorization(covariance, self.order)
        self._target_matrices = target
        self._ar_coefficients = factor.ar_coefficients
        self._innovation_covariance = factor.innovation_covariance
        self._innovation_factor = factor.innovation_factor
        self._fit_frequencies = frequency_grid
        self._model_matrices = self._evaluate_model_matrices(frequency_grid)
        self._fit_residual = matrix_band_fit_residual(
            frequency_grid[frequency_mask],
            target[frequency_mask],
            self._model_matrices[frequency_mask],
            low_frequency=self.low_frequency_cutoff,
            high_frequency=self._high_frequency_cutoff,
        )
        self._diagnostics = {
            "min_eigenvalue": min_eigenvalue,
            "regularization_ridge": ridge,
            "max_reflection_singular_value": factor.max_reflection_singular_value,
            "min_prediction_error_eigenvalue": factor.min_prediction_error_eigenvalue,
            "innovation_condition_number": factor.innovation_condition_number,
        }
        self._fit_time_seconds = time.perf_counter() - fit_start
        self._state = np.zeros((self.order, len(self.detectors)), dtype=float)
        self._pending = np.zeros((0, len(self.detectors)), dtype=float)

    def _evaluate_model_matrices(self, frequency_grid: np.ndarray) -> np.ndarray:
        """Return the fitted cross-spectral matrices on the design grid."""
        n_frequencies = frequency_grid.size
        n_channels = len(self.detectors)
        angular = 2.0 * np.pi * frequency_grid / self.sampling_frequency
        response = np.zeros((n_frequencies, n_channels, n_channels), dtype=np.complex128)
        for order_index in range(self.order):
            phase = np.exp(-1j * angular * (order_index + 1))
            response += phase[:, None, None] * self._ar_coefficients[order_index]
        response += np.eye(n_channels)
        response_inverse = np.linalg.inv(response)
        return response_inverse @ self._innovation_covariance @ response_inverse.conj().swapaxes(-1, -2)

    def _initialize_generator(self, seed: int | None) -> None:
        """Initialize the shared innovation generator."""
        self._rng = np.random.default_rng(seed)

    def _generate_block(self, n_samples: int) -> np.ndarray:
        """Generate one contiguous multichannel block from the recursion."""
        if self._rng is None:
            raise RuntimeError("Random number generator not initialized.")
        innovations = self._rng.standard_normal((n_samples, len(self.detectors)))
        coefficients = self._ar_coefficients
        state = self._state
        factor = self._innovation_factor
        block = np.empty((n_samples, len(self.detectors)), dtype=float)

        for sample in range(n_samples):
            predicted = -np.einsum("kij,kj->i", coefficients, state, optimize=True)
            current = predicted + factor @ innovations[sample]
            block[sample] = current
            if self.order > 1:
                state[1:] = state[:-1]
            if self.order > 0:
                state[0] = current

        self._state = state
        return block

    def reset(self) -> None:
        """Clear the recursion history, pending output and generator."""
        self._rng = None
        self._state = np.zeros((self.order, len(self.detectors)), dtype=float)
        self._pending = np.zeros((0, len(self.detectors)), dtype=float)

    def _pull(self, n_samples: int) -> np.ndarray:
        """Return the next ``n_samples`` of the multichannel process."""
        segments: list[np.ndarray] = []
        remaining = n_samples
        while remaining > 0:
            if self._pending.shape[0]:
                take = min(remaining, self._pending.shape[0])
                segments.append(self._pending[:take])
                self._pending = self._pending[take:]
                remaining -= take
                continue
            self._pending = self._generate_block(self.block_size)
        return np.concatenate(segments, axis=0)

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector multichannel noise with continuity across calls."""
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

        realization = self._pull(n_samples)
        detector_index = {detector: index for index, detector in enumerate(self.detectors)}
        return {detector: realization[:, detector_index[detector]].copy() for detector in self.detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield multichannel chunks lazily while preserving recursion history."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    def continuation_state(self) -> dict[str, Any]:
        """Return the bounded state needed to keep a running stream going.

        The state is the recursion history plus the pending output; its size is
        proportional to the order and independent of the generated span.
        """
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "recursion_state": self._state.copy(),
            "pending_samples": self._pending.copy(),
        }

    def resume_metadata(self) -> dict[str, Any]:
        """Return the metadata a stopped stream must persist to resume."""
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "order": self.order,
            "detectors": list(self.detectors),
            "sampling_frequency": self.sampling_frequency,
            "seed": self.seed,
            "rng_state": self._rng.bit_generator.state if self._rng is not None else None,
        }

    def export_state(self) -> dict[str, Any]:
        """Return a picklable snapshot sufficient to resume the stream."""
        return {"continuation": self.continuation_state(), "resume": self.resume_metadata()}

    def import_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot produced by :meth:`export_state`.

        Args:
            state: A snapshot from another simulator with the same configuration.

        Raises:
            ValueError: If the snapshot does not match this simulator's settings.
        """
        continuation = state["continuation"]
        resume = state["resume"]
        if resume["order"] != self.order:
            raise ValueError("resume metadata order does not match this simulator.")
        if resume["detectors"] != self.detectors:
            raise ValueError("resume metadata detectors do not match this simulator.")
        if resume["sampling_frequency"] != self.sampling_frequency:
            raise ValueError("resume metadata sampling frequency does not match this simulator.")

        self.seed = resume["seed"]
        self._initialize_generator(self.seed)
        if resume["rng_state"] is not None:
            self._rng.bit_generator.state = resume["rng_state"]
        self._state = np.asarray(continuation["recursion_state"], dtype=float).copy()
        self._pending = np.asarray(continuation["pending_samples"], dtype=float).copy()

    @property
    def ar_coefficients(self) -> np.ndarray:
        """Return a copy of the fitted coefficient matrices ``A_1 .. A_p``."""
        return self._ar_coefficients.copy()

    @property
    def innovation_covariance(self) -> np.ndarray:
        """Return a copy of the fitted innovation covariance."""
        return self._innovation_covariance.copy()

    @property
    def design_frequencies(self) -> np.ndarray:
        """Return a copy of the design-grid frequencies in hertz."""
        return self._fit_frequencies.copy()

    @property
    def target_spectral_matrices(self) -> np.ndarray:
        """Return a copy of the target cross-spectral matrices on the design grid.

        The matrices are in the covariance scale the recursion uses
        (``one_sided_psd * sampling_frequency / 2``) and carry the band mask and
        edge taper the fit was built from.
        """
        return self._target_matrices.copy()

    @property
    def model_spectral_matrices(self) -> np.ndarray:
        """Return a copy of the fitted cross-spectral matrices on the design grid."""
        return self._model_matrices.copy()

    @property
    def target_spectral_matrix_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the band-masked target cross-spectral matrices on the design grid."""
        mask = self._frequency_mask()
        return self._fit_frequencies[mask], self._target_matrices[mask]

    @property
    def model_spectral_matrix_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the band-masked fitted cross-spectral matrices on the design grid."""
        mask = self._frequency_mask()
        return self._fit_frequencies[mask], self._model_matrices[mask]

    def _frequency_mask(self) -> np.ndarray:
        """Return the band mask on the current design grid."""
        return (self._fit_frequencies >= self.low_frequency_cutoff) & (
            self._fit_frequencies <= self._high_frequency_cutoff
        )

    @property
    def state_nbytes(self) -> int:
        """Serialized size of the continuation state in bytes."""
        return len(pickle.dumps(self.continuation_state()))

    @property
    def resume_metadata_nbytes(self) -> int:
        """Serialized size of the resume metadata in bytes."""
        return len(pickle.dumps(self.resume_metadata()))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the fitted matrix spectral factor."""
        return {
            "implementation": "multichannel",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "multichannel_noise": {
                "psd_files": {detector: str(path) for detector, path in self.psd_files.items()},
                "csd_files": {
                    f"{detector_a}-{detector_b}": str(path)
                    for (detector_a, detector_b), path in self._normalized_csd_files.items()
                },
                "fit_method": "whittle-levinson-block-toeplitz",
                "order": self.order,
                "state_size": self.order * len(self.detectors),
                "state_bytes": self.state_nbytes,
                "resume_metadata_bytes": self.resume_metadata_nbytes,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "block_size": self.block_size,
                "design_size": self._design_size,
                "fit_time_seconds": self._fit_time_seconds,
                "regularization_epsilon": self.regularization_epsilon,
                "innovation_covariance": self._innovation_covariance.tolist(),
                "conditioning": {
                    "min_eigenvalue": self._diagnostics.get("min_eigenvalue", 0.0),
                    "regularization_ridge": self._diagnostics.get("regularization_ridge", 0.0),
                    "max_reflection_singular_value": self._diagnostics.get("max_reflection_singular_value", 0.0),
                    "min_prediction_error_eigenvalue": self._diagnostics.get("min_prediction_error_eigenvalue", 0.0),
                    "innovation_condition_number": self._diagnostics.get("innovation_condition_number", 0.0),
                },
                "fit_residual": self._fit_residual,
            },
        }


__all__ = ["FitError", "MultichannelNoiseSimulator"]
