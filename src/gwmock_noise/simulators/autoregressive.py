"""Autoregressive noise simulator with persistent detector state.

The all-pole model is fitted to the autocovariance implied by a tabulated PSD
with the Levinson-Durbin recursion, so the filter is stable by construction:
every reflection coefficient lies below one and every pole inside the unit
circle. The fit records its conditioning diagnostics and the relative PSD
residual per band, and a fit that cannot satisfy the pre-registered limits
raises instead of degrading.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.signal import lfilter

from gwmock_noise.simulators._fit import (
    DEFAULT_FIT_BANDS,
    FitError,
    band_fit_residual,
    levinson_durbin,
    toeplitz_condition_number,
)
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

DEFAULT_AR_ORDER = 256
DEFAULT_BLOCK_SIZE = 65_536
DEFAULT_REGULARIZATION = 1e-10
RESUME_STATE_FORMAT_VERSION = 1


def _stable_detector_hash(detector: str) -> int:
    """Return a stable integer hash for a detector label."""
    digest = hashlib.blake2b(detector.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


class ARNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate stateful detector noise from an AR model fit to a target PSD."""

    simulator_name = "ar"

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_file: str | Path,
        order: int = DEFAULT_AR_ORDER,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        regularization: float = DEFAULT_REGULARIZATION,
    ) -> None:
        """Initialize the simulator and fit the AR model once.

        Args:
            psd_file: Path or bundled preset name of a two-column PSD table.
            order: Autoregressive order ``p``.
            detectors: Detector names to generate.
            sampling_frequency: Sampling frequency in hertz.
            duration: Nominal generation duration in seconds.
            seed: Base seed for the per-detector generators.
            low_frequency_cutoff: Lower edge of the target band in hertz.
            high_frequency_cutoff: Upper edge of the target band in hertz.
            block_size: Number of samples generated per recursion call.
            regularization: Relative ridge added to the zero-lag autocovariance
                before the recursion; ``0`` disables it.
        """
        self.psd_file = Path(psd_file)
        self.order = order
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.block_size = block_size
        self.regularization = regularization

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)

        self._rngs: dict[str, np.random.Generator] = {}
        self._state: dict[str, np.ndarray] = {}
        self._ar_coefficients = np.zeros(self.order, dtype=float)
        self._denominator = np.ones(1, dtype=float)
        self._innovation_variance = 0.0
        self._fit_time_seconds = 0.0
        self._fit_grid_size = 0
        self._high_frequency_cutoff = 0.0
        self._reflection_coefficients = np.zeros(self.order, dtype=float)
        self._prediction_errors = np.zeros(self.order + 1, dtype=float)
        self._condition_number = 1.0
        self._target_frequencies = np.zeros(0, dtype=float)
        self._target_psd = np.zeros(0, dtype=float)
        self._model_psd = np.zeros(0, dtype=float)
        self._band_mask = np.zeros(0, dtype=bool)
        self._fit_residual: dict[str, object] = {
            "band_count": 0,
            "worst_relative_error": 0.0,
            "median_relative_error": 0.0,
            "bands": [],
        }

        self._fit_model()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> ARNoiseSimulator:
        """Construct an AR-noise simulator from one component definition."""
        options = dict(component.options)
        psd_file = options.pop("psd_file", None)
        if psd_file is None:
            raise ValueError("AR simulator requires 'psd_file' in the component options.")
        return cls(
            psd_file=psd_file,
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
            raise ValueError("detectors must not contain duplicate names.")
        if self.order < 1:
            raise ValueError("order must be greater than zero.")
        if self.block_size < 1:
            raise ValueError("block_size must be greater than zero.")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative.")
        if self.low_frequency_cutoff < 0:
            raise ValueError("low_frequency_cutoff must be non-negative.")

        nyquist = sampling_frequency / 2.0
        high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist
        if high_frequency_cutoff <= self.low_frequency_cutoff:
            raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")
        if high_frequency_cutoff > nyquist:
            raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

    def _fit_model(self) -> None:
        """Fit AR coefficients from the configured one-sided PSD."""
        fit_start = time.perf_counter()

        nyquist = self.sampling_frequency / 2.0
        self._high_frequency_cutoff = self.high_frequency_cutoff if self.high_frequency_cutoff is not None else nyquist
        psd_frequencies, psd_values = load_spectral_series(self.psd_file, kind="PSD")
        self._fit_grid_size = max(2 * (psd_frequencies.size - 1), 8 * self.order)

        frequency_grid = np.fft.rfftfreq(self._fit_grid_size, d=1.0 / self.sampling_frequency)
        frequency_mask = (frequency_grid >= self.low_frequency_cutoff) & (frequency_grid <= self._high_frequency_cutoff)
        if not np.any(frequency_mask):
            raise ValueError("The requested frequency range contains no simulation bins.")

        target_psd = np.zeros_like(frequency_grid, dtype=float)
        target_psd[frequency_mask] = np.clip(
            np.interp(frequency_grid[frequency_mask], psd_frequencies, psd_values, left=0.0, right=0.0),
            a_min=0.0,
            a_max=None,
        )

        autocovariance = np.fft.irfft(target_psd * self.sampling_frequency / 2.0, n=self._fit_grid_size)
        autocovariance = np.asarray(autocovariance[: self.order + 1], dtype=float)
        if autocovariance[0] <= 0.0:
            raise ValueError("Target PSD integrates to zero variance in the requested band.")
        if self.regularization > 0.0:
            autocovariance[0] *= 1.0 + self.regularization

        fit = levinson_durbin(autocovariance, self.order)
        self._ar_coefficients = np.asarray(fit.coefficients, dtype=float)
        self._innovation_variance = float(fit.prediction_error)
        self._reflection_coefficients = np.asarray(fit.reflection_coefficients, dtype=float)
        self._prediction_errors = np.asarray(fit.prediction_errors, dtype=float)
        self._condition_number = toeplitz_condition_number(autocovariance[:-1], self.order)
        self._denominator = np.concatenate(([1.0], self._ar_coefficients))

        self._target_frequencies = frequency_grid
        self._target_psd = target_psd
        self._band_mask = frequency_mask
        self._model_psd = self._evaluate_model_psd()
        self._fit_residual = band_fit_residual(
            frequency_grid[frequency_mask],
            target_psd[frequency_mask],
            self._model_psd[frequency_mask],
            low_frequency=self.low_frequency_cutoff,
            high_frequency=self._high_frequency_cutoff,
            n_bands=DEFAULT_FIT_BANDS,
        )

        self._fit_time_seconds = time.perf_counter() - fit_start
        self._state = {detector: np.zeros(self.order, dtype=float) for detector in self.detectors}

    def _evaluate_model_psd(self) -> np.ndarray:
        """Return the fitted AR spectrum on the design grid.

        The rational spectrum ``2 E_p |1 / A|^2 / f_s`` is evaluated from the
        impulse response rather than by summing the polynomial on the unit
        circle: for a high-order fit with poles close to the circle the direct
        sum loses precision to cancellation, while the impulse response is
        stable.
        """
        impulse = np.zeros(self._fit_grid_size, dtype=float)
        impulse[0] = 1.0
        response = np.fft.rfft(lfilter([1.0], self._denominator, impulse), n=self._fit_grid_size)
        return (2.0 * self._innovation_variance / self.sampling_frequency) * np.abs(response) ** 2

    def _initialize_generators(self, seed: int | None) -> None:
        """Initialize one RNG per detector."""
        if seed is None:
            seed_sequence = np.random.SeedSequence()
            child_sequences = seed_sequence.spawn(len(self.detectors))
        else:
            child_sequences = [
                np.random.SeedSequence([seed, _stable_detector_hash(detector)]) for detector in self.detectors
            ]

        self._rngs = {
            detector: np.random.default_rng(child_sequence)
            for detector, child_sequence in zip(self.detectors, child_sequences, strict=True)
        }

    def _generate_block(self, detector: str, n_samples: int) -> np.ndarray:
        """Generate one contiguous AR block for a single detector."""
        innovations = self._rngs[detector].standard_normal(n_samples)
        strain, filter_state = lfilter(
            [np.sqrt(self._innovation_variance)],
            self._denominator,
            innovations,
            zi=self._state[detector],
        )
        self._state[detector] = filter_state
        return strain

    def reset(self) -> None:
        """Clear detector state and RNGs."""
        self._rngs = {}
        self._state = {detector: np.zeros(self.order, dtype=float) for detector in self.detectors}

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector AR noise with continuity across calls."""
        runtime_detectors = list(detectors)
        self._validate_runtime(
            duration=duration,
            sampling_frequency=sampling_frequency,
            detectors=runtime_detectors,
        )

        reconfigure_fit = sampling_frequency != self.sampling_frequency
        reconfigure_state = reconfigure_fit or runtime_detectors != self.detectors

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors

        if reconfigure_fit:
            self._fit_model()
            self.reset()
        elif reconfigure_state:
            self.reset()

        if seed is not None:
            self.seed = seed
            self.reset()

        if not self._rngs:
            self._initialize_generators(self.seed)

        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        realizations: dict[str, np.ndarray] = {}
        for detector in self.detectors:
            if n_samples <= self.block_size:
                realizations[detector] = self._generate_block(detector, n_samples)
                continue

            blocks = []
            remaining = n_samples
            while remaining > 0:
                chunk_size = min(self.block_size, remaining)
                blocks.append(self._generate_block(detector, chunk_size))
                remaining -= chunk_size
            realizations[detector] = np.concatenate(blocks)

        return realizations

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield AR-noise chunks lazily while preserving recursion state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    def continuation_state(self) -> dict[str, Any]:
        """Return the bounded state needed to keep a running stream going.

        The state is the recursion memory of every detector; its size is
        proportional to the order and independent of the generated span.
        """
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "filter_state": {detector: state.copy() for detector, state in self._state.items()},
        }

    def resume_metadata(self) -> dict[str, Any]:
        """Return the metadata a stopped stream must persist to resume."""
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "order": self.order,
            "detectors": list(self.detectors),
            "sampling_frequency": self.sampling_frequency,
            "seed": self.seed,
            "rng_state": {detector: rng.bit_generator.state for detector, rng in self._rngs.items()},
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
        self._initialize_generators(self.seed)
        for detector, rng_state in resume["rng_state"].items():
            self._rngs[detector].bit_generator.state = rng_state
        for detector in self.detectors:
            self._state[detector] = np.asarray(continuation["filter_state"][detector], dtype=float).copy()

    @property
    def autoregressive_coefficients(self) -> np.ndarray:
        """Return a copy of the fitted ``a_1 .. a_p`` coefficients."""
        return self._ar_coefficients.copy()

    @property
    def target_psd_curve(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the band-masked target PSD on the design grid."""
        return self._target_frequencies[self._band_mask], self._target_psd[self._band_mask]

    @property
    def model_psd_curve(self) -> np.ndarray:
        """Return the fitted model PSD on the band-masked design grid."""
        return self._model_psd[self._band_mask]

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
        """Return metadata describing the fitted AR model."""
        return {
            "implementation": "autoregressive",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "autoregressive_noise": {
                "psd_file": str(self.psd_file),
                "fit_method": "levinson-durbin",
                "order": self.order,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "block_size": self.block_size,
                "fit_grid_size": self._fit_grid_size,
                "fit_time_seconds": self._fit_time_seconds,
                "regularization": self.regularization,
                "innovation_variance": self._innovation_variance,
                "state_size": self.order,
                "state_bytes": self.state_nbytes,
                "resume_metadata_bytes": self.resume_metadata_nbytes,
                "conditioning": {
                    "max_reflection_coefficient": float(np.max(np.abs(self._reflection_coefficients), initial=0.0)),
                    "max_reflection_limit": 1.0,
                    "min_prediction_error": float(np.min(self._prediction_errors)),
                    "toeplitz_condition_number": self._condition_number,
                },
                "fit_residual": self._fit_residual,
            },
        }


__all__ = ["ARNoiseSimulator", "FitError", "_stable_detector_hash"]
