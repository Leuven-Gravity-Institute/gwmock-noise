"""Autoregressive noise simulator with persistent detector state."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

DEFAULT_AR_ORDER = 256
DEFAULT_BLOCK_SIZE = 65_536
DEFAULT_REGULARIZATION = 1e-10


def _stable_detector_hash(detector: str) -> int:
    """Return a stable integer hash for a detector label."""
    digest = hashlib.blake2b(detector.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _toeplitz_from_autocorrelation(autocorrelation: np.ndarray) -> np.ndarray:
    """Build a symmetric Toeplitz matrix from autocorrelation lags."""
    order = autocorrelation.size
    offsets = np.abs(np.subtract.outer(np.arange(order), np.arange(order)))
    return autocorrelation[offsets]


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
        """Initialize the simulator and fit the AR model once."""
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
        self._innovation_variance = 0.0
        self._fit_time_seconds = 0.0
        self._fit_grid_size = 0
        self._high_frequency_cutoff = 0.0

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

        autocorrelation = np.fft.irfft(target_psd * self.sampling_frequency / 2.0, n=self._fit_grid_size)
        autocorrelation = np.asarray(autocorrelation[: self.order + 1], dtype=float)
        if autocorrelation[0] <= 0.0:
            raise ValueError("Target PSD integrates to zero variance in the requested band.")

        toeplitz_system = _toeplitz_from_autocorrelation(autocorrelation[:-1])
        if self.regularization > 0.0:
            diagonal_boost = self.regularization * max(autocorrelation[0], 1.0)
            toeplitz_system = toeplitz_system + diagonal_boost * np.eye(self.order)

        self._ar_coefficients = np.linalg.solve(toeplitz_system, -autocorrelation[1 : self.order + 1])
        self._innovation_variance = autocorrelation[0] + float(
            np.dot(self._ar_coefficients, autocorrelation[1 : self.order + 1])
        )
        if self._innovation_variance <= 0.0:
            raise ValueError("Innovation variance must be positive after fitting the AR model.")

        roots = np.roots(np.concatenate(([1.0], self._ar_coefficients)))
        if roots.size and np.max(np.abs(roots)) >= 1.0:
            raise ValueError("Fitted AR model is unstable; increase regularization or lower the order.")

        self._fit_time_seconds = time.perf_counter() - fit_start
        self._state = {detector: np.zeros(self.order, dtype=float) for detector in self.detectors}

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
        coefficients = self._ar_coefficients
        innovation_scale = float(np.sqrt(self._innovation_variance))
        state = self._state.setdefault(detector, np.zeros(self.order, dtype=float))
        innovations = self._rngs[detector].standard_normal(n_samples)
        strain = np.empty(n_samples, dtype=float)

        for index, innovation in enumerate(innovations):
            sample = innovation_scale * innovation - np.dot(coefficients, state)
            strain[index] = sample
            if self.order > 1:
                state[1:] = state[:-1]
            state[0] = sample

        self._state[detector] = state
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
                "order": self.order,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "block_size": self.block_size,
                "fit_grid_size": self._fit_grid_size,
                "fit_time_seconds": self._fit_time_seconds,
                "regularization": self.regularization,
                "innovation_variance": self._innovation_variance,
            },
        }
