"""Overlap-save FIR colouring noise simulator.

This module implements a bounded-state streaming generator. A colouring filter
is designed from a target spectrum as the windowed inverse transform of the
target's square root (its amplitude spectral density), optionally reduced to
minimum phase by cepstral factorisation, and applied to white noise with
overlap-save block convolution. The continuation state of a running stream is
the filter memory plus the pending block bookkeeping, so it is bounded and
independent of the generated span.

The filter length ``filter_length`` (``L_f``) is the truncation control: a
longer filter carries more of the target spectrum and state, a shorter one
trades covariance accuracy for state bytes. The public design sweep uses the
powers of two from ``2**4`` to ``2**16`` samples and a Hann design window.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.fft import next_fast_len

from gwmock_noise.simulators._spectral import (
    load_spectral_series,
    median_frequency_spacing,
    normalize_spectral_reference,
)
from gwmock_noise.simulators._stitching import warn_if_underresolved
from gwmock_noise.simulators.autoregressive import _stable_detector_hash
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.simulators.colored import _resolve_taper_alpha, _tukey_window
from gwmock_noise.utils.log import LOGGER_NAME

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

logger = logging.getLogger(LOGGER_NAME)

MIN_FILTER_LENGTH = 1 << 4
MAX_FILTER_LENGTH = 1 << 16
DEFAULT_FILTER_LENGTH = 1 << 9
DEFAULT_BLOCK_SIZE = 1 << 13

# The cepstral factorisation takes the logarithm of the amplitude response, so
# exact zeros (band edges, masked bins) are floored at this fraction of the
# peak amplitude instead of at an absolute value.
MIN_MAGNITUDE_FRACTION = 1e-12

# A one-sided spectrum needs at least a DC bin, a Nyquist-or-interior bin and
# one more before an amplitude response is meaningful.
MIN_SPECTRAL_SAMPLES = 3

# Two points are the minimum a frequency grid can be interpolated from.
MIN_INTERPOLATION_POINTS = 2

# Number of design-grid samples per filter tap. A filter cannot resolve more
# structure than a few samples per tap, and the grid must also carry the input
# spectrum's own resolution, so the design grid is the larger of the two.
DESIGN_OVERSAMPLING = 16

RESUME_STATE_FORMAT_VERSION = 1


def _is_power_of_two(value: int) -> bool:
    """Return whether ``value`` is a positive power of two."""
    return value > 0 and (value & (value - 1)) == 0


def validate_filter_length(filter_length: int) -> None:
    """Validate the documented truncation control.

    Args:
        filter_length: Filter length in samples.

    Raises:
        ValueError: If ``filter_length`` is not a power of two in the documented
            sweep range.
    """
    if not _is_power_of_two(filter_length):
        raise ValueError("filter_length must be a power of two.")
    if not MIN_FILTER_LENGTH <= filter_length <= MAX_FILTER_LENGTH:
        raise ValueError(
            f"filter_length must lie in [{MIN_FILTER_LENGTH}, {MAX_FILTER_LENGTH}] samples "
            f"(2**4 to 2**16), got {filter_length}."
        )


def minimum_phase_from_magnitude(magnitude: np.ndarray) -> np.ndarray:
    """Return a minimum-phase impulse response with the given magnitude response.

    The factorisation is the real-cepstrum method: the logarithm of the
    magnitude is folded so that its causal part is doubled, exponentiated back
    to a spectrum and inverse-transformed. The result is real and causal, and
    its magnitude response equals ``magnitude`` up to the floored bins.

    Args:
        magnitude: One-sided magnitude response on a uniformly spaced grid,
            with ``magnitude.size > 2``.

    Returns:
        A real minimum-phase sequence whose length is ``2 * (magnitude.size - 1)``.
    """
    magnitude = np.asarray(magnitude, dtype=float)
    if magnitude.ndim != 1 or magnitude.size < MIN_SPECTRAL_SAMPLES:
        raise ValueError("magnitude must be a one-dimensional array with more than two samples.")
    if not np.all(np.isfinite(magnitude)):
        raise ValueError("magnitude must be finite.")

    design_size = 2 * (magnitude.size - 1)
    floor = MIN_MAGNITUDE_FRACTION * float(np.max(magnitude, initial=0.0))
    log_magnitude = np.log(np.maximum(magnitude, max(floor, np.finfo(float).tiny)))

    cepstrum = np.fft.irfft(log_magnitude, n=design_size)
    folded = np.zeros(design_size, dtype=float)
    folded[0] = cepstrum[0]
    folded[1 : design_size // 2] = 2.0 * cepstrum[1 : design_size // 2]
    folded[design_size // 2] = cepstrum[design_size // 2]

    log_spectrum = np.fft.fft(folded)
    return np.fft.ifft(np.exp(log_spectrum)).real


def design_colouring_filter(
    target_psd: np.ndarray,
    *,
    delta_frequency: float,
    filter_length: int,
    minimum_phase: bool = True,
) -> np.ndarray:
    """Design a causal FIR colouring filter from a one-sided target PSD.

    The filter is the inverse transform of the target's square root, truncated
    to ``filter_length`` samples around zero lag and tapered with a Hann design
    window. When ``minimum_phase`` is set, the windowed response is reduced to
    minimum phase by cepstral factorisation so its energy is concentrated at
    the front and the truncation loses less of the target. The taps are then
    rescaled so the output variance equals the target band variance
    ``sum(target_psd) * delta_frequency``.

    Args:
        target_psd: One-sided target PSD, length ``design_size // 2 + 1``.
        delta_frequency: Spacing of the target PSD grid in hertz.
        filter_length: Number of filter taps (a power of two in the documented
            range, see :func:`validate_filter_length`).
        minimum_phase: Whether to apply the cepstral minimum-phase factorisation.

    Returns:
        The ``filter_length`` filter taps as a real array.

    Raises:
        ValueError: If the arguments are inconsistent or the target integrates
            to zero variance.
    """
    validate_filter_length(filter_length)
    target_psd = np.asarray(target_psd, dtype=float)
    if target_psd.ndim != 1 or target_psd.size < MIN_SPECTRAL_SAMPLES:
        raise ValueError("target_psd must be a one-dimensional array with more than two samples.")
    if delta_frequency <= 0.0:
        raise ValueError("delta_frequency must be greater than zero.")

    design_size = 2 * (target_psd.size - 1)
    if design_size < filter_length:
        raise ValueError("the target grid is too coarse for the requested filter_length.")

    amplitude = np.sqrt(np.clip(target_psd, a_min=0.0, a_max=None))
    impulse = np.fft.irfft(amplitude, n=design_size)

    half = filter_length // 2
    centered = np.concatenate((impulse[design_size - half :], impulse[: filter_length - half]))
    windowed = centered * np.hanning(filter_length)

    if minimum_phase:
        response = np.abs(np.fft.rfft(windowed, n=design_size))
        taps = minimum_phase_from_magnitude(response)[:filter_length]
    else:
        taps = windowed

    target_variance = delta_frequency * float(np.sum(target_psd))
    energy = float(np.sum(taps**2))
    if target_variance <= 0.0 or energy <= 0.0:
        raise ValueError("the target PSD integrates to zero variance in the requested band.")
    return taps * np.sqrt(target_variance / energy)


class OverlapSaveFilter:
    """Apply a fixed causal FIR filter to a sample stream with overlap-save.

    The continuation state is the filter memory only: the ``len(taps) - 1``
    input samples that the next block still needs. Its size is independent of
    the generated span.
    """

    def __init__(self, taps: np.ndarray, *, block_size: int = DEFAULT_BLOCK_SIZE) -> None:
        """Initialize the convolver.

        Args:
            taps: One-dimensional FIR coefficients.
            block_size: Number of input samples processed per transform.

        Raises:
            ValueError: If the taps or block size are invalid.
        """
        self.taps = np.asarray(taps, dtype=float)
        if self.taps.ndim != 1 or self.taps.size == 0:
            raise ValueError("taps must be a non-empty one-dimensional array.")
        if block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        self.block_size = block_size
        self.fft_length = next_fast_len(block_size + self.taps.size - 1)
        self._spectrum = np.fft.rfft(self.taps, n=self.fft_length)
        self.filter_memory = np.zeros(max(self.taps.size - 1, 0), dtype=float)

    @property
    def state_nbytes(self) -> int:
        """Return the serialized size of the filter memory in bytes."""
        return int(self.filter_memory.nbytes)

    def reset(self) -> None:
        """Clear the filter memory."""
        self.filter_memory = np.zeros(max(self.taps.size - 1, 0), dtype=float)

    def process(self, block: np.ndarray) -> np.ndarray:
        """Convolve one block with the filter and return its causal output.

        Args:
            block: Exactly ``block_size`` input samples.

        Returns:
            The ``block_size`` linear-convolution outputs aligned with ``block``.

        Raises:
            ValueError: If ``block`` does not have the configured length.
        """
        block = np.asarray(block, dtype=float)
        if block.shape != (self.block_size,):
            raise ValueError(f"block must have shape ({self.block_size},).")

        buffer = np.concatenate((self.filter_memory, block))
        spectrum = np.fft.rfft(buffer, n=self.fft_length)
        convolved = np.fft.irfft(spectrum * self._spectrum, n=self.fft_length)

        start = max(self.taps.size - 1, 0)
        valid = convolved[start : start + self.block_size]
        if self.taps.size > 1:
            self.filter_memory = buffer[-(self.taps.size - 1) :].copy()
        return valid


class OverlapSaveFirSimulator(ConfigurableNoiseSimulator):
    """Generate coloured detector noise with a minimum-phase overlap-save FIR."""

    simulator_name = "overlap_save_fir"

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_file: str | Path | None = None,
        target_psd: np.ndarray | None = None,
        target_frequencies: np.ndarray | None = None,
        filter_length: int = DEFAULT_FILTER_LENGTH,
        minimum_phase: bool = True,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        """Initialize the simulator and design the colouring filter once.

        The target may be given either as a file (``psd_file``) or directly as a
        one-sided array (``target_psd``), so an analytic target can bypass the
        tabulated-curve interpolation entirely. When ``target_frequencies`` is
        omitted for an array target it defaults to the one-sided grid matching
        its length at the configured sampling frequency.

        Args:
            psd_file: Path to a two-column PSD table.
            target_psd: One-sided target PSD array, mutually exclusive with
                ``psd_file``.
            target_frequencies: Frequencies of ``target_psd`` in hertz.
            filter_length: Truncation control ``L_f`` in samples.
            minimum_phase: Whether to apply the cepstral minimum-phase factorisation.
            detectors: Detector names to generate.
            sampling_frequency: Sampling frequency in hertz.
            duration: Nominal generation duration in seconds.
            seed: Base seed for the per-detector generators.
            low_frequency_cutoff: Lower edge of the target band in hertz.
            high_frequency_cutoff: Upper edge of the target band in hertz.
            block_size: Overlap-save block length in samples.
        """
        if (psd_file is None) == (target_psd is None):
            raise ValueError("Exactly one of psd_file or target_psd must be provided.")
        if block_size < 1:
            raise ValueError("block_size must be a positive integer.")
        validate_filter_length(filter_length)

        self.psd_file = normalize_spectral_reference(psd_file) if psd_file is not None else None
        self.filter_length = filter_length
        self.minimum_phase = minimum_phase
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.sampling_frequency = sampling_frequency
        self.duration = duration
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.block_size = block_size

        self._input_spacing = np.inf
        if target_psd is not None:
            self._target_frequencies, self._target_values = self._resolve_array_target(target_psd, target_frequencies)
        else:
            self._target_frequencies, self._target_values = load_spectral_series(self.psd_file, kind="PSD")
        self._input_spacing = median_frequency_spacing(self._target_frequencies)

        self._rngs: dict[str, np.random.Generator] = {}
        self._pending: dict[str, np.ndarray] = {}
        self._generated_samples = 0
        self._block_counter = 0
        self._fit_time_seconds = 0.0
        self._design_size = 0
        self._high_frequency_cutoff = 0.0

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._configure()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> OverlapSaveFirSimulator:
        """Construct an overlap-save FIR simulator from one component definition."""
        options = dict(component.options)
        psd_file = options.pop("psd_file", None)
        target_psd = options.pop("target_psd", None)
        if psd_file is None and target_psd is None:
            raise ValueError("Overlap-save FIR simulator requires 'psd_file' or 'target_psd' in the component options.")
        return cls(
            psd_file=psd_file,
            target_psd=target_psd,
            detectors=config.detectors,
            duration=config.duration,
            sampling_frequency=config.sampling_frequency,
            seed=config.seed,
            **options,
        )

    def _resolve_array_target(
        self,
        target_psd: np.ndarray,
        target_frequencies: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize an array target into sorted ``(frequencies, values)``."""
        values = np.asarray(target_psd, dtype=float)
        if values.ndim != 1 or values.size < MIN_SPECTRAL_SAMPLES:
            raise ValueError("target_psd must be a one-dimensional array with at least three samples.")
        if target_frequencies is None:
            frequencies = np.fft.rfftfreq(2 * (values.size - 1), d=1.0 / self.sampling_frequency)
            return frequencies, values
        frequencies = np.asarray(target_frequencies, dtype=float)
        if frequencies.shape != values.shape:
            raise ValueError("target_frequencies and target_psd must have the same length.")
        if not np.all(np.diff(frequencies) > 0.0):
            raise ValueError("target_frequencies must be strictly increasing.")
        return frequencies, values

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

    def _configure(self) -> None:
        """Build the design grid, the target array and the colouring filter."""
        configure_start = time.perf_counter()
        design_size = DESIGN_OVERSAMPLING * self.filter_length
        if self._target_values.size >= MIN_INTERPOLATION_POINTS:
            design_size = max(design_size, 2 * (self._target_values.size - 1))
        design_size += design_size % 2
        self._design_size = design_size

        self._design_grid = np.fft.rfftfreq(design_size, d=1.0 / self.sampling_frequency)
        self._delta_frequency = self.sampling_frequency / design_size
        self._high_frequency_cutoff = (
            self.high_frequency_cutoff if self.high_frequency_cutoff is not None else self.sampling_frequency / 2.0
        )
        self._frequency_mask = (self._design_grid >= self.low_frequency_cutoff) & (
            self._design_grid <= self._high_frequency_cutoff
        )
        if not np.any(self._frequency_mask):
            raise ValueError("The requested frequency range contains no design bins.")

        masked_frequencies = self._design_grid[self._frequency_mask]
        target = np.zeros(design_size // 2 + 1, dtype=float)
        interpolated = np.interp(
            masked_frequencies,
            self._target_frequencies,
            self._target_values,
            left=0.0,
            right=0.0,
        )
        target[self._frequency_mask] = np.clip(interpolated, a_min=0.0, a_max=None)
        target[self._frequency_mask] *= _tukey_window(
            masked_frequencies.size, alpha=_resolve_taper_alpha(masked_frequencies)
        )
        self._target_psd = target

        warn_if_underresolved(
            delta_frequency=self._delta_frequency,
            low_frequency_cutoff=self.low_frequency_cutoff,
            reference_spacing=self._input_spacing,
            logger=logger,
            context="overlap-save FIR simulator",
        )

        self._filter = design_colouring_filter(
            target,
            delta_frequency=self._delta_frequency,
            filter_length=self.filter_length,
            minimum_phase=self.minimum_phase,
        )
        self._refresh_convolvers()
        self._fit_time_seconds = time.perf_counter() - configure_start

    def _refresh_convolvers(self) -> None:
        """Create one overlap-save convolver per detector."""
        self._convolvers = {
            detector: OverlapSaveFilter(self._filter, block_size=self.block_size) for detector in self.detectors
        }
        self._pending = {detector: np.zeros(0, dtype=float) for detector in self.detectors}

    def _initialize_generators(self, seed: int | None) -> None:
        """Initialize one random-number generator per detector."""
        self._rngs = {
            detector: np.random.default_rng(np.random.SeedSequence([seed, _stable_detector_hash(detector)]))
            for detector in self.detectors
        }

    def reset(self) -> None:
        """Clear the filter memory, pending samples and random-number generators."""
        self._rngs = {}
        self._block_counter = 0
        self._generated_samples = 0
        self._refresh_convolvers()

    def _pull(self, detector: str, n_samples: int) -> np.ndarray:
        """Return the next ``n_samples`` of filtered noise for one detector."""
        segments: list[np.ndarray] = []
        remaining = n_samples
        while remaining > 0:
            pending = self._pending[detector]
            if pending.size:
                take = min(remaining, pending.size)
                segments.append(pending[:take])
                self._pending[detector] = pending[take:]
                remaining -= take
                continue

            block = self._rngs[detector].standard_normal(self.block_size)
            self._pending[detector] = self._convolvers[detector].process(block)
            self._block_counter += 1
        return np.concatenate(segments)

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector coloured noise with continuity across calls."""
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
            self._configure()
            self.reset()

        if seed is not None:
            self.seed = seed
            self.reset()

        if not self._rngs:
            self._initialize_generators(self.seed)

        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        realization = {detector: self._pull(detector, n_samples) for detector in self.detectors}
        self._generated_samples += n_samples
        return realization

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield coloured-noise chunks lazily while preserving filter memory."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    def continuation_state(self) -> dict[str, Any]:
        """Return the bounded state needed to keep a running stream going.

        The state is the filter memory of every detector plus the pending output
        and block bookkeeping. It contains no generated strain and its size is
        independent of the generated span.
        """
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "block_counter": self._block_counter,
            "generated_samples": self._generated_samples,
            "filter_memory": {
                detector: convolver.filter_memory.copy() for detector, convolver in self._convolvers.items()
            },
            "pending_samples": {detector: pending.copy() for detector, pending in self._pending.items()},
        }

    def resume_metadata(self) -> dict[str, Any]:
        """Return the metadata a stopped stream must persist to resume.

        This is the per-detector bit-generator state plus the settings needed to
        rebuild the filter, and is reported separately from the continuation
        state.
        """
        return {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "filter_length": self.filter_length,
            "minimum_phase": self.minimum_phase,
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
        if resume["detectors"] != self.detectors:
            raise ValueError("resume metadata detectors do not match this simulator.")
        if resume["filter_length"] != self.filter_length or resume["minimum_phase"] != self.minimum_phase:
            raise ValueError("resume metadata filter settings do not match this simulator.")
        if resume["sampling_frequency"] != self.sampling_frequency:
            raise ValueError("resume metadata sampling frequency does not match this simulator.")

        self.seed = resume["seed"]
        self._initialize_generators(self.seed)
        for detector, rng_state in resume["rng_state"].items():
            self._rngs[detector].bit_generator.state = rng_state

        self._refresh_convolvers()
        for detector in self.detectors:
            self._convolvers[detector].filter_memory = np.asarray(
                continuation["filter_memory"][detector], dtype=float
            ).copy()
            self._pending[detector] = np.asarray(continuation["pending_samples"][detector], dtype=float).copy()
        self._block_counter = int(continuation["block_counter"])
        self._generated_samples = int(continuation["generated_samples"])

    @property
    def target_psd(self) -> np.ndarray:
        """Return a copy of the band-masked target PSD the filter was designed from."""
        return self._target_psd.copy()

    @property
    def filter_taps(self) -> np.ndarray:
        """Return a copy of the designed colouring filter taps."""
        return self._filter.copy()

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
        """Return simulator metadata."""
        return {
            "implementation": "overlap_save_fir",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "overlap_save_fir": {
                "psd_file": str(self.psd_file) if self.psd_file is not None else None,
                "filter_length": self.filter_length,
                "minimum_phase": self.minimum_phase,
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "block_size": self.block_size,
                "design_size": self._design_size,
                "delta_frequency": self._delta_frequency,
                "fit_time_seconds": self._fit_time_seconds,
                "state_bytes": self.state_nbytes,
                "resume_metadata_bytes": self.resume_metadata_nbytes,
            },
        }
