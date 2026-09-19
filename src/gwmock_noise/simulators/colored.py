"""Colored noise simulator with overlap-add stitching."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from gwmock_noise.simulators._spectral import (
    load_spectral_series,
    median_frequency_spacing,
    normalize_spectral_reference,
)
from gwmock_noise.simulators._stitching import (
    DEFAULT_WINDOW_DURATION,
    OverlapAddStitcher,
    resolve_window_sizes,
    warn_if_underresolved,
)
from gwmock_noise.simulators.base import ConfigurableNoiseSimulator
from gwmock_noise.utils.log import LOGGER_NAME

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig

logger = logging.getLogger(LOGGER_NAME)

PSD_WINDOW_WIDTH_HZ = 1.0
MIN_TAPER_BINS = 2

#: Relative tolerance for a supplied psd_array frequency axis against the
#: simulator's own grid. It is tight on purpose: the array must be declared on
#: the simulator's exact grid, and an equal-length array from a different
#: spacing is an error rather than a silent reinterpretation.
FREQUENCY_RELATIVE_TOLERANCE = 1e-9


def _tukey_window(length: int, alpha: float) -> np.ndarray:
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


def _resolve_taper_alpha(masked_frequencies: np.ndarray) -> float:
    """Return the Tukey alpha that yields a PSD_WINDOW_WIDTH_HZ-wide taper on each edge.

    Bands with at most ``MIN_TAPER_BINS`` frequencies are too narrow to support a
    meaningful taper: the two-point difference can be zero (single bin, causing a
    division by zero) or small enough that the resulting alpha lands at or above 1,
    which collapses ``_tukey_window`` to a Hann window that can be entirely zero at
    these lengths (e.g. ``np.hanning(2) == [0, 0]``). Fall back to an untapered
    (all-ones) window in that case instead of destroying the band's PSD/CSD.
    """
    if masked_frequencies.size <= MIN_TAPER_BINS:
        return 0.0
    f_low_hz = masked_frequencies[0]
    f_high_hz = masked_frequencies[-1]
    return 2.0 * PSD_WINDOW_WIDTH_HZ / (f_high_hz - f_low_hz)


class ColoredNoiseSimulator(ConfigurableNoiseSimulator):
    """Generate colored detector noise from an input PSD."""

    simulator_name = "colored"

    def __init__(  # noqa: PLR0913
        self,
        *,
        psd_file: str | Path | None = None,
        psd_schedule: list[tuple[float, str | Path]] | None = None,
        psd_array: np.ndarray | None = None,
        frequencies: np.ndarray | None = None,
        delta_frequency: float | None = None,
        detectors: list[str] | None = None,
        sampling_frequency: float = 4096.0,
        duration: float = 4.0,
        seed: int | None = None,
        low_frequency_cutoff: float = 2.0,
        high_frequency_cutoff: float | None = None,
        window_duration: float = DEFAULT_WINDOW_DURATION,
    ) -> None:
        """Initialize the simulator."""
        sources = [psd_file, psd_schedule, psd_array]
        if all(source is None for source in sources):
            raise ValueError("Exactly one of psd_file, psd_schedule or psd_array must be provided.")
        if sum(source is not None for source in sources) > 1:
            raise ValueError("psd_file, psd_schedule and psd_array are mutually exclusive.")
        if psd_schedule is not None and not psd_schedule:
            raise ValueError("psd_schedule must contain at least one anchor.")
        if psd_array is None and (frequencies is not None or delta_frequency is not None):
            raise ValueError("frequencies and delta_frequency describe a psd_array and are only valid with it.")
        if psd_array is not None and frequencies is None and delta_frequency is None:
            raise ValueError(
                "psd_array requires its frequency axis: pass frequencies or delta_frequency, so an array built at a "
                "different spacing is rejected instead of silently reinterpreted on this simulator's grid."
            )
        if psd_array is not None and np.asarray(psd_array).ndim != 1:
            raise ValueError("psd_array must be one-dimensional on the simulator's frequency grid.")
        if frequencies is not None and np.asarray(frequencies).ndim != 1:
            raise ValueError("frequencies must be one-dimensional.")
        if (
            frequencies is not None
            and psd_array is not None
            and np.asarray(frequencies).shape != np.asarray(psd_array).shape
        ):
            raise ValueError("frequencies must have one value per psd_array bin.")
        if delta_frequency is not None and delta_frequency <= 0:
            raise ValueError(f"delta_frequency must be positive, got {delta_frequency!r}.")
        if psd_schedule is not None:
            offsets = [float(gps_offset_seconds) for gps_offset_seconds, _ in psd_schedule]
            if offsets != sorted(offsets):
                raise ValueError("psd_schedule entries must be sorted by GPS offset.")
            if len(offsets) != len(set(offsets)):
                raise ValueError("psd_schedule entries must use distinct GPS offsets.")

        self.psd_file = normalize_spectral_reference(psd_file) if psd_file is not None else None
        self.psd_schedule = (
            [
                (float(gps_offset_seconds), normalize_spectral_reference(path))
                for gps_offset_seconds, path in psd_schedule
            ]
            if psd_schedule is not None
            else None
        )
        self.psd_array = None if psd_array is None else np.asarray(psd_array, dtype=float).copy()
        self.frequencies = None if frequencies is None else np.asarray(frequencies, dtype=float).copy()
        self.delta_frequency = None if delta_frequency is None else float(delta_frequency)
        self.detectors = list(detectors) if detectors is not None else ["H1", "L1"]
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = high_frequency_cutoff
        self.window_duration = window_duration

        self._rngs: dict[str, np.random.Generator] = {}
        self._generated_samples = 0
        self._last_chunk_midpoint: float | None = None
        self._psd_anchors: list[tuple[float, np.ndarray]] = []

        self._validate_runtime(duration=duration, sampling_frequency=sampling_frequency, detectors=self.detectors)
        self._window_size, self._overlap_size = resolve_window_sizes(window_duration, sampling_frequency)
        self._stitcher = OverlapAddStitcher(
            self.detectors,
            window_size=self._window_size,
            overlap_size=self._overlap_size,
        )
        self._configure_frequency_grid()

    @classmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> ColoredNoiseSimulator:
        """Construct a colored-noise simulator from one component definition."""
        options = dict(component.options)
        psd_file = options.pop("psd_file", None)
        psd_schedule = options.pop("psd_schedule", None)
        return cls(
            psd_file=psd_file,
            psd_schedule=psd_schedule,
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

        self._psd_input_spacing = np.inf
        self._psd_anchors = self._load_psd_anchors()
        self._psd = self._interpolate_psd(0.0)
        warn_if_underresolved(
            delta_frequency=self._delta_frequency,
            low_frequency_cutoff=self.low_frequency_cutoff,
            reference_spacing=self._psd_input_spacing,
            logger=logger,
            context="colored noise simulator",
        )

    def _load_psd_on_grid(self, psd_file: str | Path) -> np.ndarray:
        """Load one PSD file onto the simulator frequency grid."""
        psd = np.zeros_like(self._frequency_grid, dtype=float)
        masked_frequencies = self._frequency_grid[self._frequency_mask]
        psd_frequencies, psd_values = load_spectral_series(psd_file, kind="PSD")
        self._psd_input_spacing = min(self._psd_input_spacing, median_frequency_spacing(psd_frequencies))
        interpolated_psd = np.interp(masked_frequencies, psd_frequencies, psd_values, left=0.0, right=0.0)
        psd[self._frequency_mask] = np.clip(interpolated_psd, a_min=0.0, a_max=None)
        psd[self._frequency_mask] *= _tukey_window(
            masked_frequencies.size, alpha=_resolve_taper_alpha(masked_frequencies)
        )
        return psd

    def _load_psd_anchors(self) -> list[tuple[float, np.ndarray]]:
        """Load all configured PSD anchors onto the current simulator grid."""
        if self.psd_array is not None:
            return [(0.0, self._array_psd_on_grid())]
        anchors = self.psd_schedule or [(0.0, self.psd_file)]
        return [
            (gps_offset_seconds, self._load_psd_on_grid(psd_path))
            for gps_offset_seconds, psd_path in anchors
            if psd_path is not None
        ]

    def _array_psd_on_grid(self) -> np.ndarray:
        """Return the supplied PSD array on the simulator grid, untapered.

        The array path exists for targets that are defined analytically on the
        window's own frequency grid rather than loaded from a table. It applies
        no interpolation and no Tukey taper --- either would change the target
        --- and zeroes the bins outside the configured band, exactly as the
        file path does after its interpolation.

        The array's frequency axis is checked against the simulator's own grid
        before the array is accepted, so an equal-length array built at a
        different spacing raises instead of being reinterpreted.

        Returns:
            The one-sided PSD on ``self._frequency_grid``.

        Raises:
            ValueError: If the array's length differs from the frequency grid's,
                or if its declared frequency axis does not match the grid.
        """
        if self.psd_array.shape != self._frequency_grid.shape:
            raise ValueError(
                f"psd_array must have one value per frequency grid bin: expected shape "
                f"{self._frequency_grid.shape}, got {self.psd_array.shape}."
            )
        self._validate_array_frequency_axis()
        psd = self.psd_array.copy()
        psd[~self._frequency_mask] = 0.0
        return np.clip(psd, a_min=0.0, a_max=None)

    def _validate_array_frequency_axis(self) -> None:
        """Check the supplied psd_array frequency axis against the simulator grid.

        The contract is that the array is handed over on the simulator's own
        grid, ``sampling_frequency / window_size``. A declared axis --- the
        full ``frequencies`` grid or a scalar ``delta_frequency`` --- is
        compared with that grid within :data:`FREQUENCY_RELATIVE_TOLERANCE`,
        and a disagreement is an error naming both sides.

        Raises:
            ValueError: If the declared frequencies are negative or not
                strictly increasing, or if the declared axis does not match the
                simulator's grid.
        """
        expected = self._delta_frequency
        if self.frequencies is not None:
            supplied = self.frequencies
            if supplied.shape != self._frequency_grid.shape:
                raise ValueError(
                    f"frequencies must have one value per frequency grid bin: expected shape "
                    f"{self._frequency_grid.shape}, got {supplied.shape}."
                )
            if np.any(supplied < 0.0):
                raise ValueError(f"frequencies must be non-negative, got a minimum of {float(np.min(supplied))} Hz.")
            if np.any(np.diff(supplied) <= 0.0):
                raise ValueError("frequencies must be strictly increasing.")
            if not np.allclose(supplied, self._frequency_grid, rtol=FREQUENCY_RELATIVE_TOLERANCE, atol=0.0):
                supplied_spacing = float(np.median(np.diff(supplied))) if supplied.size > 1 else float("nan")
                raise ValueError(
                    f"frequencies do not match the simulator frequency grid: supplied spacing "
                    f"{supplied_spacing!r} Hz, expected {expected!r} Hz "
                    f"(sampling_frequency / window_size)."
                )
        if self.delta_frequency is not None and not np.isclose(
            self.delta_frequency, expected, rtol=FREQUENCY_RELATIVE_TOLERANCE, atol=0.0
        ):
            raise ValueError(
                f"delta_frequency {self.delta_frequency!r} Hz does not match the simulator frequency grid spacing "
                f"{expected!r} Hz (sampling_frequency / window_size)."
            )

    def _interpolate_psd(self, t: float) -> np.ndarray:
        """Interpolate the PSD schedule log-linearly at frame midpoint time ``t``."""
        if len(self._psd_anchors) == 1:
            return self._psd_anchors[0][1].copy()

        anchor_times = np.array([time for time, _ in self._psd_anchors], dtype=float)
        if t <= anchor_times[0]:
            return self._psd_anchors[0][1].copy()
        if t >= anchor_times[-1]:
            return self._psd_anchors[-1][1].copy()

        upper_index = int(np.searchsorted(anchor_times, t, side="right"))
        lower_time, lower_psd = self._psd_anchors[upper_index - 1]
        upper_time, upper_psd = self._psd_anchors[upper_index]
        weight = (t - lower_time) / (upper_time - lower_time)

        interpolated = np.zeros_like(self._frequency_grid, dtype=float)
        lower_masked = np.maximum(lower_psd[self._frequency_mask], np.finfo(float).tiny)
        upper_masked = np.maximum(upper_psd[self._frequency_mask], np.finfo(float).tiny)
        log_psd = ((1.0 - weight) * np.log(lower_masked)) + (weight * np.log(upper_masked))
        interpolated[self._frequency_mask] = np.exp(log_psd)
        return interpolated

    def _simulate(self, *, n_samples: int) -> dict[str, np.ndarray]:
        """Generate a prefix-consistent realization while updating the PSD per frame."""
        overlap_size = self._stitcher.overlap_size
        frame_step = self._stitcher.window_size - overlap_size
        next_frame_midpoint = self._generated_samples if self.previous_strain else -frame_step

        def chunk_generator() -> dict[str, np.ndarray]:
            nonlocal next_frame_midpoint
            self._psd = self._interpolate_psd(next_frame_midpoint / self.sampling_frequency)
            self._last_chunk_midpoint = next_frame_midpoint
            next_frame_midpoint += frame_step
            return self._generate_realization_chunk()

        history = self.previous_strain
        if history:
            self._stitcher._validate_chunk_map(history)
            current_raw = {detector: history[detector].copy() for detector in self.detectors}
        else:
            warmup = self._stitcher.draw_chunk(chunk_generator)
            current_raw = {detector: warmup[detector].copy() for detector in self.detectors}

        emitted_segments = {detector: [] for detector in self.detectors}
        produced_samples = 0

        while produced_samples < n_samples:
            next_raw = self._stitcher.draw_chunk(chunk_generator)

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

    def _initialize_generators(self, seed: int | None) -> None:
        """Initialize per-detector random-number generators."""
        seed_sequence = np.random.SeedSequence() if seed is None else np.random.SeedSequence(seed)
        child_sequences = seed_sequence.spawn(len(self.detectors))
        self._rngs = {
            detector: np.random.default_rng(child_sequence)
            for detector, child_sequence in zip(self.detectors, child_sequences, strict=True)
        }
        self._stitcher.bind_rngs(self._rngs)

    def _generate_single_realization(self, detector: str) -> np.ndarray:
        """Generate one colored-noise chunk for a single detector."""
        rng = self._rngs[detector]
        n_frequencies = int(np.count_nonzero(self._frequency_mask))
        white_noise = (rng.standard_normal(n_frequencies) + 1j * rng.standard_normal(n_frequencies)) / np.sqrt(2.0)

        frequency_series = np.zeros(self._frequency_grid.size, dtype=np.complex128)
        frequency_series[self._frequency_mask] = white_noise * np.sqrt(
            self._psd[self._frequency_mask] * 0.5 / self._delta_frequency
        )
        return np.fft.irfft(frequency_series, n=self._window_size) * self._delta_frequency * self._window_size

    def _generate_realization_chunk(self) -> dict[str, np.ndarray]:
        """Generate one colored-noise chunk for every configured detector."""
        return {detector: self._generate_single_realization(detector) for detector in self.detectors}

    def reset(self) -> None:
        """Clear continuity and RNG state."""
        self._stitcher.reset()
        self._rngs = {}
        self._generated_samples = 0
        self._last_chunk_midpoint = None
        if self._psd_anchors:
            self._psd = self._psd_anchors[0][1].copy()

    def export_state(self) -> dict[str, Any]:
        """Return a picklable snapshot of the running crossfade stream.

        The snapshot persists the bit-generator state and the chunk counter of
        the stitcher plus the bookkeeping needed to interpolate the PSD schedule
        on resume. It contains no cached strain: the previous window is
        regenerated from the bit-generator state.
        """
        return {
            "generated_samples": self._generated_samples,
            "last_chunk_midpoint": self._last_chunk_midpoint,
            "stitcher": self._stitcher.export_state(),
        }

    def import_state(self, state: dict[str, Any]) -> None:
        """Resume the crossfade stream from an :meth:`export_state` snapshot.

        Args:
            state: A snapshot from another simulator with the same settings.

        Raises:
            ValueError: If the snapshot does not match this simulator.
        """
        if self._psd_anchors is None or not self._psd_anchors:
            raise ValueError("The simulator must be configured before importing state.")
        if not self._rngs:
            self._initialize_generators(self.seed)

        self._generated_samples = int(state["generated_samples"])
        self._last_chunk_midpoint = state["last_chunk_midpoint"]
        if self._last_chunk_midpoint is not None:
            self._psd = self._interpolate_psd(self._last_chunk_midpoint / self.sampling_frequency)
        self._stitcher.import_state(state["stitcher"], self._generate_realization_chunk)

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
            self._window_size, self._overlap_size = resolve_window_sizes(self.window_duration, self.sampling_frequency)
            self._stitcher = OverlapAddStitcher(
                self.detectors,
                window_size=self._window_size,
                overlap_size=self._overlap_size,
            )
            self.reset()
            self._configure_frequency_grid()

        if seed is not None:
            self.seed = seed
            self.reset()

        if not self._rngs:
            self._initialize_generators(self.seed)

        n_samples = round(duration * sampling_frequency)
        realization = self._simulate(n_samples=n_samples)
        self._generated_samples += n_samples
        return realization

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield colored-noise chunks lazily while preserving simulator state."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

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
                "psd_file": str(self.psd_file) if self.psd_file is not None else None,
                "psd_array": None if self.psd_array is None else int(self.psd_array.size),
                "psd_schedule": [
                    {"gps_offset_seconds": gps_offset_seconds, "psd_file": str(psd_path)}
                    for gps_offset_seconds, psd_path in (self.psd_schedule or [])
                ],
                "low_frequency_cutoff": self.low_frequency_cutoff,
                "high_frequency_cutoff": self._high_frequency_cutoff,
                "window_duration": self.window_duration,
                "window_size": self._window_size,
                "overlap_size": self._overlap_size,
            },
        }


TimeVaryingColoredNoiseSimulator = ColoredNoiseSimulator
