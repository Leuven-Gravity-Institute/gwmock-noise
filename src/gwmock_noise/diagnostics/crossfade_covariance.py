"""Evaluate the covariance error of the colored-noise simulator's crossfade.

:class:`~gwmock_noise.simulators.ColoredNoiseSimulator` draws independent chunks and blends
consecutive ones over half a window. The result is not stationary: it is cyclostationary
with the period of the window stride, and its second moments differ from those of the target
spectrum. This module computes that difference exactly, from the algebra of the crossfade
rather than from realizations, as the total covariance error over an analysis band ``B``::

    eps_tot^2 = int_B e^2(f) S^2(f) df / int_B S^2 df,
    e^2(f) S^2(f) = (S_0 - S)^2 + sum_{k=1..K} [ |S^{(k)}(f)|^2 1(|f - alpha_k| in B)
                                                + |S^{(k)}(f + alpha_k)|^2 1(f + alpha_k in B) ]

``S`` is the target, ``S_0`` the stationary spectrum of the stitched process, and
``S^{(k)}(f)`` its cyclic spectrum at harmonic ``k``: the covariance of the Fourier
components at ``f`` and ``f - alpha_k``, with ``alpha_k = k f_s / P`` and ``P`` the stride.
This is the squared Hilbert-Schmidt error of the covariance operator of the real process
restricted to the band, divided by that of the target. A partner component below zero
frequency, or beyond the Nyquist frequency, is the component at the folded frequency.

The cyclic sum is taken harmonic by harmonic until doubling the number of harmonics changes
``eps_tot^2`` by less than a relative tolerance, no earlier than twice the harmonic at which
pairs start to straddle zero or the Nyquist frequency. A sum that reaches the cap first, or
whose straddling pairs lie beyond the cap so that it stops at the first check, omits part of
the cyclic error and is marked not converged: it is not an upper bound. Long windows stop the
second way, at the first check or soon after it (16 or 32 harmonics for the bundled curves).

This is the method, and the code, of

    I. C. F. Wong, "Choosing a generator for streaming Gaussian noise in
    gravitational-wave mock data: accuracy, cost and reproducibility",
    arXiv:2611.01624; data: doi:10.5281/zenodo.23243286,

whose evaluated errors ``ColoredNoiseSimulator(covariance_tolerance=...)`` reads. The target is
the paper's continued target (:func:`crossfade_target`), and the simulator is run from zero to
the Nyquist frequency (:func:`simulator_settings`), so an error evaluated here is comparable
with the bundled ones. It describes that configuration, not the PSD table run with the
simulator's defaults: to generate the noise it describes, write the target to a file and pass
it with those settings. ``gwmock-noise evaluate-covariance-error`` does both from the command
line.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from gwmock_noise.data.covariance_tolerance import load_crossfade_errors
from gwmock_noise.simulators._covariance_tolerance import NYQUIST_GUARD_HZ, continued_psd, continued_target, error_band
from gwmock_noise.simulators._spectral import load_spectral_series

#: Grid points per ``f_s / (4 P)``, the scale on which the crossfade's spectra vary.
OVERSAMPLE = 4

#: The relative change of ``eps_tot^2``, on doubling the number of harmonics, that counts as converged.
CONVERGENCE_TOLERANCE = 1e-3

#: The first harmonic count at which convergence is tested, and the most harmonics summed.
FIRST_CONVERGENCE_CHECK = 16
MAXIMUM_HARMONICS = 16384

#: Fewest band bins an error is integrated over.
MINIMUM_BINS = 2


@dataclass(frozen=True)
class CovarianceError:
    """The covariance error of the crossfade over one band.

    Attributes:
        band_hz: The band, in hertz.
        total: ``eps_tot``.
        stationary: The stationary part alone, ``sqrt(int (S_0 - S)^2 / int S^2)``.
        cyclic: The cyclic part alone.
    """

    band_hz: tuple[float, float]
    total: float
    stationary: float
    cyclic: float


@dataclass(frozen=True)
class CrossfadeCovarianceError:
    """The covariance error of the crossfade at one window.

    Attributes:
        window_duration: The window the simulator used, in seconds: its whole number of samples
            over the sampling frequency, which differs from the requested duration when that
            is not a whole number of samples.
        error_band: The error over the analysis band, cut :data:`NYQUIST_GUARD_HZ` below the
            Nyquist frequency. This is the error the bundled table holds.
        analysis_band: The error over the analysis band up to at most the Nyquist frequency,
            which includes the simulator's roll-off there.
        harmonics: How many cyclic harmonics were summed.
        converged: Whether the cyclic sum met the stopping rule, with every pair that
            straddles zero or the Nyquist frequency summed, before the cap. When it did not,
            the errors omit part of the cyclic error and are not upper bounds.
    """

    window_duration: float
    error_band: CovarianceError
    analysis_band: CovarianceError
    harmonics: int
    converged: bool


def sha256_of(path: str | Path) -> str:
    """Return the SHA-256 of a file, the identity under which a PSD's errors are recorded."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _circulant_autocovariance(psd: np.ndarray, delta_frequency: float, window_size: int) -> np.ndarray:
    """Return the autocovariance of one chunk, periodic in the lag with the window.

    Each bin holds a complex Gaussian amplitude; the inverse real transform discards the
    imaginary part of the zero-frequency bin and, for an even window, of the Nyquist bin, so
    those carry half the variance.
    """
    amplitude = psd.copy()
    self_conjugate = [0, psd.size - 1] if window_size % 2 == 0 else [0]
    amplitude[self_conjugate] *= 0.5
    return 0.5 * delta_frequency * window_size * np.fft.irfft(amplitude, n=window_size)


class _Crossfade:
    """The crossfade's closed-form spectra on one dense grid, harmonic by harmonic.

    Window ``n`` covers samples ``[n P, n P + W)``; its first ``W - P`` samples are blended with
    the tail of window ``n - 1``. A sample at phase ``r`` within the stride is
    ``beta[r] old + alpha[r] new``. The covariance of two samples is the chunk autocovariance
    at their lag times a factor of the weights at their phases, and the cyclic spectra are the
    transforms of that product's Fourier coefficients in the phase. The cyclic
    autocorrelation has support ``2 P`` on each side, so the spectra vary on the scale
    ``f_s / (4 P)``; the grid is :data:`OVERSAMPLE` times finer than that. Spectra are
    one-sided, like the target.

    Attributes:
        frequencies: The grid ``0 .. f_s / 2`` in hertz.
        stationary: ``|S_0|`` on it.
        stride: The stride ``P``, the period of the process, in samples.
    """

    def __init__(
        self,
        weight_old: np.ndarray,
        weight_new: np.ndarray,
        window_size: int,
        chunk_autocovariance: np.ndarray,
        sampling_frequency: float,
    ) -> None:
        """Compute the stationary spectrum; cyclic ones are computed on request."""
        self.stride = window_size - weight_old.size
        self._window_size = window_size
        self._alpha = np.ones(self.stride)
        self._alpha[: weight_new.size] = weight_new
        self._beta = np.zeros(self.stride)
        self._beta[: weight_old.size] = weight_old
        self._chunk = chunk_autocovariance
        self._sampling_frequency = sampling_frequency
        self._n_fft = 4 * self.stride * OVERSAMPLE
        self.frequencies = np.arange(self._n_fft // 2 + 1) * sampling_frequency / self._n_fft
        self.stationary = self.upper_edge(0)[: self.frequencies.size]

    def _modulation(self, cycle_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the lags ``-(2P - 1) .. 2P - 1`` and the weights' cyclic coefficient at each.

        For each of the three ways a pair of samples can share a window, the sum over phases is
        a linear correlation of two weight sequences, one of them modulated by the cycle
        exponential.
        """
        stride = self.stride
        phase = np.exp(-2.0j * np.pi * cycle_index * np.arange(stride) / stride)
        alpha, beta = self._alpha, self._beta
        mod_alpha, mod_beta = alpha * phase, beta * phase

        def correlate(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            # sum_r x[r] y[r + d] for d = -(P - 1) .. P - 1
            return fftconvolve(x[::-1], y)

        lags = np.arange(-(2 * stride - 1), 2 * stride)
        total = np.zeros(lags.size, dtype=complex)
        offset = 2 * stride - 1
        d = np.arange(-(stride - 1), stride)
        total[d + offset] += correlate(mod_alpha, alpha) + correlate(mod_beta, beta)
        total[d + stride + offset] += correlate(mod_alpha, beta)
        total[d - stride + offset] += correlate(mod_beta, alpha)
        return lags, total / stride

    def upper_edge(self, cycle_index: int) -> np.ndarray:
        """Return ``|S^{(k)}(f)|`` over the whole period ``0 .. f_s`` of the sampled process."""
        lags, modulation = self._modulation(cycle_index)
        autocorrelation = modulation * self._chunk[np.abs(lags) % self._window_size]
        placed = np.zeros(self._n_fft, dtype=complex)
        np.add.at(placed, lags % self._n_fft, autocorrelation)
        return 2.0 * np.abs(np.fft.fft(placed)) / self._sampling_frequency

    def cycle_steps(self, cycle_index: int) -> int:
        """Return ``alpha_k`` in grid steps."""
        return 4 * cycle_index * OVERSAMPLE

    @property
    def distinct_harmonics(self) -> int:
        """Return ``P // 2``: harmonics ``k`` and ``P - k`` pair the same components."""
        return self.stride // 2

    def new_pair_onset(self, band: tuple[float, float]) -> int:
        """Return the first harmonic whose pairs can straddle zero or ``f_s / 2`` inside the band.

        Pairs straddle zero from ``alpha_k >= 2 min B`` and the Nyquist frequency from
        ``alpha_k >= f_s - 2 max B``; the later of the two onsets is returned.
        """
        step = self._sampling_frequency / self.stride
        onsets = (2.0 * band[0], self._sampling_frequency - 2.0 * band[1])
        return max(int(np.ceil(onset / step - 1e-9)) for onset in onsets)

    def row_density(self, cycle_index: int, full: np.ndarray, selected: np.ndarray) -> np.ndarray:
        """Return what harmonic ``k`` adds to ``e^2(f) S^2(f)`` on the band's rows.

        ``full`` is ``|S^{(k)}|^2`` over the whole period and ``selected`` marks the band's bins.
        Row ``f`` meets the partners ``f - alpha_k`` and ``f + alpha_k``, each at its folded
        frequency, and each counts when that frequency is a band bin. Both are whole numbers of
        grid steps, so membership is decided on bin indices: a frequency compared in floating
        point can fold an ulp past a band edge. At ``alpha_k = f_s / 2`` the two partners are
        one component and count once.
        """
        steps = self.cycle_steps(cycle_index)
        period = full.size
        rows = np.arange(self.frequencies.size)

        def inside(partner: np.ndarray) -> np.ndarray:
            folded = np.abs(np.mod(partner + period // 2, period) - period // 2)
            return selected[folded]

        density = np.where(inside(rows - steps), full[rows], 0.0)
        if 2 * steps != period:
            density += np.where(inside(rows + steps), full[(rows + steps) % period], 0.0)
        return density[selected]


def _converged_errors(
    crossfade: _Crossfade,
    target: np.ndarray,
    bands: Sequence[tuple[float, float]],
) -> tuple[list[CovarianceError], int, bool]:
    """Return the covariance error over each band, the harmonics summed, and whether the sum converged.

    Harmonics are added until doubling their number changes ``eps_tot^2`` by less than
    :data:`CONVERGENCE_TOLERANCE` in every band, or :data:`MAXIMUM_HARMONICS` is reached. The test
    is not taken before twice the first harmonic whose pairs straddle zero or the Nyquist
    frequency in some band: those pairs start a contribution the harmonics below them do not
    announce. When that harmonic lies beyond the cap, the test is taken as usual and the result is
    marked not converged, since the straddling pairs were never summed. No more than
    :attr:`_Crossfade.distinct_harmonics` are summed; they hold every pair.

    Raises:
        ValueError: If a band holds fewer than two bins, or the target is zero throughout it or
            too small, beside the table's peak, for its norm to be evaluated in double precision.
    """
    frequencies = crossfade.frequencies
    selections, denominators, stationary_numerators, cyclic_numerators = [], [], [], []
    for low, high in bands:
        selected = (frequencies >= low) & (frequencies <= high)
        if np.count_nonzero(selected) < MINIMUM_BINS:
            raise ValueError(f"the band {(low, high)} Hz holds fewer than two frequency bins; widen it.")
        reference = target[selected]
        denominator = float(np.trapezoid(reference**2, frequencies[selected]))
        if denominator <= 0.0 and np.any(reference > 0.0):
            raise ValueError(_TOO_WIDE)
        if denominator <= 0.0:
            raise ValueError(
                f"the PSD is zero throughout the band {(low, high)} Hz, so no error relative to it is defined; "
                f"choose a band where the PSD is not zero."
            )
        selections.append(selected)
        denominators.append(denominator)
        stationary_numerators.append(
            float(np.trapezoid((crossfade.stationary[selected] - reference) ** 2, frequencies[selected]))
        )
        cyclic_numerators.append(0.0)
    maximum_harmonics = min(MAXIMUM_HARMONICS, crossfade.distinct_harmonics)
    earliest_stop = max(FIRST_CONVERGENCE_CHECK, *(2 * crossfade.new_pair_onset(band) for band in bands))
    reachable = earliest_stop <= maximum_harmonics or maximum_harmonics == crossfade.distinct_harmonics
    if not reachable:
        earliest_stop = FIRST_CONVERGENCE_CHECK
    history: dict[int, list[float]] = {}
    used, converged = 0, False
    for cycle_index in range(1, maximum_harmonics + 1):
        used = cycle_index
        full = crossfade.upper_edge(cycle_index) ** 2
        for index, selected in enumerate(selections):
            row = crossfade.row_density(cycle_index, full, selected)
            cyclic_numerators[index] += float(np.trapezoid(row, frequencies[selected]))
        total = [(stationary_numerators[i] + cyclic_numerators[i]) / denominators[i] for i in range(len(bands))]
        power_of_two = (cycle_index & (cycle_index - 1)) == 0
        if cycle_index >= earliest_stop and power_of_two:
            earlier = history.get(cycle_index // 2)
            if earlier is not None and all(
                abs(total[i] - earlier[i]) <= CONVERGENCE_TOLERANCE * total[i] for i in range(len(bands))
            ):
                converged = reachable
                break
        if power_of_two:
            history[cycle_index] = total
    else:
        converged = used == crossfade.distinct_harmonics
    errors = [
        CovarianceError(
            band_hz=(float(low), float(high)),
            total=float(np.sqrt((stationary_numerators[i] + cyclic_numerators[i]) / denominators[i])),
            stationary=float(np.sqrt(stationary_numerators[i] / denominators[i])),
            cyclic=float(np.sqrt(cyclic_numerators[i] / denominators[i])),
        )
        for i, (low, high) in enumerate(bands)
    ]
    return errors, used, converged


def _check_sampling_frequency(sampling_frequency: float) -> None:
    """Refuse a sampling frequency that is not positive and finite."""
    if not (np.isfinite(sampling_frequency) and sampling_frequency > 0.0):
        raise ValueError(f"sampling_frequency must be positive and finite, got {sampling_frequency!r}.")


def simulator_settings(sampling_frequency: float, window_duration: float) -> dict[str, float]:
    """Return the :class:`~gwmock_noise.simulators.ColoredNoiseSimulator` settings an error is evaluated for.

    With :func:`crossfade_target` written to a file and passed as ``psd_file``, these
    reproduce the configuration :func:`crossfade_covariance_error` evaluates.

    Raises:
        ValueError: If the sampling frequency is not positive and finite.
    """
    _check_sampling_frequency(sampling_frequency)
    return {
        "sampling_frequency": float(sampling_frequency),
        "low_frequency_cutoff": 0.0,
        "high_frequency_cutoff": sampling_frequency / 2.0,
        "window_duration": float(window_duration),
    }


def crossfade_target(psd_file: str | Path, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the continued target an error is evaluated for, as a table from zero to Nyquist.

    The curve is continued as the bundled errors were, with the settings the bundled table
    records: held flat below 5 Hz at its value there, and between 5 Hz and a first tabulated
    point above it continued by the power law of its lowest octave. A curve tabulated below
    5 Hz is therefore replaced there by that flat value. A curve that stops below the Nyquist
    frequency stops there, and the target is zero beyond it.

    Args:
        psd_file: A two-column PSD table, or the name of a bundled curve.
        sampling_frequency: The sampling frequency, in hertz.

    Returns:
        The table's frequencies and one-sided PSD.

    Raises:
        ValueError: If the sampling frequency is not positive and finite, if the table cannot be
            continued (see :func:`~gwmock_noise.simulators._covariance_tolerance.validate_table`),
            or if the continued power law overflows.
    """
    _check_sampling_frequency(sampling_frequency)
    return _continued_table(*load_spectral_series(psd_file, kind="PSD"), sampling_frequency)


def _continued_table(
    frequencies: np.ndarray, psd: np.ndarray, sampling_frequency: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return :func:`crossfade_target` for a table already loaded."""
    settings = load_crossfade_errors()["target"]
    with np.errstate(over="ignore"):  # An overflow is refused below, with a reason.
        target_frequencies, target_psd = continued_target(
            frequencies,
            psd,
            sampling_frequency,
            continuation_hz=settings["continuation_hz"],
            slope_octave=settings["slope_octave"],
            continuation_points=settings["continuation_points"],
        )
    if not np.all(np.isfinite(target_psd)):
        raise ValueError(
            "the power law that continues the PSD table below its first point overflows; the table's lowest octave "
            "is too steep to continue."
        )
    return target_frequencies, target_psd


#: Why a PSD whose band and peak are too far apart is refused.
_TOO_WIDE = (
    "the covariance error cannot be evaluated in double precision: the PSD spans too many orders of magnitude between "
    "the band and the rest of the table (more than about 160), so the error's square, or the band's norm, overflows or "
    "underflows, although the error itself may be finite. Trim the table to the band's neighbourhood."
)


def _scale_exponent(frequencies: np.ndarray, psd: np.ndarray, band: tuple[float, float]) -> int:
    """Return the power of two to scale ``psd`` by, centring the band's largest value and the table's on one.

    The squares of the band's values form the error's denominator and must not underflow; the
    squares of the table's largest values enter the covariances and must not overflow. Centring
    the two, rather than bringing either to one, represents both well beyond the point where the
    error's square itself overflows, about 160 orders of magnitude between them. The band's values are the tabulated ones
    inside it and the interpolated ones at its edges. It is returned as an exponent, for
    :func:`numpy.ldexp`, because the power of two itself need not be representable when the
    PSD is subnormal.
    """
    largest = float(np.max(psd))
    if largest == 0.0:
        return 0
    in_band = np.concatenate(
        (psd[(frequencies >= band[0]) & (frequencies <= band[1])], np.interp(band, frequencies, psd))
    )
    band_largest = float(np.max(in_band))
    if band_largest == 0.0:
        band_largest = largest
    return -((int(np.frexp(largest)[1]) + int(np.frexp(band_largest)[1])) // 2)


def crossfade_covariance_error(
    psd_file: str | Path,
    sampling_frequency: float,
    window_duration: float,
    analysis_band: tuple[float, float],
) -> CrossfadeCovarianceError:
    """Return the covariance error of the colored-noise simulator's crossfade at one window.

    The simulator is configured as ``covariance_tolerance`` configures it: the continued
    target of :func:`crossfade_target`, with the settings of :func:`simulator_settings` and a
    half-window overlap. The error describes that configuration; pass both to
    :class:`~gwmock_noise.simulators.ColoredNoiseSimulator` to generate the noise it describes.
    It is computed over two bands: the analysis band cut :data:`NYQUIST_GUARD_HZ` below the
    Nyquist frequency, the error the bundled table holds, and the analysis band up to at most
    the Nyquist frequency. The cyclic sum must converge in both.

    The cost grows with the window: at 4096 Hz an 8 s window takes seconds, a 64 s window
    minutes and a 256 s window hours. Windows of 1024 s and more need several gigabytes of
    memory, and can be reported unconverged after 16 or 32 harmonics, when their straddling
    pairs lie beyond the harmonic cap: for the bundled curves, every window of 2048 s and
    more, and 1024 s on the aLIGO curves.

    Args:
        psd_file: A two-column PSD table covering the analysis band, or the name of a bundled
            curve.
        sampling_frequency: The sampling frequency, in hertz.
        window_duration: The crossfade window, in seconds; it must span an even number of samples.
        analysis_band: The analysis band, in hertz.

    Returns:
        The error over both bands, the harmonics summed, and whether the sum converged.

    Raises:
        ValueError: If the band is not inside ``[1 Hz, f_s / 2]``, if it extends beyond the PSD
            table's last point, if it holds fewer than two frequency bins, or the PSD is zero
            throughout it; if the table cannot be continued (see :func:`crossfade_target`); if
            the window spans an odd number of samples, so that its overlap differs from its
            stride; or if the PSD spans so many orders of magnitude between the band and its peak
            that the error cannot be evaluated in double precision.
    """
    from gwmock_noise.simulators.colored import (  # noqa: PLC0415 - avoids an import cycle.
        PSD_WINDOW_WIDTH_HZ,
        ColoredNoiseSimulator,
    )

    _check_sampling_frequency(sampling_frequency)
    nyquist = sampling_frequency / 2.0
    low, high = (float(edge) for edge in analysis_band)
    if not 0.0 < low < high <= nyquist:
        raise ValueError(f"analysis_band must satisfy 0 < low < high <= {nyquist} Hz, got {analysis_band}.")
    if low < PSD_WINDOW_WIDTH_HZ:
        raise ValueError(
            f"analysis_band starts at {low!r} Hz, inside the simulator's {PSD_WINDOW_WIDTH_HZ} Hz taper above 0 Hz; the "
            f"error there would measure the taper. Start the band at {PSD_WINDOW_WIDTH_HZ} Hz or above."
        )
    guarded_band = error_band((low, high), sampling_frequency)
    if guarded_band[0] >= guarded_band[1]:
        raise ValueError(f"analysis_band {analysis_band} lies within {NYQUIST_GUARD_HZ} Hz of the Nyquist frequency.")
    table_frequencies, table_psd = load_spectral_series(psd_file, kind="PSD")
    target_frequencies, target_psd = _continued_table(table_frequencies, table_psd, sampling_frequency)
    # The error is a ratio of integrals quadratic in the PSD, so it does not depend on the PSD's scale; evaluating
    # it with the band's values and the table's peak centred on one keeps their squares from underflowing or
    # overflowing. A power of two scales exactly, so the error is the number it would be unscaled whenever that
    # number is representable.
    exponent = _scale_exponent(target_frequencies, target_psd, (low, min(high, nyquist)))
    with np.errstate(over="ignore"):  # A table too wide to represent is refused below, once its error is not finite.
        target_psd = np.ldexp(target_psd, exponent)
    table_end = float(table_frequencies[-1])
    if min(high, nyquist) > table_end:
        raise ValueError(
            f"analysis_band {analysis_band} extends beyond the PSD table, which stops at {table_end!r} Hz; the target "
            f"is zero there. End the band at {table_end!r} Hz or below, or extend the table."
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "target.txt"
        np.savetxt(path, np.column_stack([target_frequencies, target_psd]), fmt="%.17e")
        simulator = ColoredNoiseSimulator(
            psd_file=path,
            detectors=["H1"],
            duration=1.0,
            seed=1,
            **simulator_settings(sampling_frequency, window_duration),
        )
    window_size = simulator._window_size
    stitcher = simulator._stitcher
    if 2 * stitcher.overlap_size != window_size:
        raise ValueError(
            f"window_duration {window_duration!r} s spans {window_size} samples at {sampling_frequency!r} Hz; the "
            f"stitched process this evaluates needs an even number, so that the overlap equals the stride."
        )
    chunk = _circulant_autocovariance(
        np.asarray(simulator._psd, dtype=float),
        simulator._delta_frequency,
        window_size,
    )
    crossfade = _Crossfade(
        stitcher._window_out / stitcher._blend_norm,
        stitcher._window_in / stitcher._blend_norm,
        window_size,
        chunk,
        sampling_frequency,
    )
    settings = load_crossfade_errors()["target"]
    with np.errstate(over="ignore"):
        target = np.ldexp(
            continued_psd(
                table_frequencies,
                table_psd,
                crossfade.frequencies,
                continuation_hz=settings["continuation_hz"],
                slope_octave=settings["slope_octave"],
            ),
            exponent,
        )
    with np.errstate(over="ignore", invalid="ignore"):  # A result that is not a number is refused below, with a reason.
        (guarded, unguarded), harmonics, converged = _converged_errors(
            crossfade, target, (guarded_band, (low, min(high, nyquist)))
        )
    if not all(np.isfinite([band.total, band.stationary, band.cyclic]).all() for band in (guarded, unguarded)):
        raise ValueError(_TOO_WIDE)
    return CrossfadeCovarianceError(
        window_duration=window_size / sampling_frequency,
        error_band=guarded,
        analysis_band=unguarded,
        harmonics=harmonics,
        converged=converged,
    )
