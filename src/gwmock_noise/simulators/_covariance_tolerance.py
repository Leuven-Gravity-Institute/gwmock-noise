"""Choose the crossfade window from a requested covariance-error tolerance.

The covariance error of the crossfade is the relative distance between the second moments
of the generated noise and those of the target spectrum, over an analysis band: its
stationary part, the smoothing of the spectrum by the window, and its cyclic part, the
correlation between frequencies that the periodic crossfade introduces. It falls as the
window grows. Evaluating it takes minutes to hours, so the simulator does not evaluate it:
it reads the errors evaluated for the paper

    I. C. F. Wong, "Choosing a generator for streaming Gaussian noise in
    gravitational-wave mock data: accuracy, cost and reproducibility",
    arXiv:2611.01624; data: doi:10.5281/zenodo.23243286,

bundled in :mod:`gwmock_noise.data.covariance_tolerance`, and takes the shortest evaluated
window that meets the tolerance. It does not interpolate between evaluated windows: the error
between two of them is not known to lie between their values.

Those errors hold for one configuration only, and the simulator reproduces it:

- a bundled detector curve, unmodified (checked by its SHA-256);
- a sampling frequency the paper evaluated;
- the paper's continued target, run from zero to the Nyquist frequency: held flat below 5 Hz
  at its value there, and between 5 Hz and the curve's first tabulated point, when that point
  is above 5 Hz (the aLIGO curves), continued by the power law of its lowest octave. A curve
  tabulated below 5 Hz (the ET curves, from 1 Hz) is therefore *replaced* below 5 Hz by that
  flat value, which lies orders of magnitude below its tabulated low-frequency wall;
- the error measured over the curve's analysis band, which the table records.

Anything else is refused with :class:`UnsupportedCovarianceToleranceError`, as is a tolerance
that only an evaluation which did not converge would meet.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise.data.covariance_tolerance import load_crossfade_errors
from gwmock_noise.simulators._spectral import _resolve_bundled_psd, load_spectral_series

#: Where the documentation explains the evaluated configuration and how to add a curve.
DOCUMENTATION_SECTION = "the 'Choosing the window from a covariance tolerance' section of the noise simulation guide"

#: The error band stops this far below the Nyquist frequency, clear of the generator's roll-off there.
NYQUIST_GUARD_HZ = 4.0


def error_band(analysis_band: tuple[float, float], sampling_frequency: float) -> tuple[float, float]:
    """Return the band an error is measured over: the analysis band, cut :data:`NYQUIST_GUARD_HZ` below Nyquist."""
    return (float(analysis_band[0]), min(float(analysis_band[1]), sampling_frequency / 2.0 - NYQUIST_GUARD_HZ))


class UnsupportedCovarianceToleranceError(ValueError):
    """A covariance tolerance was requested for a configuration the bundled errors do not cover."""


@dataclass(frozen=True)
class CrossfadeChoice:
    """The crossfade window chosen for a covariance tolerance.

    Attributes:
        preset: The bundled detector curve.
        window_duration: The chosen window, in seconds.
        covariance_error: The evaluated total covariance error at that window.
        error_band_hz: The band the error was measured over, in hertz.
        target_frequencies: The continued target's frequencies, in hertz.
        target_psd: The continued target, the one-sided PSD the simulator is given.
        first_tabulated_hz: The curve's first tabulated frequency.
        continuation_hz: Below this frequency the target is held flat; it replaces the curve
            there when the curve is tabulated below it.
        slope_octave: The factor of frequency above the first point the continuing power law
            is fitted over, when the first point is above ``continuation_hz``.
        source: The DOI of the data release the error was copied from.
    """

    preset: str
    window_duration: float
    covariance_error: float
    error_band_hz: tuple[float, float]
    target_frequencies: np.ndarray
    target_psd: np.ndarray
    first_tabulated_hz: float
    continuation_hz: float
    slope_octave: float
    source: str


#: The fewest points a PSD table may have.
MINIMUM_TABLE_POINTS = 2


def fitted_octave(table_frequencies: np.ndarray, slope_octave: float) -> np.ndarray:
    """Return which tabulated points the continuing power law is fitted to: those up to ``slope_octave`` times the first."""
    return table_frequencies <= slope_octave * float(table_frequencies[0])


def validate_table(
    table_frequencies: np.ndarray, table_psd: np.ndarray, *, continuation_hz: float, slope_octave: float
) -> None:
    """Check that a PSD table can be continued.

    The table needs two or more points, finite, non-negative and strictly ascending in
    frequency, with a finite, non-negative PSD. A zero is a PSD of zero there. When the table
    starts above ``continuation_hz`` its lowest ``slope_octave`` factor of frequency is fitted
    with a power law in log-log, so it needs two or more points there, all positive; one point
    would leave the slope undetermined, and a zero would make it not a number.

    Raises:
        ValueError: If any of these does not hold.
    """
    if table_frequencies.size < MINIMUM_TABLE_POINTS or table_frequencies.shape != table_psd.shape:
        raise ValueError("the PSD table needs at least two matching frequency and PSD points.")
    if not np.all(np.isfinite(table_frequencies) & (table_frequencies >= 0.0)):
        raise ValueError("the PSD table's frequencies must be finite and non-negative.")
    if np.any(np.diff(table_frequencies) <= 0.0):
        raise ValueError("the PSD table's frequencies must ascend strictly.")
    if not np.all(np.isfinite(table_psd) & (table_psd >= 0.0)):
        raise ValueError("the PSD table must be finite and non-negative at every point.")
    first = float(table_frequencies[0])
    if first > continuation_hz:
        octave = fitted_octave(table_frequencies, slope_octave)
        if np.count_nonzero(octave) < MINIMUM_TABLE_POINTS or np.any(table_psd[octave] <= 0.0):
            raise ValueError(
                f"the PSD table is continued below its first point, {first!r} Hz, by the power law fitted to its "
                f"points up to {slope_octave * first!r} Hz, so it needs two or more points there, all positive."
            )


def continued_psd(
    table_frequencies: np.ndarray,
    table_psd: np.ndarray,
    frequencies: np.ndarray,
    *,
    continuation_hz: float,
    slope_octave: float,
) -> np.ndarray:
    """Return the paper's continued target at the given frequencies.

    Above the curve's first tabulated point the target is the curve, interpolated linearly
    in frequency and zero beyond its last point. Between ``continuation_hz`` and that point
    it continues the curve with the power law fitted, in log-log, to the curve's lowest
    ``slope_octave`` factor of frequency. Below ``continuation_hz`` it is held flat at its
    value there.

    Args:
        table_frequencies: The curve's frequencies, ascending, in hertz.
        table_psd: The curve's one-sided PSD, positive.
        frequencies: Where to evaluate the target, in hertz.
        continuation_hz: Where the continuation is held flat, in hertz.
        slope_octave: The factor of frequency above the first point the slope is fitted over.

    Returns:
        The one-sided PSD at ``frequencies``.

    Raises:
        ValueError: If the table fails :func:`validate_table`.
    """
    validate_table(table_frequencies, table_psd, continuation_hz=continuation_hz, slope_octave=slope_octave)
    frequencies = np.asarray(frequencies, dtype=float)
    first = float(table_frequencies[0])
    target = np.interp(frequencies, table_frequencies, table_psd, left=0.0, right=0.0)
    if first > continuation_hz:
        octave = fitted_octave(table_frequencies, slope_octave)
        slope = float(np.polyfit(np.log(table_frequencies[octave]), np.log(table_psd[octave]), 1)[0])

        def level(at: np.ndarray) -> np.ndarray:
            return np.where(
                at < first,
                float(table_psd[0]) * (np.maximum(at, 1e-300) / first) ** slope,
                np.interp(at, table_frequencies, table_psd),
            )

    else:

        def level(at: np.ndarray) -> np.ndarray:
            return np.interp(at, table_frequencies, table_psd)

    below_floor = frequencies < continuation_hz
    continued = (frequencies < first) & ~below_floor
    target[continued] = level(frequencies[continued])
    target[below_floor] = float(level(np.array([continuation_hz]))[0])
    return target


def continued_target(  # noqa: PLR0913 - the table, the rate and the three continuation settings are distinct.
    table_frequencies: np.ndarray,
    table_psd: np.ndarray,
    sampling_frequency: float,
    *,
    continuation_hz: float,
    slope_octave: float,
    continuation_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the paper's continued target as a table from zero to the Nyquist frequency.

    The target is :func:`continued_psd`. Its continuation is tabulated at
    ``continuation_points`` geometrically spaced frequencies from ``continuation_hz`` to the
    curve's first point, so that the simulator's linear interpolation follows the power law;
    the table also holds zero, every tabulated point up to the Nyquist frequency, and the
    Nyquist frequency when the curve reaches it. A curve that stops below the Nyquist
    frequency stops there, so that the simulator's interpolation, like the target, is zero
    beyond it rather than ramping down to zero.

    Args:
        table_frequencies: The curve's frequencies, ascending, in hertz.
        table_psd: The curve's one-sided PSD, positive.
        sampling_frequency: The sampling frequency, in hertz.
        continuation_hz: Where the continuation is held flat, in hertz.
        slope_octave: The factor of frequency above the first point the slope is fitted over.
        continuation_points: How many points tabulate the continuation.

    Returns:
        The table's frequencies and one-sided PSD.

    Raises:
        ValueError: If the table fails :func:`validate_table`.
    """
    validate_table(table_frequencies, table_psd, continuation_hz=continuation_hz, slope_octave=slope_octave)
    nyquist = sampling_frequency / 2.0
    first = float(table_frequencies[0])
    # A table from 0 Hz has no geometric continuation; it is flat below continuation_hz all the same.
    continuation = np.geomspace(continuation_hz, first, continuation_points) if first > 0.0 else [continuation_hz]
    end = [nyquist] if table_frequencies[-1] >= nyquist else []
    grid = np.unique(np.concatenate(([0.0], continuation, table_frequencies[table_frequencies <= nyquist], end)))
    # A first point above the Nyquist frequency would carry the continuation past it; the target stops there.
    grid = grid[grid <= nyquist]
    return grid, continued_psd(
        table_frequencies, table_psd, grid, continuation_hz=continuation_hz, slope_octave=slope_octave
    )


def _bundled_preset(psd_file: str | Path, spectra: dict[str, Any]) -> tuple[str, Path]:
    """Return the bundled preset ``psd_file`` names and its path, verified unmodified.

    Raises:
        UnsupportedCovarianceToleranceError: If ``psd_file`` is not a bundled curve the table
            covers, or if the file differs from the one the errors were evaluated for.
    """
    given = Path(str(psd_file))
    preset = given.stem
    resolved = _resolve_bundled_psd(preset) if preset in spectra else None
    # A bare file name selects the bundled curve, as the PSD loader does; one with a directory part is a path, and
    # must be the bundled file itself, so that a mistyped path is reported instead of falling back to the curve.
    has_directory = given.name != str(psd_file)
    if resolved is None or (
        (has_directory or given.exists()) and (not given.exists() or given.resolve() != resolved.resolve())
    ):
        raise UnsupportedCovarianceToleranceError(
            f"No covariance errors are bundled for the PSD {str(psd_file)!r}. A covariance tolerance is supported "
            f"only for the bundled curves {sorted(spectra)}, passed by name or as the bundled file's own path. For "
            f"another PSD, evaluate the crossfade's covariance error with `gwmock-noise evaluate-covariance-error`, "
            f"which writes the target it evaluated and the ColoredNoiseSimulator settings that reproduce it; to add the "
            f"PSD to the bundled errors, open a pull request with that result. See {DOCUMENTATION_SECTION}."
        )
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != spectra[preset]["sha256"]:
        raise UnsupportedCovarianceToleranceError(
            f"The bundled PSD {preset!r} differs from the one its covariance errors were evaluated for "
            f"(SHA-256 {digest}, expected {spectra[preset]['sha256']}), so the bundled errors do not describe it."
        )
    return preset, resolved


def choose_crossfade_window(
    psd_file: str | Path,
    sampling_frequency: float,
    covariance_tolerance: float,
) -> CrossfadeChoice:
    """Return the shortest evaluated window whose covariance error meets the tolerance.

    Args:
        psd_file: A bundled detector curve, by name or as the path of the bundled file itself.
        sampling_frequency: The sampling frequency, in hertz.
        covariance_tolerance: The largest acceptable total covariance error, positive.

    Returns:
        The chosen window, its evaluated error, the band that error was measured over, and
        the continued target the simulator must be given for that error to hold.

    Raises:
        TypeError: If the tolerance is not a real number.
        ValueError: If the tolerance is not positive and finite.
        UnsupportedCovarianceToleranceError: If the curve or the sampling frequency was not
            evaluated, if no evaluated window meets the tolerance, or if the shortest one
            that does was not converged.
    """
    if not isinstance(covariance_tolerance, numbers.Real) or isinstance(covariance_tolerance, bool | np.bool_):
        raise TypeError(f"covariance_tolerance must be a number, got {covariance_tolerance!r}.")
    covariance_tolerance = float(covariance_tolerance)
    if not math.isfinite(covariance_tolerance):
        raise ValueError(f"covariance_tolerance must be finite, got {covariance_tolerance!r}.")
    if covariance_tolerance <= 0:
        raise ValueError(f"covariance_tolerance must be positive, got {covariance_tolerance!r}.")
    table = load_crossfade_errors()
    preset, path = _bundled_preset(psd_file, table["spectra"])
    rates = table["spectra"][preset]["sampling_frequencies"]
    rate = rates.get(str(round(sampling_frequency))) if float(sampling_frequency).is_integer() else None
    if rate is None:
        raise UnsupportedCovarianceToleranceError(
            f"No covariance errors are bundled for {preset!r} at {sampling_frequency!r} Hz; they cover "
            f"{sorted(float(key) for key in rates)} Hz. Use one of those rates, or evaluate the error at yours with "
            f"`gwmock-noise evaluate-covariance-error`, which writes the target it evaluated and the "
            f"ColoredNoiseSimulator settings that reproduce it."
        )
    for row in rate["windows"]:
        if row["eps_total"] > covariance_tolerance:
            continue
        if not row["converged"]:
            raise UnsupportedCovarianceToleranceError(
                f"The tolerance {covariance_tolerance!r} on {preset!r} at {sampling_frequency!r} Hz is first met at a "
                f"window of {row['window_duration']!r} s, where the evaluation of the cyclic error did not converge: "
                f"its value {row['eps_total']!r} omits part of that error and is not an upper bound. Choose a larger "
                f"tolerance, or window_duration directly."
            )
        frequencies, psd = load_spectral_series(path, kind="PSD")
        target = table["target"]
        target_frequencies, target_psd = continued_target(
            frequencies,
            psd,
            sampling_frequency,
            continuation_hz=target["continuation_hz"],
            slope_octave=target["slope_octave"],
            continuation_points=target["continuation_points"],
        )
        return CrossfadeChoice(
            preset=preset,
            window_duration=float(row["window_duration"]),
            covariance_error=float(row["eps_total"]),
            error_band_hz=(float(rate["error_band_hz"][0]), float(rate["error_band_hz"][1])),
            target_frequencies=target_frequencies,
            target_psd=target_psd,
            first_tabulated_hz=float(frequencies[0]),
            continuation_hz=float(target["continuation_hz"]),
            slope_octave=float(target["slope_octave"]),
            source=f"doi:{table['source']['doi']}",
        )
    converged = [row["eps_total"] for row in rate["windows"] if row["converged"]]
    smallest = (
        f"the smallest converged evaluated error is {min(converged)!r}"
        if converged
        else "no evaluation there converged"
    )
    raise UnsupportedCovarianceToleranceError(
        f"No evaluated window meets the tolerance {covariance_tolerance!r} on {preset!r} at {sampling_frequency!r} "
        f"Hz; {smallest}. Choose a larger tolerance."
    )
