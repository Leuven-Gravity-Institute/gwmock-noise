"""Shared numerical helpers for fitting AR and ARMA noise models.

The functions here implement the pieces the M4b simulators share: the
Levinson-Durbin recursion that turns a target autocovariance into a stable
all-pole filter, the conditioning diagnostics that say whether that fit can be
trusted, and the per-band relative PSD residual both simulators record.

Stability policy. The recursion produces a bounded-error stable filter *by
construction*: for a positive-definite autocovariance every reflection
coefficient has magnitude below one, which places every root of the denominator
strictly inside the unit circle. When floating point breaks that property --
a non-positive-definite input, or a prediction error that collapses to zero --
the recursion cannot continue and raises :class:`FitError` rather than clamping,
regularising, or reflecting a pole. A fit that is not trustworthy is reported
as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

__all__ = [
    "FitError",
    "LevinsonFit",
    "band_fit_residual",
    "geometric_band_edges",
    "levinson_durbin",
    "require_nonnegative_spectrum",
    "toeplitz_condition_number",
]

#: Pre-registered conditioning limits. ``max_reflection_limit`` is the strict
#: upper bound the recursion enforces at every order; a reflection coefficient
#: at or above it means the denominator would have a pole on or outside the
#: unit circle. ``min_prediction_error`` is the corresponding lower bound on the
#: remaining prediction error -- a non-positive value means the recursion has
#: lost all significance. Both are checked while the recursion runs, so a fit
#: either satisfies them or raises.
MAX_REFLECTION_COEFFICIENT = 1.0
MIN_PREDICTION_ERROR = 0.0

#: Number of geometric bands used for the recorded per-band PSD residual.
DEFAULT_FIT_BANDS = 8


class FitError(ValueError):
    """Raised when a model fit is numerically invalid or unstable."""


@dataclass(frozen=True)
class LevinsonFit:
    """Result of a Levinson-Durbin recursion.

    Attributes:
        coefficients: The ``a_1 .. a_p`` prediction coefficients; the denominator
            polynomial is ``1 + sum a_i z**-i``.
        prediction_error: The final residual variance ``E_p``.
        reflection_coefficients: The ``k_1 .. k_p`` reflection coefficients.
        prediction_errors: The ``E_0 .. E_p`` prediction errors, one per order.
    """

    coefficients: np.ndarray
    prediction_error: float
    reflection_coefficients: np.ndarray
    prediction_errors: np.ndarray


def levinson_durbin(autocovariance: np.ndarray, order: int) -> LevinsonFit:
    """Fit a stable all-pole model with the Levinson-Durbin recursion.

    The recursion is the in-place Durbin update, so each reflection coefficient
    is obtained from the prediction error of the previous order and the whole
    fit costs ``O(order**2)``.

    Args:
        autocovariance: Autocovariance lags ``r[0 .. order]``.
        order: Model order ``p`` (at least zero).

    Returns:
        The fitted coefficients, final prediction error, reflection
        coefficients and prediction-error sequence.

    Raises:
        FitError: If ``order`` is negative, the lag sequence is too short or not
            finite, ``r[0]`` is not positive, a reflection coefficient reaches
            the unit circle, or the prediction error loses its sign.
    """
    autocovariance = np.asarray(autocovariance, dtype=float)
    if order < 0:
        raise FitError("order must be non-negative.")
    if autocovariance.ndim != 1 or autocovariance.size < order + 1:
        raise FitError("autocovariance must provide r[0] through r[order].")
    if not np.all(np.isfinite(autocovariance)):
        raise FitError("autocovariance must be finite.")
    if autocovariance[0] <= 0.0:
        raise FitError("autocovariance r[0] must be positive.")

    coefficients = np.zeros(order + 1, dtype=float)
    coefficients[0] = 1.0
    prediction_error = float(autocovariance[0])
    reflection_coefficients = np.zeros(order, dtype=float)
    prediction_errors = np.zeros(order + 1, dtype=float)
    prediction_errors[0] = prediction_error

    for index in range(1, order + 1):
        residual = autocovariance[index] + float(np.dot(coefficients[1:index], autocovariance[index - 1 : 0 : -1]))
        reflection = -residual / prediction_error
        if not np.isfinite(reflection) or abs(reflection) >= MAX_REFLECTION_COEFFICIENT:
            raise FitError(
                "Levinson-Durbin reflection coefficient reached the unit circle; "
                "the autocovariance is not positive definite at the requested order."
            )
        reflection_coefficients[index - 1] = reflection
        coefficients[1:index] = coefficients[1:index] + reflection * coefficients[index - 1 : 0 : -1]
        coefficients[index] = reflection
        prediction_error *= 1.0 - reflection**2
        if not np.isfinite(prediction_error) or prediction_error <= MIN_PREDICTION_ERROR:
            raise FitError("Levinson-Durbin prediction error lost its sign; the fit is not usable.")
        prediction_errors[index] = prediction_error

    return LevinsonFit(
        coefficients=np.asarray(coefficients[1:], dtype=float),
        prediction_error=prediction_error,
        reflection_coefficients=reflection_coefficients,
        prediction_errors=prediction_errors,
    )


def require_nonnegative_spectrum(values: np.ndarray, *, label: str) -> None:
    """Reject a spectral density that is not a valid, finite, non-negative curve.

    A negative or non-finite sample cannot be turned into a bandwidth-limited
    process by clamping it: the clamp would return a filter for a *different*
    target than the caller asked for, which is exactly the silent degradation
    the fit contract forbids.

    Args:
        values: Interpolated spectral samples inside the fitted band.
        label: Human-readable name of the spectrum, used in the message.

    Raises:
        FitError: If any sample is negative or not finite.
    """
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise FitError(f"The interpolated {label} is not finite inside the fitted band.")
    if np.any(values < 0.0):
        raise FitError(f"The {label} must be non-negative inside the fitted band; a negative value is not valid.")


def toeplitz_condition_number(autocovariance: np.ndarray, order: int) -> float:
    """Return the condition number of the order-``order`` Toeplitz system.

    This is the second pre-registered conditioning number: it measures how much
    the autocovariance matrix amplifies floating-point error, independently of
    the recursion's own stability margin.

    Args:
        autocovariance: Autocovariance lags ``r[0 .. order - 1]``.
        order: System size.

    Returns:
        The 2-norm condition number, or ``inf`` if the matrix is singular.
    """
    autocovariance = np.asarray(autocovariance, dtype=float)
    if order <= 0:
        return 1.0
    offsets = np.abs(np.subtract.outer(np.arange(order), np.arange(order)))
    toeplitz = autocovariance[offsets]
    try:
        return float(np.linalg.cond(toeplitz))
    except np.linalg.LinAlgError:  # pragma: no cover - cond does not raise on finite input
        return float("inf")


def geometric_band_edges(low_frequency: float, high_frequency: float, n_bands: int = DEFAULT_FIT_BANDS) -> np.ndarray:
    """Return ``n_bands + 1`` geometrically spaced band edges.

    Args:
        low_frequency: Lower edge of the fit band in hertz.
        high_frequency: Upper edge of the fit band in hertz.
        n_bands: Number of bands.

    Returns:
        The band edges from ``low_frequency`` to ``high_frequency``.
    """
    if n_bands < 1:
        raise ValueError("n_bands must be at least one.")
    if low_frequency <= 0.0 or high_frequency <= low_frequency:
        return np.linspace(low_frequency, high_frequency, n_bands + 1)
    return np.geomspace(low_frequency, high_frequency, n_bands + 1)


def band_fit_residual(  # noqa: PLR0913
    frequencies: np.ndarray,
    target: np.ndarray,
    model: np.ndarray,
    *,
    low_frequency: float,
    high_frequency: float,
    n_bands: int = DEFAULT_FIT_BANDS,
) -> dict[str, object]:
    """Summarise the model PSD error band by band.

    The primary number per band is the band-integrated relative error
    ``|sum(model) - sum(target)| / sum(target)``, which is the quantity a
    simulator has to get right; the largest per-bin relative error is recorded
    alongside it for diagnosis.

    Args:
        frequencies: Frequencies of the target and model samples.
        target: Target PSD values.
        model: Model PSD values on the same frequencies.
        low_frequency: Lower edge of the fit band in hertz.
        high_frequency: Upper edge of the fit band in hertz.
        n_bands: Number of geometric bands.

    Returns:
        A JSON-serialisable mapping with the per-band list and the summary
        statistics.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    target = np.asarray(target, dtype=float)
    model = np.asarray(model, dtype=float)
    edges = geometric_band_edges(low_frequency, high_frequency, n_bands)

    bands: list[dict[str, float]] = []
    for band_low, band_high in pairwise(edges):
        selected = (frequencies >= band_low) & (frequencies < band_high)
        if not np.any(selected):
            continue
        band_target = target[selected]
        band_model = model[selected]
        reference = float(np.sum(band_target))
        if reference <= 0.0:
            continue
        relative_error = abs(float(np.sum(band_model)) - reference) / reference
        positive = band_target > 0.0
        max_relative_error = (
            float(np.max(np.abs(band_model[positive] - band_target[positive]) / band_target[positive]))
            if np.any(positive)
            else float("nan")
        )
        bands.append(
            {
                "low_frequency": float(band_low),
                "high_frequency": float(band_high),
                "relative_error": relative_error,
                "max_bin_relative_error": max_relative_error,
            }
        )

    relative_errors = [band["relative_error"] for band in bands]
    return {
        "band_count": len(bands),
        "worst_relative_error": max(relative_errors, default=0.0),
        "median_relative_error": float(np.median(relative_errors)) if relative_errors else 0.0,
        "bands": bands,
    }
