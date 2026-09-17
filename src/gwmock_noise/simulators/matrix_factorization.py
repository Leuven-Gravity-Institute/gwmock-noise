"""Canonical matrix spectral factorization of a tabulated cross-spectral matrix.

A multichannel stationary process is fixed by its cross-spectral density matrix
``S(f)``, a Hermitian positive-definite matrix at every frequency. To generate it
from white innovations one needs a *causal minimum-phase matrix filter* ``H`` with
``H(f) H(f)**H = S(f)``: causal so it can run online, minimum-phase so its inverse
is causal too and a bounded-state realization keeps the target.

:func:`whittle_levinson_factorization` returns that filter in its autoregressive
form. Starting from the matrix autocovariance sequence ``R_0 .. R_p`` implied by
the target, the block-Toeplitz analogue of the Levinson-Durbin recursion (Whittle
1963, *On the fitting of multivariate autoregressions and the approximate
canonical factorization of a spectral density matrix*, Biometrika 50, 129-134)
produces coefficient matrices ``A_1 .. A_p`` and an innovation covariance ``V``
such that

    A(z) = I + A_1 z**-1 + ... + A_p z**-p,
    S(f) = A(f)**-1 V A(f)**-H,       H(f) = A(f)**-1 V**(1/2).

``H`` is causal and minimum-phase by construction: the recursion enforces that
the prediction-error covariance stays positive definite, which places every zero
of ``det A(z)`` strictly inside the unit circle, so ``H`` and ``H**-1`` both have
causal expansions. Stability is therefore a property of the recursion, not
something repaired afterwards -- an autocovariance that is not positive definite
raises :class:`FitError` instead of being clamped.

This is the multivariate form of the same recursion the AR simulator uses (see
:mod:`gwmock_noise.simulators.autoregressive` and
:mod:`gwmock_noise.simulators._fit`), specialized back to a scalar when one
channel is configured. Wilson's (1972) matrix-logarithm algorithm computes the
same canonical factor; the block recursion is used here because it is exact at
finite order and yields the bounded state a streaming simulator carries directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from gwmock_noise.simulators._fit import DEFAULT_FIT_BANDS, FitError, geometric_band_edges

__all__ = [
    "WhittleFactorization",
    "matrix_autocovariance",
    "matrix_band_fit_residual",
    "whittle_levinson_factorization",
]

#: Relative tolerance below which a prediction-error eigenvalue counts as lost.
POSITIVE_DEFINITE_TOLERANCE = 1e-12

#: An autocovariance stack carries one lag axis and two channel axes.
AUTOCOVARIANCE_DIMENSIONS = 3


@dataclass(frozen=True)
class WhittleFactorization:
    """A causal minimum-phase matrix filter and its conditioning diagnostics.

    Attributes:
        ar_coefficients: The ``(order, n_channels, n_channels)`` coefficient
            matrices ``A_1 .. A_p`` of ``A(z) = I + sum A_k z**-k``.
        innovation_covariance: The final prediction-error covariance ``V``, an
            ``(n_channels, n_channels)`` positive-definite matrix.
        innovation_factor: A matrix ``V**(1/2)`` with ``factor factor**T = V``.
        reflection_coefficients: The ``(order, n, n)`` coefficient matrices the
            recursion appends at each order; their singular values are the
            multivariate reflection coefficients.
        prediction_errors: The ``(order + 1, n, n)`` prediction-error covariance
            at each order, starting from ``R_0``.
        max_reflection_singular_value: Largest singular value over every
            reflection matrix; below one is the stability margin.
        min_prediction_error_eigenvalue: Smallest eigenvalue seen by the
            recursion; non-positive would mean the fit is not usable.
        innovation_condition_number: Condition number of ``V``.
    """

    ar_coefficients: np.ndarray
    innovation_covariance: np.ndarray
    innovation_factor: np.ndarray
    reflection_coefficients: np.ndarray
    prediction_errors: np.ndarray
    max_reflection_singular_value: float
    min_prediction_error_eigenvalue: float
    innovation_condition_number: float


def _require_positive_definite(matrices: np.ndarray, label: str) -> np.ndarray:
    """Reject a batch of matrices that is not positive definite and return the eigenvalues."""
    hermitian = 0.5 * (matrices + matrices.conj().swapaxes(-1, -2))
    eigenvalues = np.linalg.eigvalsh(hermitian)
    if not np.all(np.isfinite(eigenvalues)):
        raise FitError(f"The {label} is not finite.")
    if np.any(eigenvalues <= 0.0):
        raise FitError(f"The {label} is not positive definite; the fit is not usable.")
    return eigenvalues


def whittle_levinson_factorization(autocovariance: np.ndarray, order: int) -> WhittleFactorization:
    """Factorize a matrix autocovariance with the Whittle block recursion.

    Args:
        autocovariance: The autocovariance matrices ``R_0 .. R_p``, shape
            ``(order + 1, n_channels, n_channels)``. The block-Toeplitz matrix
            they build must be positive definite.
        order: Model order ``p`` (at least one).

    Returns:
        The fitted autoregressive coefficients, innovation covariance and the
        conditioning diagnostics.

    Raises:
        FitError: If the sequence is too short, not finite, or not positive
            definite at some order of the recursion.
        ValueError: If ``order`` is not a positive integer.
    """
    autocovariance = np.asarray(autocovariance, dtype=float)
    if autocovariance.ndim != AUTOCOVARIANCE_DIMENSIONS or autocovariance.shape[1] != autocovariance.shape[2]:
        raise ValueError("autocovariance must have shape (order + 1, n_channels, n_channels).")
    if order < 1:
        raise ValueError("order must be a positive integer.")
    if autocovariance.shape[0] < order + 1:
        raise FitError("autocovariance must provide R[0] through R[order].")
    if not np.all(np.isfinite(autocovariance)):
        raise FitError("autocovariance must be finite.")

    n_channels = autocovariance.shape[1]
    identity = np.eye(n_channels, dtype=float)

    _require_positive_definite(autocovariance[0][None, ...], "zero-lag autocovariance")

    forward = np.zeros((order + 1, n_channels, n_channels), dtype=float)
    backward = np.zeros((order + 1, n_channels, n_channels), dtype=float)
    innovation = np.zeros((order + 1, n_channels, n_channels), dtype=float)
    backward_innovation = np.zeros((order + 1, n_channels, n_channels), dtype=float)
    forward[0] = identity
    backward[0] = identity
    innovation[0] = autocovariance[0]
    backward_innovation[0] = autocovariance[0]

    min_prediction_eigenvalue = float(np.min(np.linalg.eigvalsh(autocovariance[0])))
    max_reflection_singular_value = 0.0
    reflection_coefficients = np.zeros((order, n_channels, n_channels), dtype=float)
    prediction_errors = np.zeros((order + 1, n_channels, n_channels), dtype=float)
    prediction_errors[0] = autocovariance[0]

    for step in range(order):
        forward_cross = np.zeros((n_channels, n_channels), dtype=float)
        backward_cross = np.zeros((n_channels, n_channels), dtype=float)
        for lag in range(step + 1):
            forward_cross = forward_cross + forward[lag] @ autocovariance[step - lag + 1]
            backward_cross = backward_cross + backward[lag] @ autocovariance[step - lag + 1].T

        next_forward = forward.copy()
        next_backward = backward.copy()
        next_forward[step + 1] = -np.linalg.solve(backward_innovation[step], forward_cross.T).T
        next_backward[step + 1] = -np.linalg.solve(innovation[step], backward_cross.T).T
        for lag in range(1, step + 1):
            next_forward[lag] = forward[lag] + next_forward[step + 1] @ backward[step - lag + 1]
            next_backward[lag] = backward[lag] + next_backward[step + 1] @ forward[step - lag + 1]

        forward = next_forward
        backward = next_backward
        innovation[step + 1] = innovation[step] + forward[step + 1] @ backward_cross
        backward_innovation[step + 1] = backward_innovation[step] + backward[step + 1] @ forward_cross

        eigenvalues = _require_positive_definite(innovation[step + 1][None, ...], "prediction-error covariance")
        min_prediction_eigenvalue = min(min_prediction_eigenvalue, float(np.min(eigenvalues)))

        reflection_coefficients[step] = forward[step + 1]
        prediction_errors[step + 1] = innovation[step + 1]
        max_reflection_singular_value = max(
            max_reflection_singular_value,
            float(np.linalg.svd(forward[step + 1], compute_uv=False)[0]),
        )

    innovation_covariance = innovation[order]
    try:
        innovation_factor = np.linalg.cholesky(innovation_covariance)
    except np.linalg.LinAlgError as error:  # pragma: no cover - guarded by the eigenvalue check
        raise FitError("The innovation covariance is not positive definite.") from error

    return WhittleFactorization(
        ar_coefficients=np.asarray(forward[1:], dtype=float),
        innovation_covariance=np.asarray(innovation_covariance, dtype=float),
        innovation_factor=np.asarray(innovation_factor, dtype=float),
        reflection_coefficients=reflection_coefficients,
        prediction_errors=prediction_errors,
        max_reflection_singular_value=max_reflection_singular_value,
        min_prediction_error_eigenvalue=min_prediction_eigenvalue,
        innovation_condition_number=float(np.linalg.cond(innovation_covariance)),
    )


def matrix_autocovariance(target_matrices: np.ndarray, n_samples: int) -> np.ndarray:
    """Return the matrix autocovariance implied by a one-sided cross-spectrum.

    The transform is the matrix form of the scalar construction the AR simulator
    uses: ``R_tau`` is the inverse real transform of the target on its own grid,
    so the variance ``R_0`` and the whole sequence are those of the sampled
    spectrum rather than of the tabulated curve behind it.

    Args:
        target_matrices: Hermitian spectral matrices on a one-sided grid, shape
            ``(n_frequencies, n_channels, n_channels)``.
        n_samples: Length of the transform; ``2 * (n_frequencies - 1)``.

    Returns:
        The ``(n_samples, n_channels, n_channels)`` real autocovariance sequence.
    """
    return np.asarray(np.fft.irfft(target_matrices, n=n_samples, axis=0), dtype=float)


def matrix_band_fit_residual(  # noqa: PLR0913
    frequencies: np.ndarray,
    target: np.ndarray,
    model: np.ndarray,
    *,
    low_frequency: float,
    high_frequency: float,
    n_bands: int = DEFAULT_FIT_BANDS,
) -> dict[str, object]:
    """Summarise the matrix spectral error band by band.

    The primary number per band is the relative Frobenius error of the band sum,
    ``||sum(model) - sum(target)||_F / ||sum(target)||_F``, which penalises power
    and coherence errors together at the level a simulator has to get right. The
    largest per-bin relative error is recorded alongside it.

    Args:
        frequencies: Frequencies of the target and model samples.
        target: Target spectral matrices, shape ``(n_frequencies, n, n)``.
        model: Model spectral matrices of the same shape.
        low_frequency: Lower edge of the fit band in hertz.
        high_frequency: Upper edge of the fit band in hertz.
        n_bands: Number of geometric bands.

    Returns:
        A JSON-serialisable mapping with the per-band list and the summary
        statistics.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    target = np.asarray(target, dtype=np.complex128)
    model = np.asarray(model, dtype=np.complex128)
    edges = geometric_band_edges(low_frequency, high_frequency, n_bands)

    bands: list[dict[str, float]] = []
    for band_low, band_high in pairwise(edges):
        selected = (frequencies >= band_low) & (frequencies < band_high)
        if not np.any(selected):
            continue
        selected_target = target[selected]
        selected_model = model[selected]
        band_target = np.sum(selected_target, axis=0)
        band_model = np.sum(selected_model, axis=0)
        reference = float(np.linalg.norm(band_target))
        if reference <= 0.0:
            continue
        relative_error = float(np.linalg.norm(band_model - band_target)) / reference
        per_bin = float(
            np.max(
                np.linalg.norm(selected_model - selected_target, axis=(-2, -1))
                / np.maximum(np.linalg.norm(selected_target, axis=(-2, -1)), np.finfo(float).tiny)
            )
        )
        bands.append(
            {
                "low_frequency": float(band_low),
                "high_frequency": float(band_high),
                "relative_error": relative_error,
                "max_bin_relative_error": per_bin,
            }
        )

    relative_errors = [band["relative_error"] for band in bands]
    return {
        "band_count": len(bands),
        "worst_relative_error": max(relative_errors, default=0.0),
        "median_relative_error": float(np.median(relative_errors)) if relative_errors else 0.0,
        "bands": bands,
    }
