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

#: An autocovariance stack carries one lag axis and two channel axes.
AUTOCOVARIANCE_DIMENSIONS = 3

#: A one-sided grid needs at least the zero-frequency and Nyquist bins.
MINIMUM_FREQUENCY_BINS = 2

#: Relative tolerance for the Hermitian symmetry and real endpoints a one-sided
#: cross-spectral matrix must carry before it can be inverse-transformed.
SPECTRUM_SYMMETRY_TOLERANCE = 1e-8


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


def _require_one_sided_spectrum(target_matrices: np.ndarray) -> np.ndarray:
    """Reject a matrix that is not a valid one-sided cross-spectral matrix.

    A real multichannel process has Hermitian spectral matrices whose
    zero-frequency and Nyquist blocks are real, so the inverse transform is real.
    A target that breaks either property cannot be turned into a covariance by
    dropping the offending part: the dropped part is exactly the phase that
    distinguishes one process from another.

    Args:
        target_matrices: The candidate one-sided spectral matrices.

    Returns:
        The target as a complex array.

    Raises:
        ValueError: If the array is not a stack of square matrices.
        FitError: If the target is not Hermitian to tolerance, or one of its
            zero-frequency or Nyquist blocks is not real.
    """
    target_matrices = np.asarray(target_matrices, dtype=np.complex128)
    if target_matrices.ndim != AUTOCOVARIANCE_DIMENSIONS or target_matrices.shape[1] != target_matrices.shape[2]:
        raise ValueError("target_matrices must have shape (n_frequencies, n_channels, n_channels).")
    if target_matrices.shape[0] < MINIMUM_FREQUENCY_BINS:
        raise ValueError("target_matrices must carry at least two frequency bins.")
    if not np.all(np.isfinite(target_matrices)):
        raise FitError("target_matrices must be finite.")

    conjugate_transpose = target_matrices.conj().swapaxes(-1, -2)
    asymmetry = float(np.max(np.abs(target_matrices - conjugate_transpose), initial=0.0))
    scale = max(float(np.max(np.abs(target_matrices), initial=0.0)), np.finfo(float).tiny)
    if asymmetry > SPECTRUM_SYMMETRY_TOLERANCE * scale:
        raise FitError("target_matrices must be Hermitian at every frequency.")

    for index in (0, target_matrices.shape[0] - 1):
        imaginary = float(np.max(np.abs(target_matrices[index].imag), initial=0.0))
        if imaginary > SPECTRUM_SYMMETRY_TOLERANCE * scale:
            raise FitError("The zero-frequency and Nyquist blocks of target_matrices must be real.")
    return target_matrices


def matrix_autocovariance(target_matrices: np.ndarray, n_samples: int) -> np.ndarray:
    """Return the matrix autocovariance implied by a one-sided cross-spectrum.

    The target is completed to a full two-sided spectrum before the inverse
    transform: the negative-frequency block is the element-wise conjugate of the
    one-sided block, ``S(-f) = conj(S(f))``, which is the symmetry a real process
    has. That conjugation is what carries a complex cross-spectrum's phase into
    the cross-channel lag covariances: without it a phase-shifted CSD would be
    fitted as a different, time-reversed process. The transform is the matrix form
    of the one the AR simulator uses, so the sequence describes the sampled
    spectrum rather than the tabulated curve behind it.

    With this convention the matrix ``target_matrices[j]`` is the Fourier
    transform of the autocovariance, ``S(f_j) = sum_tau R_tau exp(-2 pi i f_j tau
    / f_s)`` with ``R_tau[i, k] = E[x_{t + tau, i} x_{t, k}]``. A CSD exported by
    a tool that instead defines ``S(f) = E[conj(X_i(f)) X_k(f)]`` is the conjugate
    of this one; store its conjugate to match.

    Args:
        target_matrices: Hermitian spectral matrices on a one-sided grid, shape
            ``(n_frequencies, n_channels, n_channels)``. The zero-frequency and
            Nyquist blocks must be real.
        n_samples: Length of the transform; ``2 * (n_frequencies - 1)``.

    Returns:
        The ``(n_samples, n_channels, n_channels)`` real autocovariance sequence,
        with ``R[-tau] = R[tau].T``.

    Raises:
        ValueError: If the shape is invalid or ``n_samples`` is not
            ``2 * (n_frequencies - 1)``.
        FitError: If the target is not Hermitian or has a complex zero-frequency
            or Nyquist block.
    """
    target_matrices = _require_one_sided_spectrum(target_matrices)
    n_frequencies = target_matrices.shape[0]
    if n_samples != 2 * (n_frequencies - 1):
        raise ValueError("n_samples must equal 2 * (n_frequencies - 1).")

    full_spectrum = np.empty((n_samples, *target_matrices.shape[1:]), dtype=np.complex128)
    full_spectrum[:n_frequencies] = target_matrices
    full_spectrum[n_frequencies:] = np.conj(target_matrices[1 : n_frequencies - 1][::-1])
    autocovariance = np.fft.ifft(full_spectrum, axis=0)
    return np.ascontiguousarray(autocovariance.real)


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
    largest per-bin relative error is recorded alongside it, over the bins whose
    target is non-zero; a bin where the target itself is exactly zero (e.g. an
    edge taper's zeroed sample) has no well-defined *relative* error against it
    and is excluded rather than compared against a floor value, matching the
    band-level skip when a whole band's target sum is zero.

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
        target_norms = np.linalg.norm(selected_target, axis=(-2, -1))
        positive = target_norms > 0.0
        per_bin = (
            float(
                np.max(
                    np.linalg.norm(selected_model[positive] - selected_target[positive], axis=(-2, -1))
                    / target_norms[positive]
                )
            )
            if np.any(positive)
            else float("nan")
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
