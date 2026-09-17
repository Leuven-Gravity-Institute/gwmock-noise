"""Unit tests for the Whittle block-Toeplitz matrix spectral factorization.

The tests pin the construction: the recursion must recover a known
autoregressive matrix filter from its autocovariance, reduce exactly to the
scalar Levinson-Durbin recursion of the AR simulator, satisfy the block
Yule-Walker equations, and refuse a covariance sequence that is not positive
definite rather than clamping it. A known filter and its closed-form
autocovariance are the independent anchor; no simulator is involved here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise.simulators._fit import FitError, levinson_durbin
from gwmock_noise.simulators.matrix_factorization import (
    matrix_autocovariance,
    matrix_band_fit_residual,
    whittle_levinson_factorization,
)

IMPULSE_TRUNCATION = 8192


def _var_autocovariance(
    coefficients: list[np.ndarray],
    innovation_covariance: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Return lag 0..lags of a vector AR process from its impulse response."""
    n_channels = innovation_covariance.shape[0]
    impulse = np.zeros((IMPULSE_TRUNCATION, n_channels, n_channels), dtype=float)
    impulse[0] = np.linalg.cholesky(innovation_covariance)
    for lag in range(1, IMPULSE_TRUNCATION):
        accumulator = np.zeros((n_channels, n_channels), dtype=float)
        for index in range(1, min(lag, len(coefficients)) + 1):
            accumulator = accumulator + coefficients[index - 1] @ impulse[lag - index]
        impulse[lag] = -accumulator

    autocovariance = np.zeros((lags + 1, n_channels, n_channels), dtype=float)
    for lag in range(lags + 1):
        autocovariance[lag] = np.einsum("tij,tkj->ik", impulse[lag:], impulse[: IMPULSE_TRUNCATION - lag])
    return autocovariance


def test_whittle_recovers_a_known_vector_autoregressive_filter() -> None:
    """The recursion inverts the autocovariance of a known VAR(2) process."""
    coefficients = [
        np.array([[0.15, 0.05], [0.0, 0.125]]),
        np.array([[0.03, -0.015], [0.015, 0.06]]),
    ]
    innovation_covariance = np.array([[1.0, 0.4], [0.4, 0.8]])
    autocovariance = _var_autocovariance(coefficients, innovation_covariance, lags=2)

    factor = whittle_levinson_factorization(autocovariance, order=2)

    np.testing.assert_allclose(factor.ar_coefficients, coefficients, atol=1e-8)
    np.testing.assert_allclose(factor.innovation_covariance, innovation_covariance, atol=1e-8)
    assert factor.max_reflection_singular_value < 1.0
    assert factor.min_prediction_error_eigenvalue > 0.0
    np.testing.assert_allclose(
        factor.innovation_factor @ factor.innovation_factor.T,
        innovation_covariance,
        atol=1e-10,
    )


def test_whittle_satisfies_the_block_yule_walker_equations() -> None:
    """The fitted filter whitens every lag up to its order and keeps the innovation power."""
    rng = np.random.default_rng(4)
    n_channels = 2
    order = 6
    impulse = np.zeros((512, n_channels, n_channels))
    for lag in range(512):
        impulse[lag] = rng.standard_normal((n_channels, n_channels)) * 0.5**lag
    autocovariance = np.zeros((order + 1, n_channels, n_channels))
    for lag in range(order + 1):
        autocovariance[lag] = np.einsum("tij,tkj->ik", impulse[lag:], impulse[: 512 - lag])

    factor = whittle_levinson_factorization(autocovariance, order=order)

    for lag in range(order + 1):
        residual = np.zeros((n_channels, n_channels))
        for index in range(order + 1):
            coefficient = np.eye(n_channels) if index == 0 else factor.ar_coefficients[index - 1]
            offset = lag - index
            block = autocovariance[offset] if offset >= 0 else autocovariance[-offset].T
            residual = residual + coefficient @ block
        expected = factor.innovation_covariance if lag == 0 else np.zeros((n_channels, n_channels))
        np.testing.assert_allclose(residual, expected, atol=1e-10)


def test_whittle_reduces_to_levinson_durbin_for_one_channel() -> None:
    """With one channel the block recursion is the scalar Levinson-Durbin recursion."""
    autocovariance = np.array([0.7**lag * np.cos(0.3 * lag) for lag in range(65)])
    scalar = levinson_durbin(autocovariance, order=64)

    factor = whittle_levinson_factorization(autocovariance[:, None, None], order=64)

    np.testing.assert_allclose(factor.ar_coefficients[:, 0, 0], scalar.coefficients, atol=1e-12)
    assert factor.innovation_covariance[0, 0] == pytest.approx(scalar.prediction_error, rel=1e-12)


def test_whittle_rejects_a_non_positive_definite_covariance() -> None:
    """A covariance sequence with a singular block-Toeplitz matrix is a fit failure."""
    autocovariance = np.zeros((3, 2, 2))
    autocovariance[0] = np.eye(2)
    autocovariance[1] = np.array([[0.9, 0.0], [0.0, 0.0]])
    with pytest.raises(FitError, match="positive definite"):
        whittle_levinson_factorization(autocovariance, order=2)


def test_whittle_validates_its_inputs() -> None:
    """Bad shapes, orders and non-finite inputs are refused."""
    covariance = np.eye(2)[None, ...]
    with pytest.raises(ValueError, match="positive integer"):
        whittle_levinson_factorization(covariance, order=0)
    with pytest.raises(ValueError, match="shape"):
        whittle_levinson_factorization(np.zeros((2, 3)), order=1)
    with pytest.raises(FitError, match="R\\[0\\] through R\\[order\\]"):
        whittle_levinson_factorization(covariance, order=3)
    with pytest.raises(FitError, match="finite"):
        whittle_levinson_factorization(np.full((2, 2, 2), np.nan), order=1)


def test_matrix_autocovariance_is_the_inverse_transform_of_the_target() -> None:
    """The autocovariance sequence is the target's own inverse real transform."""
    n_samples = 32
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / 64.0)
    target = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0e-3 * (1.0 + 0.5 * np.cos(2.0 * np.pi * frequencies / 64.0))
    target[:, 1, 1] = 2.0e-3
    target[:, 0, 1] = 5.0e-4
    target[:, 1, 0] = 5.0e-4

    autocovariance = matrix_autocovariance(target, n_samples)

    np.testing.assert_allclose(autocovariance, np.fft.irfft(target, n=n_samples, axis=0), atol=1e-18)


def test_matrix_band_fit_residual_reports_integrated_and_per_bin_errors() -> None:
    """The residual helper compares matrix estimates with a target band by band."""
    frequencies = np.linspace(1.0, 9.0, 9)
    target = np.zeros((9, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0
    target[:, 1, 1] = 1.0
    model = 2.0 * target
    residual = matrix_band_fit_residual(frequencies, target, model, low_frequency=1.0, high_frequency=9.0, n_bands=2)
    assert len(residual["bands"]) == 2
    assert residual["worst_relative_error"] == pytest.approx(1.0)
    assert residual["median_relative_error"] == pytest.approx(1.0)


def _delayed_pair_target(n_samples: int, delay: int, *, sampling_frequency: float = 128.0) -> np.ndarray:
    """Return the spectral matrix of channel 2 being channel 1 delayed by ``delay``.

    With this module's convention the cross-spectrum is ``exp(-i w delay)`` in
    ``[0, 1]``, so the cross-channel autocovariance is a spike at lag ``-delay``.
    """
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    angular = 2.0 * np.pi * frequencies / sampling_frequency
    target = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0
    target[:, 1, 1] = 1.0
    cross = 0.2 * np.exp(-1j * angular * delay)
    target[:, 0, 1] = cross
    target[:, 1, 0] = np.conj(cross)
    for index in (0, frequencies.size - 1):
        target[index, 0, 1] = target[index, 0, 1].real
        target[index, 1, 0] = target[index, 0, 1]
    return target


def test_matrix_autocovariance_carries_a_complex_csd_phase() -> None:
    """A complex CSD phase becomes a full-weight cross-channel lag.

    The exact reference is the explicit full two-sided Hermitian inverse
    transform, whose negative-frequency block is the conjugate of the one-sided
    block. The phase-induced cross-lag must appear at the delayed lag with the
    cross-spectrum's amplitude (``0.2``), not half of it, and the sequence must
    stay real and stationary.
    """
    n_samples = 64
    delay = 3
    target = _delayed_pair_target(n_samples, delay)

    autocovariance = matrix_autocovariance(target, n_samples)

    full_spectrum = np.empty((n_samples, 2, 2), dtype=np.complex128)
    full_spectrum[: target.shape[0]] = target
    full_spectrum[target.shape[0] :] = np.conj(target[1 : target.shape[0] - 1][::-1])
    reference = np.fft.ifft(full_spectrum, axis=0).real
    np.testing.assert_allclose(autocovariance, reference, atol=1e-12)

    assert autocovariance[delay, 0, 1] == pytest.approx(0.2, abs=1e-9)
    assert autocovariance[0, 0, 1] == pytest.approx(0.0, abs=1e-9)
    assert autocovariance[(n_samples - delay) % n_samples, 1, 0] == pytest.approx(0.2, abs=1e-9)
    np.testing.assert_allclose(
        autocovariance[(n_samples - delay) % n_samples],
        autocovariance[delay].T,
        atol=1e-12,
    )


def test_matrix_autocovariance_rejects_a_non_hermitian_target() -> None:
    """A target whose cross-spectrum is not Hermitian is refused, not silently dropped."""
    target = _delayed_pair_target(64, 3)
    target[:, 1, 0] = target[:, 0, 1]
    with pytest.raises(FitError, match="Hermitian"):
        matrix_autocovariance(target, 64)


def test_matrix_autocovariance_rejects_a_complex_endpoint_block() -> None:
    """A complex zero-frequency or Nyquist block is not a real process and is refused."""
    target = _delayed_pair_target(64, 3)
    target[0, 0, 1] = 0.2 + 0.1j
    target[0, 1, 0] = np.conj(target[0, 0, 1])
    with pytest.raises(FitError, match="Nyquist"):
        matrix_autocovariance(target, 64)


def test_matrix_autocovariance_requires_a_matching_length() -> None:
    """The transform length is fixed by the one-sided grid, not free."""
    target = _delayed_pair_target(64, 3)
    with pytest.raises(ValueError, match="n_samples"):
        matrix_autocovariance(target, 32)
