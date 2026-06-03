"""Tests for public spectral-covariance utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.simulators import CorrelatedNoiseSimulator
from gwmock_noise.simulators.colored import _tukey_window
from gwmock_noise.spectral import (
    assemble_hermitian_spectral_matrices,
    build_spectral_covariance_from_files,
    interpolate_complex_spectral_series,
    interpolate_real_spectral_series,
    normalize_detector_pair,
    regularized_cholesky,
    sample_complex_frequency_coefficients,
    simulate_spectral_covariance_chunk,
    time_series_from_frequency_coefficients,
)

FLAT_PSD = 2.0e-3
FLAT_CSD = 8.0e-4 + 1.0e-4j


def _write_psd_file(path: Path, *, value: float = FLAT_PSD) -> Path:
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _write_csd_file(path: Path, *, value: complex = FLAT_CSD) -> Path:
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full(frequencies.shape, value, dtype=np.complex128)
    np.save(path, np.column_stack((frequencies, values)))
    return path


def test_interpolate_real_spectral_series_clips_and_tapers() -> None:
    """Real spectral interpolation clips negative PSD values."""
    frequencies = np.array([0.0, 1.0, 2.0])
    values = np.array([1.0, -1.0, 3.0])
    target = np.array([0.5, 1.0, 1.5, 3.0])
    taper = np.array([1.0, 0.5, 2.0, 1.0])

    interpolated = interpolate_real_spectral_series(frequencies, values, target, taper=taper)

    np.testing.assert_allclose(interpolated, np.array([0.0, 0.0, 2.0, 0.0]))


def test_interpolate_complex_spectral_series_interpolates_components() -> None:
    """Complex CSD interpolation treats real and imaginary parts independently."""
    frequencies = np.array([0.0, 2.0])
    values = np.array([1.0 + 2.0j, 3.0 - 2.0j])
    target = np.array([0.0, 1.0, 2.0])

    interpolated = interpolate_complex_spectral_series(frequencies, values, target)

    np.testing.assert_allclose(interpolated, np.array([1.0 + 2.0j, 2.0 + 0.0j, 3.0 - 2.0j]))


def test_assemble_hermitian_spectral_matrices_places_psd_and_csd() -> None:
    """PSD arrays populate diagonals and CSD arrays populate conjugate pairs."""
    detectors = ["H1", "L1", "V1"]
    psd = {detector: np.array([1.0, 2.0]) for detector in detectors}
    csd = {("L1", "H1"): np.array([0.1 + 0.2j, 0.3 + 0.4j])}

    matrices = assemble_hermitian_spectral_matrices(detectors, psd, csd)

    assert matrices.shape == (2, 3, 3)
    np.testing.assert_allclose(np.diagonal(matrices, axis1=1, axis2=2), np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]))
    np.testing.assert_allclose(matrices[:, 0, 1], np.array([0.1 + 0.2j, 0.3 + 0.4j]))
    np.testing.assert_allclose(matrices[:, 1, 0], np.array([0.1 - 0.2j, 0.3 - 0.4j]))
    np.testing.assert_allclose(matrices, np.swapaxes(matrices.conj(), 1, 2))


def test_regularized_cholesky_handles_indefinite_matrix() -> None:
    """Regularization produces a finite factor for a slightly indefinite matrix."""
    matrix = np.array([[1.0, 1.1], [1.1, 1.0]], dtype=np.complex128)

    factor = regularized_cholesky(matrix)
    regularized = factor @ factor.conj().T

    assert np.all(np.isfinite(factor))
    assert np.min(np.linalg.eigvalsh(regularized)) > 0.0


def test_regularized_cholesky_preserves_tiny_physical_scale() -> None:
    """Regularization must stay relative for SGWB-scale strain spectra."""
    matrix = np.array([[1.0e-47, -3.7e-48], [-3.7e-48, 1.0e-47]], dtype=np.complex128)

    factor = regularized_cholesky(matrix)
    reconstructed = factor @ factor.conj().T

    np.testing.assert_allclose(reconstructed, matrix, rtol=1.0e-9, atol=1.0e-60)


def test_public_covariance_api_reproduces_correlated_simulator_spectra(tmp_path: Path) -> None:
    """Public covariance builders reproduce the simulator's flat PSD/CSD setup."""
    detectors = ["H1", "L1"]
    psd_files = {detector: _write_psd_file(tmp_path / f"{detector}_psd.txt") for detector in detectors}
    csd_files = {("H1", "L1"): _write_csd_file(tmp_path / "H1_L1_csd.npy")}
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=123,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )
    masked_frequencies = simulator._frequency_grid[simulator._frequency_mask]
    taper = _tukey_window(masked_frequencies.size)

    covariance = build_spectral_covariance_from_files(
        detectors=detectors,
        psd_files=psd_files,
        csd_files=csd_files,
        frequencies=masked_frequencies,
        taper=taper,
        delta_frequency=simulator._delta_frequency,
        regularization_epsilon=simulator.regularization_epsilon,
    )

    np.testing.assert_allclose(covariance.matrices, simulator._spectral_matrices)
    np.testing.assert_allclose(covariance.cholesky_factors, simulator._cholesky_factors)
    np.testing.assert_allclose(covariance.psd["H1"], FLAT_PSD * taper)
    np.testing.assert_allclose(covariance.csd[("H1", "L1")], FLAT_CSD * taper)


def test_simulate_spectral_covariance_chunk_returns_real_detector_series() -> None:
    """Spectral covariance factors can be sampled into real multi-detector chunks."""
    rng = np.random.default_rng(10)
    detectors = ["H1", "L1"]
    mask = np.array([False, True, True, False, False])
    cholesky_factors = np.tile(np.eye(2, dtype=np.complex128)[None, :, :], (2, 1, 1))

    chunk = simulate_spectral_covariance_chunk(
        rng,
        cholesky_factors,
        detectors=detectors,
        frequency_grid_size=mask.size,
        frequency_mask=mask,
        delta_frequency=0.25,
        window_size=8,
    )

    assert set(chunk) == set(detectors)
    assert all(strain.shape == (8,) for strain in chunk.values())
    assert all(np.issubdtype(strain.dtype, np.floating) for strain in chunk.values())


def test_endpoint_frequency_bins_are_real_only_before_irfft() -> None:
    """Masked DC and Nyquist bins must stay real before np.fft.irfft."""
    rng = np.random.default_rng(0)
    detectors = ["H1"]
    window_size = 8
    frequency_grid_size = window_size // 2 + 1
    frequency_mask = np.ones(frequency_grid_size, dtype=bool)
    cholesky_factors = np.tile(np.eye(1, dtype=np.complex128)[None, :, :], (frequency_grid_size, 1, 1))

    coefficients = sample_complex_frequency_coefficients(
        rng,
        cholesky_factors,
        real_only_indices=(0, frequency_grid_size - 1),
    )

    assert np.allclose(coefficients[0].imag, 0.0)
    assert np.allclose(coefficients[-1].imag, 0.0)

    chunk = time_series_from_frequency_coefficients(
        coefficients,
        detectors=detectors,
        frequency_grid_size=frequency_grid_size,
        frequency_mask=frequency_mask,
        delta_frequency=0.25,
        window_size=window_size,
    )

    assert chunk["H1"].shape == (window_size,)
    assert np.all(np.isfinite(chunk["H1"]))

    chunk = simulate_spectral_covariance_chunk(
        np.random.default_rng(1),
        cholesky_factors,
        detectors=detectors,
        frequency_grid_size=frequency_grid_size,
        frequency_mask=frequency_mask,
        delta_frequency=0.25,
        window_size=window_size,
    )

    assert chunk["H1"].shape == (window_size,)
    assert np.all(np.isfinite(chunk["H1"]))


def test_normalize_detector_pair_rejects_auto_pairs() -> None:
    """CSD keys must refer to two distinct detectors."""
    with pytest.raises(ValueError, match="distinct"):
        normalize_detector_pair(("H1", "H1"))
