"""Signal-agnostic spectral covariance utilities.

The utilities in this module use one-sided PSD/CSD inputs in units of strain^2/Hz
and sample masked real-FFT coefficients with covariance ``S(f) / (2 df)``. The
inverse transform multiplies by ``df * n`` so the generated real time series is
consistent with the one-sided periodogram convention used by the simulators.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from gwmock_noise.simulators._spectral import load_spectral_series

DETECTOR_PAIR_SIZE = 2
StrPath = str | Path

__all__ = [
    "SpectralCovariance",
    "assemble_hermitian_spectral_matrices",
    "build_spectral_covariance_from_files",
    "cholesky_factors_from_spectral_matrices",
    "interpolate_complex_spectral_series",
    "interpolate_real_spectral_series",
    "load_and_interpolate_csd",
    "load_and_interpolate_psd",
    "normalize_csd_mapping",
    "normalize_detector_pair",
    "regularized_cholesky",
    "sample_complex_frequency_coefficients",
    "simulate_spectral_covariance_chunk",
    "time_series_from_frequency_coefficients",
]


@dataclass(frozen=True, slots=True)
class SpectralCovariance:
    """Spectral covariance arrays on a common frequency grid."""

    frequencies: NDArray[np.float64]
    detector_index: dict[str, int]
    psd: dict[str, NDArray[np.float64]]
    csd: dict[tuple[str, str], NDArray[np.complex128]]
    matrices: NDArray[np.complex128]
    cholesky_factors: NDArray[np.complex128]


def normalize_detector_pair(pair: tuple[str, str]) -> tuple[str, str]:
    """Normalize a detector-pair key to sorted order."""
    if len(pair) != DETECTOR_PAIR_SIZE:
        msg = "Detector-pair keys must contain exactly two detector names."
        raise ValueError(msg)
    detector_a, detector_b = tuple(sorted(pair))
    if detector_a == detector_b:
        msg = "Detector-pair keys must reference two distinct detectors."
        raise ValueError(msg)
    return detector_a, detector_b


def normalize_csd_mapping(
    csd_values: Mapping[tuple[str, str], NDArray[np.complex128]],
    *,
    detectors: Sequence[str],
) -> dict[tuple[str, str], NDArray[np.complex128]]:
    """Validate and normalize detector-pair CSD mappings."""
    detector_set = set(detectors)
    normalized: dict[tuple[str, str], NDArray[np.complex128]] = {}
    for pair, values in csd_values.items():
        detector_a, detector_b = normalize_detector_pair(pair)
        if detector_a not in detector_set or detector_b not in detector_set:
            msg = "CSD detector pairs must reference configured detectors."
            raise ValueError(msg)
        normalized_key = (detector_a, detector_b)
        if normalized_key in normalized:
            msg = f"Duplicate CSD mapping for detector pair {detector_a}-{detector_b}."
            raise ValueError(msg)
        normalized[normalized_key] = np.asarray(values, dtype=np.complex128)
    return normalized


def interpolate_real_spectral_series(
    frequencies: NDArray[np.float64],
    values: NDArray[np.float64],
    target_frequencies: NDArray[np.float64],
    *,
    taper: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Interpolate and clip a real non-negative spectral series."""
    interpolated = np.interp(target_frequencies, frequencies, values, left=0.0, right=0.0)
    clipped = np.clip(interpolated, a_min=0.0, a_max=None)
    return clipped if taper is None else clipped * taper


def interpolate_complex_spectral_series(
    frequencies: NDArray[np.float64],
    values: NDArray[np.complex128],
    target_frequencies: NDArray[np.float64],
    *,
    taper: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Interpolate a complex spectral series by real and imaginary parts."""
    real = np.interp(target_frequencies, frequencies, values.real, left=0.0, right=0.0)
    imag = np.interp(target_frequencies, frequencies, values.imag, left=0.0, right=0.0)
    interpolated = real + 1j * imag
    return interpolated if taper is None else interpolated * taper


def load_and_interpolate_psd(
    file_path: StrPath,
    target_frequencies: NDArray[np.float64],
    *,
    taper: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Load and interpolate a one-sided PSD file."""
    frequencies, values = load_spectral_series(file_path, kind="PSD")
    return interpolate_real_spectral_series(frequencies, values, target_frequencies, taper=taper)


def load_and_interpolate_csd(
    file_path: StrPath,
    target_frequencies: NDArray[np.float64],
    *,
    taper: NDArray[np.float64] | None = None,
) -> NDArray[np.complex128]:
    """Load and interpolate a complex one-sided CSD file."""
    frequencies, values = load_spectral_series(file_path, kind="CSD", complex_values=True)
    return interpolate_complex_spectral_series(frequencies, values, target_frequencies, taper=taper)


def assemble_hermitian_spectral_matrices(
    detectors: Sequence[str],
    psd: Mapping[str, NDArray[np.float64]],
    csd: Mapping[tuple[str, str], NDArray[np.complex128]] | None = None,
) -> NDArray[np.complex128]:
    """Assemble one Hermitian spectral covariance matrix per frequency."""
    detector_list = list(detectors)
    if not detector_list:
        msg = "detectors must contain at least one detector."
        raise ValueError(msg)
    if len(set(detector_list)) != len(detector_list):
        msg = "detectors must not contain duplicates."
        raise ValueError(msg)
    if set(psd) != set(detector_list):
        msg = "PSD keys must exactly match detectors."
        raise ValueError(msg)

    first_detector = detector_list[0]
    n_frequencies = np.asarray(psd[first_detector]).shape[0]
    matrices = np.zeros((n_frequencies, len(detector_list), len(detector_list)), dtype=np.complex128)
    detector_index = {detector: index for index, detector in enumerate(detector_list)}

    for detector, values in psd.items():
        psd_values = np.asarray(values, dtype=float)
        if psd_values.shape != (n_frequencies,):
            msg = "All PSD arrays must have shape (n_frequencies,)."
            raise ValueError(msg)
        matrices[:, detector_index[detector], detector_index[detector]] = psd_values

    normalized_csd = normalize_csd_mapping(csd or {}, detectors=detector_list)
    for (detector_a, detector_b), values in normalized_csd.items():
        csd_values = np.asarray(values, dtype=np.complex128)
        if csd_values.shape != (n_frequencies,):
            msg = "All CSD arrays must have shape (n_frequencies,)."
            raise ValueError(msg)
        index_a = detector_index[detector_a]
        index_b = detector_index[detector_b]
        matrices[:, index_a, index_b] = csd_values
        matrices[:, index_b, index_a] = np.conj(csd_values)

    return matrices


def regularized_cholesky(
    spectral_matrix: NDArray[np.complex128],
    *,
    regularization_epsilon: float = 1.0e-12,
) -> NDArray[np.complex128]:
    """Return a numerically stable Cholesky factor for a Hermitian matrix."""
    if regularization_epsilon <= 0:
        msg = "regularization_epsilon must be greater than zero."
        raise ValueError(msg)

    hermitian_matrix = 0.5 * (spectral_matrix + spectral_matrix.conj().T)
    diagonal_scale = max(float(np.max(np.real(np.diag(hermitian_matrix)))), np.finfo(float).tiny)
    epsilon = regularization_epsilon * diagonal_scale
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hermitian_matrix)))

    regularized = hermitian_matrix
    if minimum_eigenvalue < epsilon:
        regularized = regularized + np.eye(hermitian_matrix.shape[0]) * (epsilon - minimum_eigenvalue)

    try:
        return np.linalg.cholesky(regularized)
    except np.linalg.LinAlgError:
        diagonal = np.clip(np.real(np.diag(hermitian_matrix)), a_min=0.0, a_max=None)
        return np.diag(np.sqrt(diagonal + epsilon))


def cholesky_factors_from_spectral_matrices(
    spectral_matrices: NDArray[np.complex128],
    *,
    delta_frequency: float,
    regularization_epsilon: float = 1.0e-12,
) -> NDArray[np.complex128]:
    """Build coefficient-covariance Cholesky factors from one-sided spectra."""
    if delta_frequency <= 0.0:
        msg = "delta_frequency must be positive."
        raise ValueError(msg)
    matrices = np.asarray(spectral_matrices, dtype=np.complex128)
    factors = np.zeros_like(matrices)
    scale = 0.5 / delta_frequency
    for frequency_index, matrix in enumerate(matrices):
        factors[frequency_index] = regularized_cholesky(
            matrix * scale,
            regularization_epsilon=regularization_epsilon,
        )
    return factors


def build_spectral_covariance_from_files(  # noqa: PLR0913
    *,
    detectors: Sequence[str],
    psd_files: Mapping[str, StrPath],
    csd_files: Mapping[tuple[str, str], StrPath] | None,
    frequencies: NDArray[np.float64],
    taper: NDArray[np.float64] | None,
    delta_frequency: float,
    regularization_epsilon: float = 1.0e-12,
) -> SpectralCovariance:
    """Load PSD/CSD files and build spectral covariance products."""
    detector_list = list(detectors)
    if set(psd_files) != set(detector_list):
        msg = "psd_files keys must exactly match detectors."
        raise ValueError(msg)
    psd = {
        detector: load_and_interpolate_psd(str(path), frequencies, taper=taper) for detector, path in psd_files.items()
    }
    normalized_csd_files: dict[tuple[str, str], StrPath] = {}
    for pair, path in (csd_files or {}).items():
        detector_a, detector_b = normalize_detector_pair(pair)
        normalized_key = (detector_a, detector_b)
        if normalized_key in normalized_csd_files:
            msg = f"Duplicate CSD mapping for detector pair {detector_a}-{detector_b}."
            raise ValueError(msg)
        normalized_csd_files[normalized_key] = path
    csd = {
        pair: load_and_interpolate_csd(str(path), frequencies, taper=taper)
        for pair, path in normalized_csd_files.items()
    }
    matrices = assemble_hermitian_spectral_matrices(detector_list, psd, csd)
    factors = cholesky_factors_from_spectral_matrices(
        matrices,
        delta_frequency=delta_frequency,
        regularization_epsilon=regularization_epsilon,
    )
    return SpectralCovariance(
        frequencies=np.asarray(frequencies, dtype=float),
        detector_index={detector: index for index, detector in enumerate(detector_list)},
        psd=psd,
        csd=csd,
        matrices=matrices,
        cholesky_factors=factors,
    )


def _real_only_rfft_grid_indices(window_size: int) -> tuple[int, ...]:
    """Return rfft grid indices that must carry real-only coefficients."""
    indices = [0]
    if window_size % 2 == 0:
        indices.append(window_size // 2)
    return tuple(indices)


def _real_only_masked_coefficient_indices(
    frequency_mask: NDArray[np.bool_],
    *,
    window_size: int,
) -> tuple[int, ...]:
    """Map masked DC/Nyquist grid bins to coefficient rows."""
    masked_grid_indices = np.flatnonzero(frequency_mask)
    coefficient_index_by_grid = {
        grid_index: coefficient_index for coefficient_index, grid_index in enumerate(masked_grid_indices)
    }
    return tuple(
        coefficient_index_by_grid[grid_index]
        for grid_index in _real_only_rfft_grid_indices(window_size)
        if grid_index in coefficient_index_by_grid
    )


def sample_complex_frequency_coefficients(
    rng: np.random.Generator,
    cholesky_factors: NDArray[np.complex128],
    *,
    real_only_indices: Sequence[int] | None = None,
) -> NDArray[np.complex128]:
    """Sample complex Gaussian frequency coefficients from Cholesky factors."""
    factors = np.asarray(cholesky_factors, dtype=np.complex128)
    n_frequencies, n_detectors, _ = factors.shape
    white_noise = (
        rng.standard_normal((n_frequencies, n_detectors)) + 1j * rng.standard_normal((n_frequencies, n_detectors))
    ) / np.sqrt(2.0)
    if real_only_indices:
        real_only = np.asarray(real_only_indices, dtype=int)
        white_noise[real_only] = rng.standard_normal((real_only.size, n_detectors))
    coefficients = np.einsum("fij,fj->fi", factors, white_noise)
    if real_only_indices:
        coefficients[real_only] = coefficients[real_only].real
    return coefficients


def time_series_from_frequency_coefficients(  # noqa: PLR0913
    coefficients: NDArray[np.complex128],
    *,
    detectors: Sequence[str],
    frequency_grid_size: int,
    frequency_mask: NDArray[np.bool_],
    delta_frequency: float,
    window_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Convert masked detector frequency coefficients to real time series."""
    detector_list = list(detectors)
    frequency_series = np.zeros((len(detector_list), frequency_grid_size), dtype=np.complex128)
    frequency_series[:, frequency_mask] = np.asarray(coefficients).T
    real_only_grid_indices = [
        grid_index for grid_index in _real_only_rfft_grid_indices(window_size) if frequency_mask[grid_index]
    ]
    if real_only_grid_indices:
        frequency_series[:, real_only_grid_indices] = frequency_series[:, real_only_grid_indices].real
    time_series = np.fft.irfft(frequency_series, n=window_size, axis=1) * delta_frequency * window_size
    return {detector: time_series[index].copy() for index, detector in enumerate(detector_list)}


def simulate_spectral_covariance_chunk(  # noqa: PLR0913
    rng: np.random.Generator,
    cholesky_factors: NDArray[np.complex128],
    *,
    detectors: Sequence[str],
    frequency_grid_size: int,
    frequency_mask: NDArray[np.bool_],
    delta_frequency: float,
    window_size: int,
) -> dict[str, NDArray[np.float64]]:
    """Sample one real multi-detector chunk from spectral covariance factors."""
    coefficients = sample_complex_frequency_coefficients(
        rng,
        cholesky_factors,
        real_only_indices=_real_only_masked_coefficient_indices(
            frequency_mask,
            window_size=window_size,
        ),
    )
    return time_series_from_frequency_coefficients(
        coefficients,
        detectors=detectors,
        frequency_grid_size=frequency_grid_size,
        frequency_mask=frequency_mask,
        delta_frequency=delta_frequency,
        window_size=window_size,
    )
