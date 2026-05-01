"""Statistical diagnostics for generated noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest, kurtosis, levene

KURTOSIS_THRESHOLD = 1.0
DECORRELATION_DIVISOR = 16.0
MIN_SEGMENTS = 2


@dataclass(slots=True)
class DiagnosticResult:
    """Outcome of a statistical diagnostic check."""

    passed: bool
    statistic: float
    p_value: float
    message: str


def _test_gaussianity_impl(
    data: np.ndarray,
    sampling_frequency: float,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Test whether a realization is consistent with Gaussian noise."""
    samples = _validate_samples(data)
    _validate_sampling_frequency(sampling_frequency)
    _validate_alpha(alpha)

    mean = float(np.mean(samples))
    std = float(np.std(samples, ddof=0))
    if std == 0.0:
        return DiagnosticResult(
            passed=False,
            statistic=0.0,
            p_value=0.0,
            message="Gaussianity test failed: data have zero variance, so a normal fit is undefined.",
        )

    z_scores = (samples - mean) / std
    ks_result = kstest(z_scores, "norm")
    excess_kurtosis = float(kurtosis(samples, fisher=True, bias=False))
    passed = bool(ks_result.pvalue > alpha and abs(excess_kurtosis) < KURTOSIS_THRESHOLD)

    if passed:
        message = (
            "Gaussianity test passed: KS p-value "
            f"{ks_result.pvalue:.3g} > alpha {alpha:.3g} and excess kurtosis {excess_kurtosis:.3f} is within "
            f"+/-{KURTOSIS_THRESHOLD:.1f}."
        )
    elif ks_result.pvalue <= alpha:
        message = (
            "Gaussianity test failed: KS test rejected the fitted normal distribution "
            f"(p={ks_result.pvalue:.3g} <= alpha {alpha:.3g})."
        )
    else:
        message = (
            "Gaussianity test failed: excess kurtosis "
            f"{excess_kurtosis:.3f} exceeds the allowed magnitude of {KURTOSIS_THRESHOLD:.1f}."
        )

    return DiagnosticResult(
        passed=passed,
        statistic=float(ks_result.statistic),
        p_value=float(ks_result.pvalue),
        message=message,
    )


def _test_stationarity_impl(
    data: np.ndarray,
    sampling_frequency: float,
    n_segments: int = 8,
    alpha: float = 0.05,
) -> DiagnosticResult:
    """Test whether segment variances are consistent with stationarity."""
    samples = _validate_samples(data)
    _validate_sampling_frequency(sampling_frequency)
    _validate_alpha(alpha)
    if n_segments < MIN_SEGMENTS:
        raise ValueError(f"n_segments must be at least {MIN_SEGMENTS}.")

    decorrelation_step = max(1, int(sampling_frequency / DECORRELATION_DIVISOR))
    decorrelated = samples[::decorrelation_step]
    if decorrelated.size < n_segments:
        raise ValueError("data must contain at least one decorrelated sample per segment.")

    segments = np.array_split(decorrelated, n_segments)
    if any(segment.size == 0 for segment in segments):
        raise ValueError("n_segments must not exceed the number of decorrelated samples.")

    levene_result = levene(*segments, center="median")
    passed = bool(levene_result.pvalue > alpha)
    if passed:
        message = (
            "Stationarity test passed: Levene variance-homogeneity p-value "
            f"{levene_result.pvalue:.3g} > alpha {alpha:.3g} across {n_segments} segments."
        )
    else:
        message = (
            "Stationarity test failed: Levene variance-homogeneity test found segment variance changes "
            f"(p={levene_result.pvalue:.3g} <= alpha {alpha:.3g}) across {n_segments} segments."
        )

    return DiagnosticResult(
        passed=passed,
        statistic=float(levene_result.statistic),
        p_value=float(levene_result.pvalue),
        message=message,
    )


def run_diagnostics(data: np.ndarray, sampling_frequency: float) -> dict[str, DiagnosticResult]:
    """Run the default Gaussianity and stationarity diagnostics."""
    return {
        "gaussianity": test_gaussianity(data, sampling_frequency=sampling_frequency),
        "stationarity": test_stationarity(data, sampling_frequency=sampling_frequency),
    }


def _validate_samples(data: np.ndarray) -> np.ndarray:
    """Validate the diagnostic input array."""
    samples = np.asarray(data, dtype=float)
    if samples.ndim != 1:
        raise ValueError("data must be a one-dimensional array.")
    if samples.size == 0:
        raise ValueError("data must contain at least one sample.")
    if not np.all(np.isfinite(samples)):
        raise ValueError("data must contain only finite samples.")
    return samples


def _validate_sampling_frequency(sampling_frequency: float) -> None:
    """Validate the sampling frequency argument."""
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be finite and greater than zero.")


def _validate_alpha(alpha: float) -> None:
    """Validate the significance threshold."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must satisfy 0 < alpha < 1.")


test_gaussianity = _test_gaussianity_impl
test_stationarity = _test_stationarity_impl
test_gaussianity.__test__ = False
test_stationarity.__test__ = False
