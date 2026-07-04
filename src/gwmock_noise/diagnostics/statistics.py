"""Statistical diagnostics for generated noise."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest, kurtosis, levene

from gwmock_noise.diagnostics.psd import estimate_psd

KURTOSIS_THRESHOLD = 1.0
DECORRELATION_DIVISOR = 16.0
MIN_SEGMENTS = 2
_WHITENING_SEGMENT_DURATION = 4.0  # Welch segment used for the whitening PSD and the edge trim
# Bins whose estimated PSD sits this far below the median carry no real signal
# power (they are spectral-leakage floor from band-limited synthesis); whitening
# would amplify that junk to unit level, so they are zeroed instead. Physical
# sensitivity curves span ~1e2-1e3 around their median, comfortably above this.
_WHITENING_RELATIVE_FLOOR = 1e-6


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
    whiten: bool = True,
) -> DiagnosticResult:
    """Test whether a realization is consistent with Gaussian noise.

    Coloured noise is whitened first (see ``_whiten_samples``): the KS test
    assumes independent samples, which steeply coloured spectra violate.
    Pass ``whiten=False`` only for data that is already broadband.
    """
    samples = _validate_samples(data)
    _validate_sampling_frequency(sampling_frequency)
    _validate_alpha(alpha)
    if whiten:
        samples = _whiten_samples(samples, sampling_frequency)

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
    whiten: bool = True,
) -> DiagnosticResult:
    """Test whether segment variances are consistent with stationarity.

    Coloured noise is whitened first (see ``_whiten_samples``): the Levene
    variance-homogeneity test assumes (nearly) independent samples, and the
    long correlation times of steeply coloured spectra make segment variances
    genuinely differ even for perfectly stationary noise. Pass
    ``whiten=False`` only for data that is already broadband.
    """
    samples = _validate_samples(data)
    _validate_sampling_frequency(sampling_frequency)
    _validate_alpha(alpha)
    if n_segments < MIN_SEGMENTS:
        raise ValueError(f"n_segments must be at least {MIN_SEGMENTS}.")
    if whiten:
        samples = _whiten_samples(samples, sampling_frequency)

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


def run_diagnostics(
    data: np.ndarray,
    sampling_frequency: float,
    whiten: bool = True,
) -> dict[str, DiagnosticResult]:
    """Run the default Gaussianity and stationarity diagnostics.

    Data is whitened against its own Welch PSD estimate before testing (see
    ``_whiten_samples``), which makes both checks applicable to coloured
    noise. Pass ``whiten=False`` to test the raw samples instead — only
    meaningful for broadband data.
    """
    if whiten:
        samples = _validate_samples(data)
        _validate_sampling_frequency(sampling_frequency)
        data = _whiten_samples(samples, sampling_frequency)
    return {
        "gaussianity": test_gaussianity(data, sampling_frequency=sampling_frequency, whiten=False),
        "stationarity": test_stationarity(data, sampling_frequency=sampling_frequency, whiten=False),
    }


def _whiten_samples(samples: np.ndarray, sampling_frequency: float) -> np.ndarray:
    """Whiten a realization against its own Welch PSD estimate.

    Steeply coloured spectra (e.g. Einstein Telescope sensitivity curves)
    have correlation times of seconds, which breaks the independence
    assumptions of the KS and Levene tests: perfectly stationary Gaussian
    coloured noise was systematically flagged as non-stationary. Dividing
    the spectrum by the amplitude spectral density estimated from the same
    data yields an approximately white, unit-level series on which the
    tests are valid.

    Bins with no meaningful estimated power (below ``_WHITENING_RELATIVE_FLOOR``
    of the median positive PSD — the leakage floor of band-limited data) are
    zeroed rather than divided, since whitening would amplify their numerical
    junk to unit level. The circular FFT treats the series as periodic, so the
    wrap-around discontinuity contaminates roughly one whitening-filter length
    at each end; those edges are trimmed from the returned series.
    """
    frequencies, psd = estimate_psd(
        samples,
        sampling_frequency=sampling_frequency,
        segment_duration=_WHITENING_SEGMENT_DURATION,
    )
    spectrum = np.fft.rfft(samples)
    fft_frequencies = np.fft.rfftfreq(samples.size, d=1.0 / sampling_frequency)
    interpolated_psd = np.interp(fft_frequencies, frequencies, psd)
    positive_psd = psd[psd > 0.0]
    floor = _WHITENING_RELATIVE_FLOOR * float(np.median(positive_psd)) if positive_psd.size else 0.0
    amplitude = np.sqrt(interpolated_psd)
    whitened_spectrum = np.zeros_like(spectrum)
    keep = interpolated_psd > floor
    whitened_spectrum[keep] = spectrum[keep] / amplitude[keep]
    whitened = np.fft.irfft(whitened_spectrum, n=samples.size)

    # The whitening filter's impulse response is bounded by the Welch segment
    # length (the estimate's spectral resolution); trim that much from each
    # end, falling back to an eighth of the data when the series is short.
    edge = round(_WHITENING_SEGMENT_DURATION * sampling_frequency)
    if samples.size <= 4 * edge:
        edge = samples.size // 8
    return whitened[edge : samples.size - edge] if edge > 0 else whitened


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
