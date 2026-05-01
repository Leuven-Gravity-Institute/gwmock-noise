"""PSD diagnostics based on Welch estimation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
from scipy.signal import welch

from gwmock_noise.simulators._spectral import load_spectral_series

_MATPLOTLIB_IMPORT_ERROR = "matplotlib is required to use plot_psd. Install it with `pip install matplotlib`."


def estimate_psd(
    data: np.ndarray,
    sampling_frequency: float,
    segment_duration: float = 4.0,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD in physical density units.

    The returned frequencies are in Hz. For strain input data, the PSD is in
    strain^2/Hz.
    """
    samples = np.asarray(data, dtype=float)
    if samples.ndim != 1:
        raise ValueError("data must be a one-dimensional array.")
    if samples.size == 0:
        raise ValueError("data must contain at least one sample.")
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be greater than zero.")
    if segment_duration <= 0.0:
        raise ValueError("segment_duration must be greater than zero.")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must satisfy 0 <= overlap < 1.")

    nperseg = max(1, int(segment_duration * sampling_frequency))
    nperseg = min(nperseg, samples.size)
    noverlap = min(int(overlap * nperseg), nperseg - 1)

    frequencies, psd = welch(
        samples,
        fs=sampling_frequency,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return np.asarray(frequencies, dtype=float), np.asarray(psd, dtype=float)


def compare_psd(  # noqa: PLR0913
    data: np.ndarray,
    target_psd_file: Path,
    sampling_frequency: float,
    rtol: float = 0.1,
    fmin: float = 10.0,
    fmax: float | None = None,
) -> bool:
    """Compare an estimated PSD against a target PSD file over a frequency band.

    This helper compares the median PSD level in-band rather than every bin
    individually, which makes it robust to the residual variance of a single
    Welch estimate from stochastic noise.
    """
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")
    if fmin < 0.0:
        raise ValueError("fmin must be non-negative.")
    if fmax is not None and fmax <= fmin:
        raise ValueError("fmax must be greater than fmin.")

    frequencies, estimated_psd = estimate_psd(data, sampling_frequency=sampling_frequency)
    target_frequencies, target_psd = load_spectral_series(target_psd_file, kind="PSD")
    interpolated_target = np.interp(frequencies, target_frequencies, target_psd, left=0.0, right=0.0)

    upper_frequency = frequencies[-1] if fmax is None else fmax
    band = (frequencies >= fmin) & (frequencies <= upper_frequency) & (interpolated_target > 0.0)
    if not np.any(band):
        raise ValueError("The requested comparison band contains no positive target PSD values.")

    estimated_level = float(np.median(estimated_psd[band]))
    target_level = float(np.median(interpolated_target[band]))
    relative_error = abs(estimated_level - target_level) / target_level
    return relative_error <= rtol


def plot_psd(frequencies: np.ndarray, psd: np.ndarray, ax=None):
    """Plot a PSD on log-log axes, importing matplotlib only when needed."""
    try:
        pyplot = import_module("matplotlib.pyplot")
    except ImportError as exc:
        raise ImportError(_MATPLOTLIB_IMPORT_ERROR) from exc

    frequency_array = np.asarray(frequencies, dtype=float)
    psd_array = np.asarray(psd, dtype=float)
    if frequency_array.shape != psd_array.shape:
        raise ValueError("frequencies and psd must have the same shape.")

    positive = (frequency_array > 0.0) & (psd_array > 0.0)
    if not np.any(positive):
        raise ValueError("frequencies and psd must contain at least one positive value for log plotting.")

    if ax is None:
        _, ax = pyplot.subplots()

    ax.loglog(frequency_array[positive], psd_array[positive])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [strain^2/Hz]")
    return ax
