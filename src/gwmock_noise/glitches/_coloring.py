"""Shared PSD-coloring helpers for whitened glitch waveforms."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from gwmock_noise.simulators.colored import _resolve_taper_alpha, _tukey_window


class ColoredWaveform(NamedTuple):
    """Colored waveform pieces returned by :func:`color_whitened_waveform`.

    Attributes:
        time_series: Colored time-domain strain.
        frequency_series: Colored one-sided frequency series (``rfft / fs``).
        interpolated_psd: Raw interpolated PSD on the rFFT grid (pre-window,
            clipped to be non-negative, zero outside the analysis band).
        band_mask: Boolean mask selecting the analysis-band frequency bins,
            with the DC and Nyquist bins excluded.
    """

    time_series: np.ndarray
    frequency_series: np.ndarray
    interpolated_psd: np.ndarray
    band_mask: np.ndarray


def color_whitened_waveform(  # noqa: PLR0913
    white_waveform: np.ndarray,
    *,
    sampling_frequency: float,
    psd_frequencies: np.ndarray,
    psd_values: np.ndarray,
    low_frequency_cutoff: float,
    high_frequency_cutoff: float | None,
) -> ColoredWaveform:
    """Color a whitened waveform against a PSD inside the analysis band.

    Args:
        white_waveform: Whitened time-domain waveform.
        sampling_frequency: Sampling frequency of ``white_waveform`` in Hz.
        psd_frequencies: Frequencies of the PSD table in Hz.
        psd_values: PSD values matching ``psd_frequencies``.
        low_frequency_cutoff: Lower edge of the analysis band in Hz.
        high_frequency_cutoff: Upper edge of the analysis band in Hz, or
            ``None`` to use the Nyquist frequency.

    Returns:
        The colored waveform pieces, including the raw interpolated PSD used
        for noise-weighted inner products.

    Raises:
        ValueError: If the analysis band is invalid or empty.
    """
    if sampling_frequency <= 0.0:
        raise ValueError("sampling_frequency must be greater than zero.")
    n_samples = int(white_waveform.size)
    if n_samples == 0:
        raise ValueError("white_waveform must contain at least one sample.")
    nyquist = sampling_frequency / 2.0
    if high_frequency_cutoff is None:
        high_frequency_cutoff = nyquist
    if high_frequency_cutoff > nyquist:
        raise ValueError("high_frequency_cutoff must not exceed the Nyquist frequency.")

    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    # Exclude the DC and Nyquist bins: they are real-valued in the rFFT and the
    # one-sided ``4 df sum(|h|^2 / S)`` inner product would over-weight them by a
    # factor of two, biasing the SNR calibration at the band edges.
    frequency_mask = (
        (frequencies > 0.0)
        & (frequencies < nyquist)
        & (frequencies >= low_frequency_cutoff)
        & (frequencies <= high_frequency_cutoff)
    )
    if not np.any(frequency_mask):
        raise ValueError("The requested frequency range contains no simulation bins.")

    interpolated_psd = np.zeros_like(frequencies, dtype=float)
    masked_frequencies = frequencies[frequency_mask]
    interpolated_psd[frequency_mask] = np.interp(
        masked_frequencies,
        psd_frequencies,
        psd_values,
        left=0.0,
        right=0.0,
    )
    interpolated_psd[frequency_mask] = np.clip(interpolated_psd[frequency_mask], a_min=0.0, a_max=None)
    coloring_psd = interpolated_psd.copy()
    coloring_psd[frequency_mask] *= _tukey_window(masked_frequencies.size, _resolve_taper_alpha(masked_frequencies))

    white_waveform_fd = np.fft.rfft(white_waveform) / sampling_frequency
    frequency_series = np.zeros_like(white_waveform_fd, dtype=np.complex128)
    frequency_series[frequency_mask] = white_waveform_fd[frequency_mask] * np.sqrt(coloring_psd[frequency_mask])
    time_series = np.fft.irfft(frequency_series, n=n_samples) * sampling_frequency
    return ColoredWaveform(
        time_series=time_series,
        frequency_series=frequency_series,
        interpolated_psd=interpolated_psd,
        band_mask=frequency_mask,
    )


def optimal_snr(colored: ColoredWaveform, *, sampling_frequency: float) -> float:
    """Compute the optimal SNR of a colored waveform against its PSD.

    Uses the standard noise-weighted inner product
    ``rho^2 = 4 df sum(|h(f)|^2 / S(f))`` over the analysis band, skipping
    bins where the PSD vanishes (those carry no colored signal energy).

    The denominator must be the raw interpolated PSD, not the Tukey-windowed
    PSD used for coloring: dividing by the windowed PSD would cancel the taper
    and overstate rho at the band edges relative to a matched filter against
    the true PSD, so rescaled waveforms would land below their target SNR.

    Args:
        colored: Output of :func:`color_whitened_waveform`.
        sampling_frequency: Sampling frequency of the waveform in Hz.

    Returns:
        The optimal SNR.
    """
    n_samples = colored.time_series.size
    if n_samples == 0:
        return 0.0
    delta_frequency = sampling_frequency / n_samples
    valid = colored.band_mask & (colored.interpolated_psd > 0.0)
    if not np.any(valid):
        return 0.0
    snr_squared = (
        4.0
        * delta_frequency
        * float(np.sum(np.abs(colored.frequency_series[valid]) ** 2 / colored.interpolated_psd[valid]))
    )
    return float(np.sqrt(snr_squared))
