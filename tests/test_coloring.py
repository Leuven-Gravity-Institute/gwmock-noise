"""Tests for the shared glitch PSD-coloring helpers."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise.glitches._coloring import ColoredWaveform, color_whitened_waveform, optimal_snr


def _flat_psd_table(value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return a flat PSD table covering 0-2048 Hz."""
    frequencies = np.linspace(0.0, 2048.0, 65)
    return frequencies, np.full_like(frequencies, value)


def _color(white: np.ndarray, **overrides) -> ColoredWaveform:
    """Color a waveform against the flat table with test-friendly defaults."""
    psd_frequencies, psd_values = _flat_psd_table()
    arguments = {
        "sampling_frequency": 256.0,
        "psd_frequencies": psd_frequencies,
        "psd_values": psd_values,
        "low_frequency_cutoff": 2.0,
        "high_frequency_cutoff": None,
    }
    arguments.update(overrides)
    return color_whitened_waveform(white, **arguments)


def test_rejects_cutoff_beyond_nyquist() -> None:
    """A high-frequency cutoff above Nyquist is rejected."""
    with pytest.raises(ValueError, match="Nyquist"):
        _color(np.ones(64), high_frequency_cutoff=200.0)


def test_rejects_empty_analysis_band() -> None:
    """A band containing no rFFT bins is rejected."""
    with pytest.raises(ValueError, match="no simulation bins"):
        _color(np.ones(16), low_frequency_cutoff=129.0)


def test_zero_psd_yields_zero_waveform_and_snr() -> None:
    """An all-zero PSD colors to silence and reports zero optimal SNR."""
    psd_frequencies, _ = _flat_psd_table()
    colored = _color(
        np.random.default_rng(0).normal(size=128),
        psd_values=np.zeros_like(psd_frequencies),
    )

    assert not np.any(colored.time_series)
    assert optimal_snr(colored, sampling_frequency=256.0) == 0.0


def test_optimal_snr_handles_empty_waveform() -> None:
    """optimal_snr returns zero instead of dividing by a zero-length waveform."""
    empty = np.empty(0, dtype=float)
    colored = ColoredWaveform(
        time_series=empty,
        frequency_series=np.empty(0, dtype=np.complex128),
        interpolated_psd=empty,
        band_mask=np.empty(0, dtype=bool),
    )

    assert optimal_snr(colored, sampling_frequency=256.0) == 0.0


def test_interpolated_psd_is_raw_clipped_and_band_limited() -> None:
    """The returned PSD is unwindowed, non-negative, and zero outside the band."""
    frequencies_table, psd_values = _flat_psd_table()
    psd_values = psd_values.copy()
    negative_knot = 32.0  # inside the 8-96 Hz analysis band
    psd_values[frequencies_table == negative_knot] = -3.0

    colored = _color(
        np.random.default_rng(1).normal(size=256),
        psd_values=psd_values,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    frequencies = np.fft.rfftfreq(256, d=1.0 / 256.0)
    expected_mask = (frequencies >= 8.0) & (frequencies <= 96.0)
    np.testing.assert_array_equal(colored.band_mask, expected_mask)
    assert np.all(colored.interpolated_psd >= 0.0)
    assert not np.any(colored.interpolated_psd[~expected_mask])
    assert not np.any(colored.frequency_series[~expected_mask])
    # The in-band negative table value is clipped to zero at its own bin.
    assert colored.interpolated_psd[frequencies == negative_knot] == 0.0


def test_optimal_snr_matches_independent_computation() -> None:
    """optimal_snr agrees with a direct noise-weighted inner product."""
    colored = _color(np.random.default_rng(2).normal(size=512))
    sampling_frequency = 256.0

    valid = colored.band_mask & (colored.interpolated_psd > 0.0)
    expected = np.sqrt(
        4.0
        * (sampling_frequency / colored.time_series.size)
        * np.sum(np.abs(colored.frequency_series[valid]) ** 2 / colored.interpolated_psd[valid])
    )

    assert optimal_snr(colored, sampling_frequency=sampling_frequency) == pytest.approx(float(expected), rel=1e-12)
    # The time series round-trips to the frequency series it was built from.
    np.testing.assert_allclose(
        np.fft.rfft(colored.time_series) / sampling_frequency,
        colored.frequency_series,
        atol=1e-12,
    )
