"""Tests for built-in PSD diagnostics."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.diagnostics import compare_psd, estimate_psd, plot_psd
from gwmock_noise.simulators import ColoredNoiseSimulator


def _write_psd_file(path: Path, *, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def test_psd_helpers_are_importable_from_top_level_package() -> None:
    """Top-level package re-exports the PSD diagnostics helpers."""
    assert gwmock_noise.estimate_psd is estimate_psd
    assert gwmock_noise.compare_psd is compare_psd


def test_estimate_psd_returns_expected_frequency_grid() -> None:
    """Welch PSD estimation returns one-sided arrays with matching shapes."""
    sampling_frequency = 256.0
    data = np.sin(2.0 * np.pi * 32.0 * np.arange(2048) / sampling_frequency)

    frequencies, psd = estimate_psd(data, sampling_frequency=sampling_frequency, segment_duration=2.0, overlap=0.5)

    assert frequencies.shape == psd.shape
    assert frequencies[0] == pytest.approx(0.0)
    assert frequencies[-1] == pytest.approx(sampling_frequency / 2.0)
    assert np.all(np.diff(frequencies) > 0.0)
    assert np.all(psd >= 0.0)


def test_estimate_psd_recovers_colored_noise_target_in_band(tmp_path: Path) -> None:
    """Averaged Welch estimates recover the configured flat PSD in band."""
    psd_path = _write_psd_file(tmp_path / "flat_psd.txt")
    sampling_frequency = 256.0
    band_estimates = []

    for seed in range(12):
        simulator = ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=sampling_frequency,
            seed=seed,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )
        strain = simulator.generate(duration=32.0, sampling_frequency=sampling_frequency, detectors=["H1"])["H1"]
        frequencies, estimated_psd = estimate_psd(strain, sampling_frequency=sampling_frequency, segment_duration=4.0)
        band = (frequencies >= 12.0) & (frequencies <= 80.0)
        band_estimates.append(estimated_psd[band])

    mean_band_psd = np.mean(np.stack(band_estimates), axis=0)
    assert np.median(mean_band_psd) == pytest.approx(2.0e-3, rel=0.1)


def test_compare_psd_returns_true_for_matching_target_psd(tmp_path: Path) -> None:
    """The PSD comparison helper accepts matching colored-noise realizations."""
    psd_path = _write_psd_file(tmp_path / "compare_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=7,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    strain = simulator.generate(duration=64.0, sampling_frequency=256.0, detectors=["H1"])["H1"]

    assert compare_psd(strain, psd_path, sampling_frequency=256.0, rtol=0.1, fmin=12.0, fmax=80.0)


def test_plot_psd_raises_clear_import_error_without_matplotlib(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plotting helper raises ImportError with a helpful message."""
    diagnostics_psd = import_module("gwmock_noise.diagnostics.psd")
    original_import_module = diagnostics_psd.import_module

    def fake_import_module(name: str):
        if name == "matplotlib.pyplot":
            raise ImportError("No module named 'matplotlib'")
        return original_import_module(name)

    monkeypatch.setattr(diagnostics_psd, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="matplotlib is required"):
        plot_psd(np.array([1.0, 2.0]), np.array([1.0e-3, 2.0e-3]))
