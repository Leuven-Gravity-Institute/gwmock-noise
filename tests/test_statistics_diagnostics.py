"""Tests for Gaussianity and stationarity diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gwmock_noise
from gwmock_noise.diagnostics import DiagnosticResult, run_diagnostics
from gwmock_noise.diagnostics import statistics as statistics_diagnostics
from gwmock_noise.simulators import ColoredNoiseSimulator


def _write_psd_file(path: Path, *, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def test_statistics_helpers_are_importable_from_top_level_package() -> None:
    """Top-level package re-exports the statistics diagnostics helpers."""
    assert gwmock_noise.DiagnosticResult is DiagnosticResult
    assert gwmock_noise.run_diagnostics is run_diagnostics


def test_stationary_gaussian_colored_noise_passes_diagnostics(tmp_path: Path) -> None:
    """Stationary Gaussian colored noise passes both default diagnostics."""
    psd_path = _write_psd_file(tmp_path / "flat_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=7,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    strain = simulator.generate(duration=128.0, sampling_frequency=256.0, detectors=["H1"])["H1"]
    results = run_diagnostics(strain, sampling_frequency=256.0)

    assert results["gaussianity"].passed
    assert results["stationarity"].passed


def test_stationarity_fails_for_step_change_in_variance(tmp_path: Path) -> None:
    """A clear variance step is flagged as non-stationary."""
    psd_path = _write_psd_file(tmp_path / "step_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=7,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    strain = simulator.generate(duration=64.0, sampling_frequency=256.0, detectors=["H1"])["H1"]
    strain[strain.size // 2 :] *= 3.0

    result = statistics_diagnostics.test_stationarity(strain, sampling_frequency=256.0, n_segments=8, alpha=0.05)

    assert not result.passed
    assert "Stationarity test failed" in result.message
    assert "variance changes" in result.message


def test_gaussianity_failure_message_is_human_readable() -> None:
    """Failure messages identify which Gaussianity condition failed."""
    data = np.concatenate((np.zeros(1024), np.full(1024, 10.0)))

    result = statistics_diagnostics.test_gaussianity(data, sampling_frequency=256.0, alpha=0.05)

    assert not result.passed
    assert "Gaussianity test failed" in result.message
    assert "kurtosis" in result.message or "KS test" in result.message


def test_steeply_colored_et_noise_passes_diagnostics() -> None:
    """Regression: stationary Gaussian ET-coloured noise passes both diagnostics.

    The ET sensitivity curves are steeply coloured (orders of magnitude of
    dynamic range), giving correlation times of seconds. Without whitening,
    the Levene segment-variance test rejected the package's own perfectly
    stationary output for every seed (p ~ 1e-11), and the KS Gaussianity
    check rejected it for durations of 128 s and above.
    """
    for seed in (1, 42):
        strain = ColoredNoiseSimulator(psd_file="ET_10_full_cryo_psd").generate(
            duration=128.0, sampling_frequency=1024.0, detectors=["E1"], seed=seed
        )["E1"]
        results = run_diagnostics(strain, sampling_frequency=1024.0)

        assert results["gaussianity"].passed, (seed, results["gaussianity"].message)
        assert results["stationarity"].passed, (seed, results["stationarity"].message)


def test_whiten_false_reproduces_raw_sample_behaviour() -> None:
    """whiten=False tests the raw samples, where coloured noise misfires.

    Documents both the escape hatch and why whitening is the default: on raw
    ET-coloured samples the Levene test's independence assumption is violated
    and stationary noise is flagged as non-stationary; at this duration the
    KS Gaussianity check misfires on the correlated raw samples too.
    """
    strain = ColoredNoiseSimulator(psd_file="ET_10_full_cryo_psd").generate(
        duration=128.0, sampling_frequency=1024.0, detectors=["E1"], seed=42
    )["E1"]
    results = run_diagnostics(strain, sampling_frequency=1024.0, whiten=False)

    assert not results["stationarity"].passed
    assert not results["gaussianity"].passed


def test_variance_step_in_colored_noise_still_detected() -> None:
    """Whitening must not wash out a genuine non-stationarity."""
    strain = np.array(
        ColoredNoiseSimulator(psd_file="ET_10_full_cryo_psd").generate(
            duration=128.0, sampling_frequency=1024.0, detectors=["E1"], seed=42
        )["E1"],
        dtype=float,
    )
    strain[strain.size // 2 :] *= 3.0

    results = run_diagnostics(strain, sampling_frequency=1024.0)

    assert not results["stationarity"].passed


def test_glitches_in_colored_noise_still_detected() -> None:
    """Whitening must not wash out genuine non-Gaussian transients."""
    strain = np.array(
        ColoredNoiseSimulator(psd_file="ET_10_full_cryo_psd").generate(
            duration=128.0, sampling_frequency=1024.0, detectors=["E1"], seed=7
        )["E1"],
        dtype=float,
    )
    rng = np.random.default_rng(0)
    for start_time in rng.uniform(10.0, 118.0, 20):
        index = int(start_time * 1024.0)
        strain[index : index + 8] += 50.0 * strain.std()

    results = run_diagnostics(strain, sampling_frequency=1024.0)

    assert not results["gaussianity"].passed


def test_run_diagnostics_returns_diagnostic_results() -> None:
    """run_diagnostics labels both statistical checks."""
    rng = np.random.default_rng(0)
    results = run_diagnostics(rng.normal(size=4096), sampling_frequency=256.0)

    assert set(results) == {"gaussianity", "stationarity"}
    assert all(isinstance(result, DiagnosticResult) for result in results.values())
