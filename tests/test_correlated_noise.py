"""Tests for the correlated-noise simulator."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators import CorrelatedNoiseSimulator, DefaultNoiseSimulator

FLAT_PSD = 2.0e-3
FLAT_CSD = 8.0e-4


def _write_psd_file(path: Path, *, value: float = FLAT_PSD) -> Path:
    """Write a flat PSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _write_csd_file(path: Path, *, value: complex = FLAT_CSD) -> Path:
    """Write a flat complex CSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full(frequencies.shape, value, dtype=np.complex128)
    np.save(path, np.column_stack((frequencies, values)))
    return path


def _estimate_one_sided_psd(strain: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided periodogram."""
    n_samples = strain.size
    frequency_series = np.fft.rfft(strain)
    psd = (2.0 / (sampling_frequency * n_samples)) * np.abs(frequency_series) ** 2
    psd[0] /= 2.0
    if n_samples % 2 == 0:
        psd[-1] /= 2.0
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    return frequencies, psd


def _estimate_one_sided_csd(
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided cross-spectral density."""
    n_samples = strain_a.size
    series_a = np.fft.rfft(strain_a)
    series_b = np.fft.rfft(strain_b)
    csd = (2.0 / (sampling_frequency * n_samples)) * (np.conj(series_a) * series_b)
    csd[0] /= 2.0
    if n_samples % 2 == 0:
        csd[-1] /= 2.0
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / sampling_frequency)
    return frequencies, csd


def _build_spectral_inputs(
    tmp_path: Path,
    detectors: list[str],
    *,
    psd_value: float = FLAT_PSD,
    csd_value: complex = FLAT_CSD,
) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    """Create PSD and CSD files for a detector network."""
    psd_files = {detector: _write_psd_file(tmp_path / f"{detector}_psd.txt", value=psd_value) for detector in detectors}
    csd_files = {
        pair: _write_csd_file(tmp_path / f"{pair[0]}_{pair[1]}_csd.npy", value=csd_value)
        for pair in combinations(sorted(detectors), 2)
    }
    return psd_files, csd_files


def test_correlated_simulator_is_importable_from_top_level_package() -> None:
    """CorrelatedNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.CorrelatedNoiseSimulator is CorrelatedNoiseSimulator


def test_default_simulator_uses_correlated_noise_when_csd_is_configured(tmp_path: Path) -> None:
    """DefaultNoiseSimulator dispatches to CorrelatedNoiseSimulator when configured."""
    detectors = ["H1", "L1"]
    psd_files, _ = _build_spectral_inputs(tmp_path, detectors)
    csd_config = {"H1-L1": _write_csd_file(tmp_path / "dispatch_csd.npy")}
    out_dir = tmp_path / "output"

    config = NoiseConfig(
        detectors=detectors,
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="correlated"),
        seed=42,
        psd_files=psd_files,
        csd_files=csd_config,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "correlated_H1.json").read_text())
    assert metadata["implementation"] == "correlated"
    assert metadata["correlated_noise"]["psd_files"] == {detector: str(path) for detector, path in psd_files.items()}
    assert metadata["correlated_noise"]["csd_files"] == {"H1-L1": str(csd_config["H1-L1"])}
    assert metadata["correlated_noise"]["low_frequency_cutoff"] == 8.0
    assert metadata["correlated_noise"]["high_frequency_cutoff"] == 96.0


def test_generated_correlations_match_input_spectra_within_tolerance(tmp_path: Path) -> None:
    """Averaged realizations preserve both PSD and CSD levels."""
    detectors = ["H1", "L1"]
    sampling_frequency = 256.0
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    estimated_psds = []
    estimated_csds = []

    for seed in range(24):
        simulator = CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files=csd_files,
            detectors=detectors,
            sampling_frequency=sampling_frequency,
            seed=seed,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )
        realization = simulator.generate(
            duration=8.0,
            sampling_frequency=sampling_frequency,
            detectors=detectors,
        )
        _, estimated_psd = _estimate_one_sided_psd(realization["H1"], sampling_frequency)
        _, estimated_csd = _estimate_one_sided_csd(realization["H1"], realization["L1"], sampling_frequency)
        estimated_psds.append(estimated_psd)
        estimated_csds.append(estimated_csd)

    mean_psd = np.mean(np.stack(estimated_psds), axis=0)
    mean_csd = np.mean(np.stack(estimated_csds), axis=0)
    frequencies = np.fft.rfftfreq(realization["H1"].size, d=1.0 / sampling_frequency)
    band = (frequencies >= 12.0) & (frequencies <= 80.0)

    assert np.median(mean_psd[band]) == pytest.approx(FLAT_PSD, rel=0.35)
    assert np.median(mean_csd.real[band]) == pytest.approx(FLAT_CSD, rel=0.4)
    assert np.median(np.abs(mean_csd.imag[band])) < 0.2 * FLAT_CSD


@pytest.mark.parametrize("detectors", [["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def test_correlated_simulator_supports_one_two_and_three_detectors(
    tmp_path: Path,
    detectors: list[str],
) -> None:
    """CorrelatedNoiseSimulator supports 1-3 detectors."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=123,
    )

    realization = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)

    assert set(realization) == set(detectors)
    assert all(strain.shape == (1024,) for strain in realization.values())


def test_consecutive_generate_calls_are_continuous_across_detectors(tmp_path: Path) -> None:
    """Joint overlap-add stitching avoids detector boundary jumps."""
    detectors = ["H1", "L1", "V1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=1234,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)

    for detector in detectors:
        combined = np.concatenate((first[detector], second[detector]))
        jumps = np.abs(np.diff(combined))
        boundary_jump = jumps[first[detector].size - 1]
        assert boundary_jump <= np.quantile(jumps, 0.995)


def test_near_singular_spectral_matrices_are_regularized(tmp_path: Path) -> None:
    """Near-singular spectra do not raise during initialization or generation."""
    detectors = ["H1", "L1", "V1"]
    psd_files, _ = _build_spectral_inputs(tmp_path, detectors, psd_value=1.0e-3)
    csd_files = {
        pair: _write_csd_file(tmp_path / f"{pair[0]}_{pair[1]}_near_singular.npy", value=9.99999999e-4)
        for pair in combinations(sorted(detectors), 2)
    }

    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=9,
    )
    realization = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)

    assert all(strain.shape == (1024,) for strain in realization.values())
