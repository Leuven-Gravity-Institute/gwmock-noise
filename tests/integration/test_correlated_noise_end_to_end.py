"""Slow end-to-end tests for correlated detector noise."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators import DefaultNoiseSimulator

pytestmark = [pytest.mark.integration, pytest.mark.slow]

SAMPLING_FREQUENCY = 256.0
DURATION = 64.0
PSD_LEVEL = 2.0e-3
CSD_LEVEL = 6.0e-4
SEGMENT_SIZE = 2048


def _write_psd_file(path: Path, *, value: float = PSD_LEVEL) -> Path:
    """Write a flat one-sided PSD over the simulation band."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, 2049)
    np.savetxt(path, np.column_stack((frequencies, np.full_like(frequencies, value))))
    return path


def _write_csd_file(path: Path, *, value: complex = CSD_LEVEL) -> Path:
    """Write a flat one-sided complex CSD over the simulation band."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, 2049)
    np.save(path, np.column_stack((frequencies, np.full(frequencies.shape, value, dtype=np.complex128))))
    return path


def _welch_psd(strain: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD by averaging overlapping Hann-windowed segments."""
    frequencies, spectra = _welch_csd(strain, strain)
    return frequencies, spectra.real


def _welch_csd(strain_a: np.ndarray, strain_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided cross-spectral density."""
    step = SEGMENT_SIZE // 2
    window = np.hanning(SEGMENT_SIZE)
    scale = 2.0 / (SAMPLING_FREQUENCY * np.sum(window**2))
    spectra = []
    for start in range(0, strain_a.size - SEGMENT_SIZE + 1, step):
        series_a = np.fft.rfft(strain_a[start : start + SEGMENT_SIZE] * window)
        series_b = np.fft.rfft(strain_b[start : start + SEGMENT_SIZE] * window)
        estimate = scale * np.conj(series_a) * series_b
        estimate[0] /= 2.0
        if SEGMENT_SIZE % 2 == 0:
            estimate[-1] /= 2.0
        spectra.append(estimate)
    frequencies = np.fft.rfftfreq(SEGMENT_SIZE, d=1.0 / SAMPLING_FREQUENCY)
    return frequencies, np.mean(np.stack(spectra), axis=0)


def test_default_simulator_writes_valid_three_detector_correlated_noise(tmp_path: Path) -> None:
    """The public runner preserves configured PSD/CSD levels in output files."""
    detectors = ["H1", "L1", "V1"]
    psd_files = {detector: _write_psd_file(tmp_path / f"{detector}_psd.txt") for detector in detectors}
    csd_files = {
        f"{detector_a}-{detector_b}": _write_csd_file(tmp_path / f"{detector_a}_{detector_b}_csd.npy")
        for detector_a, detector_b in combinations(detectors, 2)
    }
    output_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=detectors,
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=8675309,
        output=OutputConfig(directory=output_dir, prefix="correlated"),
        components=[
            {
                "simulator": "correlated",
                "psd_files": psd_files,
                "csd_files": csd_files,
                "low_frequency_cutoff": 8.0,
                "high_frequency_cutoff": 96.0,
            }
        ],
    )

    result = DefaultNoiseSimulator().run(config)

    assert set(result.output_paths) == set(detectors)
    strains = {detector: np.load(path) for detector, path in result.output_paths.items()}
    assert all(strain.shape == (round(DURATION * SAMPLING_FREQUENCY),) for strain in strains.values())
    assert all(np.all(np.isfinite(strain)) for strain in strains.values())
    assert all(np.std(strain) > 0.0 for strain in strains.values())

    metadata = json.loads((output_dir / "correlated_H1.json").read_text())
    assert metadata["implementation"] == "correlated"
    assert metadata["detectors"] == detectors
    assert metadata["correlated_noise"]["csd_files"] == {pair: str(path) for pair, path in csd_files.items()}

    frequencies, psd_h1 = _welch_psd(strains["H1"])
    _, psd_l1 = _welch_psd(strains["L1"])
    _, csd_h1_l1 = _welch_csd(strains["H1"], strains["L1"])
    band = (frequencies >= 16.0) & (frequencies <= 80.0)
    coherence = np.abs(csd_h1_l1[band]) / np.sqrt(psd_h1[band] * psd_l1[band])

    assert np.median(psd_h1[band]) == pytest.approx(PSD_LEVEL, rel=0.35)
    assert np.median(psd_l1[band]) == pytest.approx(PSD_LEVEL, rel=0.35)
    assert np.median(csd_h1_l1.real[band]) == pytest.approx(CSD_LEVEL, rel=0.45)
    assert np.median(np.abs(csd_h1_l1.imag[band])) < 0.5 * CSD_LEVEL
    assert np.median(coherence) == pytest.approx(CSD_LEVEL / PSD_LEVEL, rel=0.45)
