"""Tests for the colored-noise simulator."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators import ColoredNoiseSimulator, DefaultNoiseSimulator


def _write_psd_file(path: Path) -> Path:
    """Write a flat PSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, 2.0e-3)
    np.savetxt(path, np.column_stack((frequencies, values)))
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


def _write_npy(path: Path, data: np.ndarray) -> None:
    """Write PSD data to an NPY file."""
    np.save(path, data)


def _write_txt(path: Path, data: np.ndarray) -> None:
    """Write PSD data to a TXT file."""
    np.savetxt(path, data)


def _write_csv(path: Path, data: np.ndarray) -> None:
    """Write PSD data to a CSV file."""
    np.savetxt(path, data, delimiter=",")


@pytest.mark.parametrize(
    ("suffix", "writer"),
    [
        (".npy", _write_npy),
        (".txt", _write_txt),
        (".csv", _write_csv),
    ],
)
def test_colored_simulator_loads_supported_psd_formats(
    tmp_path: Path,
    suffix: str,
    writer: Callable[[Path, np.ndarray], None],
) -> None:
    """ColoredNoiseSimulator loads PSD data from supported file types."""
    psd_path = tmp_path / f"psd{suffix}"
    psd_data = np.column_stack((np.linspace(0.0, 128.0, 129), np.full(129, 1.5e-3)))
    writer(psd_path, psd_data)

    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=11,
    )

    realization = simulator.generate(
        duration=2.0,
        sampling_frequency=256.0,
        detectors=["H1"],
    )

    assert realization["H1"].shape == (512,)


def test_generated_psd_matches_input_psd_within_tolerance(tmp_path: Path) -> None:
    """Averaged realizations preserve the input PSD shape."""
    psd_path = _write_psd_file(tmp_path / "flat_psd.txt")
    sampling_frequency = 256.0
    estimated_psds = []

    for seed in range(24):
        simulator = ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=sampling_frequency,
            seed=seed,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )
        realization = simulator.generate(
            duration=8.0,
            sampling_frequency=sampling_frequency,
            detectors=["H1"],
        )["H1"]
        _, estimated_psd = _estimate_one_sided_psd(realization, sampling_frequency)
        estimated_psds.append(estimated_psd)

    mean_psd = np.mean(np.stack(estimated_psds), axis=0)
    frequencies = np.fft.rfftfreq(realization.size, d=1.0 / sampling_frequency)
    band = (frequencies >= 12.0) & (frequencies <= 80.0)

    assert np.median(mean_psd[band]) == pytest.approx(2.0e-3, rel=0.3)


def test_consecutive_generate_calls_are_continuous(tmp_path: Path) -> None:
    """Overlap-add stitching avoids a boundary jump between calls."""
    psd_path = _write_psd_file(tmp_path / "continuity_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=1234,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1"])["H1"]
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1"])["H1"]
    combined = np.concatenate((first, second))
    jumps = np.abs(np.diff(combined))
    boundary_jump = jumps[first.size - 1]

    assert boundary_jump <= np.quantile(jumps, 0.995)


def test_generate_is_deterministic_after_reset(tmp_path: Path) -> None:
    """Resetting and reusing the same seed reproduces the same realization."""
    psd_path = _write_psd_file(tmp_path / "deterministic_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1", "L1"], seed=99)
    simulator.reset()
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1", "L1"], seed=99)

    np.testing.assert_allclose(first["H1"], second["H1"])
    np.testing.assert_allclose(first["L1"], second["L1"])


def test_default_simulator_uses_colored_noise_when_psd_is_configured(tmp_path: Path) -> None:
    """DefaultNoiseSimulator dispatches to ColoredNoiseSimulator when configured."""
    psd_path = _write_psd_file(tmp_path / "dispatch_psd.txt")
    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="colored"),
        seed=42,
        psd_file=psd_path,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "colored_H1.json").read_text())
    assert metadata["implementation"] == "colored"
    assert metadata["colored_noise"]["psd_file"] == str(psd_path)
    assert metadata["colored_noise"]["low_frequency_cutoff"] == 8.0
    assert metadata["colored_noise"]["high_frequency_cutoff"] == 96.0
