"""Tests for spectral-line injection simulators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig, SpectralLine
from gwmock_noise.simulators import AddLines, DefaultNoiseSimulator, SpectralLineSimulator


class ZeroNoiseSimulator:
    """Minimal protocol-compatible base simulator for additive tests."""

    def __init__(self) -> None:
        """Initialize zero-valued base-simulator state."""
        self.duration = 4.0
        self.sampling_frequency = 256.0
        self.detectors = ["H1", "L1"]
        self.seed: int | None = None

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return zero strain for each requested detector."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, str]:
        """Return representative metadata for the wrapper tests."""
        return {"implementation": "zero"}


def _peak_frequency_and_config_amplitude(
    strain: np.ndarray,
    *,
    sampling_frequency: float,
    target_frequency: float,
) -> tuple[float, float]:
    """Return the nearest-bin peak frequency and recovered configured amplitude."""
    frequency_series = np.fft.rfft(strain)
    frequencies = np.fft.rfftfreq(strain.size, d=1.0 / sampling_frequency)
    rayleigh = sampling_frequency / strain.size
    band = np.abs(frequencies - target_frequency) <= rayleigh
    peak_index = np.flatnonzero(band)[np.argmax(np.abs(frequency_series[band]))]
    peak_frequency = float(frequencies[peak_index])
    time_amplitude = (2.0 * np.abs(frequency_series[peak_index])) / strain.size
    configured_amplitude = float(time_amplitude / np.sqrt(sampling_frequency / 2.0))
    return peak_frequency, configured_amplitude


def test_spectral_line_simulator_recovers_fft_peaks_and_amplitudes() -> None:
    """Fixed spectral lines appear at the requested frequencies and amplitudes."""
    sampling_frequency = 1024.0
    simulator = SpectralLineSimulator(
        lines=[
            SpectralLine(frequency=32.0, amplitude=3.0e-2, phase=0.1),
            SpectralLine(frequency=96.0, amplitude=1.5e-2, phase=1.2),
        ],
        detectors=["H1"],
        sampling_frequency=sampling_frequency,
    )

    strain = simulator.generate(duration=8.0, sampling_frequency=sampling_frequency, detectors=["H1"])["H1"]

    peak_32_hz = _peak_frequency_and_config_amplitude(
        strain,
        sampling_frequency=sampling_frequency,
        target_frequency=32.0,
    )
    peak_96_hz = _peak_frequency_and_config_amplitude(
        strain,
        sampling_frequency=sampling_frequency,
        target_frequency=96.0,
    )
    rayleigh = sampling_frequency / strain.size

    assert peak_32_hz[0] == pytest.approx(32.0, abs=rayleigh)
    assert peak_32_hz[1] == pytest.approx(3.0e-2, rel=5.0e-2)
    assert peak_96_hz[0] == pytest.approx(96.0, abs=rayleigh)
    assert peak_96_hz[1] == pytest.approx(1.5e-2, rel=5.0e-2)


def test_drifting_spectral_lines_accumulate_phase_across_generate_calls() -> None:
    """Consecutive calls match one continuous drifting-line realization."""
    line = SpectralLine(frequency=24.0, amplitude=2.0e-2, phase=0.25, drift_rate=0.5)
    first_simulator = SpectralLineSimulator(
        lines=[line],
        detectors=["H1"],
        sampling_frequency=512.0,
    )
    second_simulator = SpectralLineSimulator(
        lines=[line],
        detectors=["H1"],
        sampling_frequency=512.0,
    )

    split = np.concatenate(
        [
            first_simulator.generate(duration=4.0, sampling_frequency=512.0, detectors=["H1"])["H1"],
            first_simulator.generate(duration=4.0, sampling_frequency=512.0, detectors=["H1"])["H1"],
        ]
    )
    continuous = second_simulator.generate(duration=8.0, sampling_frequency=512.0, detectors=["H1"])["H1"]

    np.testing.assert_allclose(split, continuous)


def test_add_lines_wraps_base_simulator_output() -> None:
    """AddLines adds the generated line signal on top of the base output."""
    simulator = AddLines(
        ZeroNoiseSimulator(),
        [SpectralLine(frequency=48.0, amplitude=2.5e-2, phase=0.4)],
    )

    realization = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1", "L1"])

    assert realization["H1"].shape == (1024,)
    np.testing.assert_allclose(realization["H1"], realization["L1"])
    assert not np.allclose(realization["H1"], 0.0)


def test_default_simulator_reports_line_only_metadata(tmp_path: Path) -> None:
    """DefaultNoiseSimulator dispatches to SpectralLineSimulator when configured."""
    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="lines"),
        seed=42,
        spectral_lines=[SpectralLine(frequency=60.0, amplitude=1.0e-2, phase=0.0)],
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "lines_H1.json").read_text())
    assert metadata["implementation"] == "spectral_lines"
    assert metadata["spectral_lines"]["lines"] == [
        {"frequency": 60.0, "amplitude": 1.0e-2, "phase": 0.0, "drift_rate": 0.0}
    ]


def test_default_simulator_reports_additive_line_metadata(tmp_path: Path) -> None:
    """DefaultNoiseSimulator wraps colored noise when spectral lines are configured."""
    psd_path = tmp_path / "flat_psd.txt"
    frequencies = np.linspace(0.0, 128.0, 129)
    np.savetxt(psd_path, np.column_stack((frequencies, np.full_like(frequencies, 2.0e-3))))

    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="colored_lines"),
        seed=7,
        psd_file=psd_path,
        spectral_lines=[SpectralLine(frequency=48.0, amplitude=2.0e-2, phase=0.0)],
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "colored_lines_H1.json").read_text())
    assert metadata["implementation"] == "add_lines"
    assert metadata["base_implementation"] == "colored"
    assert metadata["spectral_lines"]["lines"] == [
        {"frequency": 48.0, "amplitude": 2.0e-2, "phase": 0.0, "drift_rate": 0.0}
    ]
