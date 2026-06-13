"""Regression tests for the seconds-based stitching window (issue #196).

The synthesis window is expressed as a fixed duration in seconds so the frequency
resolution ``df = 1 / window_duration`` is invariant to the sampling frequency.
Previously the window was hard coded to 2048 samples, making ``df = fs / 2048``
(8 Hz at fs = 16384), which could not capture fast-varying PSD structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from gwmock_noise.simulators import ColoredNoiseSimulator, CorrelatedNoiseSimulator


def _write_fast_psd(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Write the issue's fast-varying PSD (≈5 Hz wiggle) and return its arrays."""
    frequencies = np.linspace(2.0, 22.0, 2000)
    values = 1.0e-34 * (3.0 + np.cos(2.0 * np.pi * frequencies / 5.0))
    np.save(path, np.column_stack([frequencies, values]))
    return frequencies, values


@pytest.mark.parametrize("sampling_frequency", [64.0, 4096.0, 16384.0])
def test_delta_frequency_is_invariant_to_sampling_frequency(tmp_path: Path, sampling_frequency: float) -> None:
    """Df depends only on window_duration, not on the sampling frequency."""
    _write_fast_psd(tmp_path / "psd.npy")
    window_duration = 4.0
    simulator = ColoredNoiseSimulator(
        psd_file=tmp_path / "psd.npy",
        detectors=["H1"],
        sampling_frequency=sampling_frequency,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=22.0,
        window_duration=window_duration,
        seed=0,
    )
    assert simulator._window_size == round(window_duration * sampling_frequency)
    assert simulator._delta_frequency == pytest.approx(1.0 / window_duration)


def test_runtime_sampling_frequency_change_rescales_window(tmp_path: Path) -> None:
    """Changing fs at generate() time keeps df fixed by rescaling the window."""
    _write_fast_psd(tmp_path / "psd.npy")
    simulator = ColoredNoiseSimulator(
        psd_file=tmp_path / "psd.npy",
        detectors=["H1"],
        sampling_frequency=256.0,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=22.0,
        window_duration=4.0,
        seed=0,
    )
    simulator.generate(duration=4.0, sampling_frequency=1024.0, detectors=["H1"])
    assert simulator._window_size == round(4.0 * 1024.0)
    assert simulator._delta_frequency == pytest.approx(0.25)


def test_fast_psd_is_recovered_at_high_sampling_frequency(tmp_path: Path) -> None:
    """The fast-varying PSD is recovered at fs=16384 with the default window."""
    sampling_frequency = 16384.0
    frequencies, values = _write_fast_psd(tmp_path / "psd.npy")
    simulator = CorrelatedNoiseSimulator(
        psd_files={"D1": tmp_path / "psd.npy"},
        detectors=["D1"],
        sampling_frequency=sampling_frequency,
        low_frequency_cutoff=float(frequencies.min()),
        seed=42,
    )
    data = simulator.generate(duration=128.0, sampling_frequency=sampling_frequency, detectors=["D1"])["D1"]

    nperseg = int(16 * sampling_frequency)
    estimate_frequencies, estimate_psd = welch(data, fs=sampling_frequency, nperseg=nperseg, window="hann")
    band = (estimate_frequencies >= 4.0) & (estimate_frequencies <= 20.0)
    target = np.interp(estimate_frequencies[band], frequencies, values)
    fractional_error = np.abs(np.sqrt(estimate_psd[band]) - np.sqrt(target)) / np.sqrt(target)
    # The 5 Hz wiggle is resolved (df = 0.25 Hz); previously df = 8 Hz washed it out.
    assert np.median(fractional_error) < 0.15


def test_warns_when_window_is_too_coarse(tmp_path: Path) -> None:
    """A short window relative to the input PSD spacing triggers a warning (#139)."""
    _write_fast_psd(tmp_path / "psd.npy")

    # Capture directly on the module logger so the assertion is immune to global
    # logging state (e.g. the CLI sets ``propagate=False`` on the package logger).
    records: list[logging.LogRecord] = []
    capture = logging.Handler()
    capture.emit = records.append  # type: ignore[method-assign]
    module_logger = logging.getLogger("gwmock-noise")
    previous_level = module_logger.level
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    module_logger.setLevel(logging.WARNING)
    module_logger.addHandler(capture)
    try:
        ColoredNoiseSimulator(
            psd_file=tmp_path / "psd.npy",
            detectors=["H1"],
            sampling_frequency=256.0,
            low_frequency_cutoff=2.0,
            high_frequency_cutoff=22.0,
            window_duration=0.25,  # df = 4 Hz, far coarser than the input spacing
            seed=0,
        )
    finally:
        module_logger.removeHandler(capture)
        module_logger.setLevel(previous_level)
        logging.disable(previous_disable)

    assert any("frequency resolution" in record.getMessage() for record in records)


def test_metadata_reports_window_duration(tmp_path: Path) -> None:
    """Metadata exposes the resolved window duration and sample sizes."""
    _write_fast_psd(tmp_path / "psd.npy")
    simulator = ColoredNoiseSimulator(
        psd_file=tmp_path / "psd.npy",
        detectors=["H1"],
        sampling_frequency=256.0,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=22.0,
        window_duration=4.0,
        seed=0,
    )
    colored_metadata = simulator.metadata["colored_noise"]
    assert colored_metadata["window_duration"] == 4.0
    assert colored_metadata["window_size"] == round(4.0 * 256.0)
    assert colored_metadata["overlap_size"] == colored_metadata["window_size"] // 2
