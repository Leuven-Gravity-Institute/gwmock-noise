"""Tests for the colored-noise simulator."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import TimeVaryingColoredNoiseSimulator as PackageTimeVaryingColoredNoiseSimulator
from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators import (
    ColoredNoiseSimulator,
    DefaultNoiseSimulator,
    TimeVaryingColoredNoiseSimulator,
)
from gwmock_noise.simulators.colored import WINDOW_SIZE, _tukey_window


def _write_psd_file(path: Path, *, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the full detector band."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
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


def test_time_varying_generated_psd_matches_start_and_end_anchors(tmp_path: Path) -> None:
    """Long realizations recover the first and last PSD anchors near the ends."""
    start_psd_path = _write_psd_file(tmp_path / "start_psd.txt", value=2.0e-3)
    end_psd_path = _write_psd_file(tmp_path / "end_psd.txt", value=8.0e-3)
    sampling_frequency = 256.0
    start_band_medians = []
    end_band_medians = []

    for seed in range(20):
        simulator = ColoredNoiseSimulator(
            psd_schedule=[(0.0, start_psd_path), (64.0, end_psd_path)],
            detectors=["H1"],
            sampling_frequency=sampling_frequency,
            seed=seed,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )
        realization = simulator.generate(
            duration=96.0,
            sampling_frequency=sampling_frequency,
            detectors=["H1"],
        )["H1"]
        start_frequencies, start_psd = _estimate_one_sided_psd(realization[:WINDOW_SIZE], sampling_frequency)
        # Measure the final fully blended window rather than the tapered tail of the last chunk.
        end_window = realization[-(WINDOW_SIZE + (WINDOW_SIZE // 2)) : -(WINDOW_SIZE // 2)]
        end_frequencies, end_psd = _estimate_one_sided_psd(end_window, sampling_frequency)
        start_band = (start_frequencies >= 12.0) & (start_frequencies <= 80.0)
        end_band = (end_frequencies >= 12.0) & (end_frequencies <= 80.0)
        start_band_medians.append(float(np.median(start_psd[start_band])))
        end_band_medians.append(float(np.median(end_psd[end_band])))

    assert np.mean(start_band_medians) == pytest.approx(2.0e-3, rel=0.35)
    assert np.mean(end_band_medians) == pytest.approx(8.0e-3, rel=0.35)


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


def test_time_varying_consecutive_generate_calls_are_continuous(tmp_path: Path) -> None:
    """Time-varying PSD updates preserve overlap-add continuity across calls."""
    start_psd_path = _write_psd_file(tmp_path / "tv_start_psd.txt", value=2.0e-3)
    end_psd_path = _write_psd_file(tmp_path / "tv_end_psd.txt", value=6.0e-3)
    simulator = ColoredNoiseSimulator(
        psd_schedule=[(0.0, start_psd_path), (64.0, end_psd_path)],
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


def test_single_anchor_schedule_matches_stationary_behavior(tmp_path: Path) -> None:
    """A single PSD anchor follows the stationary colored-noise codepath."""
    psd_path = _write_psd_file(tmp_path / "single_anchor_psd.txt")
    stationary = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=99,
    )
    time_varying = ColoredNoiseSimulator(
        psd_schedule=[(0.0, psd_path)],
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=99,
    )

    stationary_realization = stationary.generate(duration=8.0, sampling_frequency=256.0, detectors=["H1"])
    time_varying_realization = time_varying.generate(duration=8.0, sampling_frequency=256.0, detectors=["H1"])

    np.testing.assert_allclose(stationary_realization["H1"], time_varying_realization["H1"])


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


def test_default_simulator_uses_colored_noise_when_psd_schedule_is_configured(tmp_path: Path) -> None:
    """DefaultNoiseSimulator dispatches to colored noise for a PSD schedule."""
    start_psd_path = _write_psd_file(tmp_path / "schedule_start_psd.txt", value=2.0e-3)
    end_psd_path = _write_psd_file(tmp_path / "schedule_end_psd.txt", value=6.0e-3)
    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=4.0,
        sampling_frequency=256.0,
        output=OutputConfig(directory=out_dir, prefix="colored_schedule"),
        seed=42,
        psd_schedule=[(0.0, start_psd_path), (64.0, end_psd_path)],
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
    )

    simulator = DefaultNoiseSimulator()
    simulator.run(config)

    metadata = json.loads((out_dir / "colored_schedule_H1.json").read_text())
    assert metadata["implementation"] == "colored"
    assert metadata["colored_noise"]["psd_file"] is None
    assert metadata["colored_noise"]["psd_schedule"] == [
        {"gps_offset_seconds": 0.0, "psd_file": str(start_psd_path)},
        {"gps_offset_seconds": 64.0, "psd_file": str(end_psd_path)},
    ]


def test_time_varying_alias_reuses_colored_simulator() -> None:
    """The time-varying convenience alias points at the shared implementation."""
    assert TimeVaryingColoredNoiseSimulator is ColoredNoiseSimulator
    assert PackageTimeVaryingColoredNoiseSimulator is ColoredNoiseSimulator


def test_tukey_window_validates_positive_length() -> None:
    """Tukey helper rejects non-positive lengths."""
    with pytest.raises(ValueError, match="length must be positive"):
        _tukey_window(0)


def test_tukey_window_returns_ones_when_alpha_non_positive() -> None:
    """Alpha <= 0 disables tapering."""
    np.testing.assert_allclose(_tukey_window(8, alpha=0.0), np.ones(8))


def test_tukey_window_matches_hanning_when_alpha_ge_one() -> None:
    """Alpha >= 1 follows the Hann branch."""
    np.testing.assert_allclose(_tukey_window(8, alpha=1.0), np.hanning(8))


def test_tukey_window_skips_taper_for_tiny_windows() -> None:
    """Tiny windows return all ones to avoid collapsing narrow masks."""
    np.testing.assert_allclose(_tukey_window(2, alpha=0.5), np.ones(2))
    np.testing.assert_allclose(_tukey_window(3, alpha=0.5), np.ones(3))


def test_tukey_window_applies_taper_for_regular_windows() -> None:
    """Regular-size windows still use tapering."""
    window = _tukey_window(64, alpha=0.5)
    assert window.shape == (64,)
    assert window.dtype == float
    assert window[0] == pytest.approx(0.0)
    assert window[32] == pytest.approx(1.0)


def test_generate_rejects_invalid_previous_strain_shape(tmp_path: Path) -> None:
    """Generate validates continuity buffer shapes."""
    psd_path = _write_psd_file(tmp_path / "shape_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=7,
    )
    simulator.previous_strain["H1"] = np.zeros(WINDOW_SIZE - 1)

    with pytest.raises(ValueError, match="must have shape"):
        simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1"])


def test_generate_reconfigures_when_runtime_sampling_frequency_changes(tmp_path: Path) -> None:
    """Changing runtime sampling_frequency reconfigures and resets continuity state."""
    psd_path = _write_psd_file(tmp_path / "runtime_change_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=21,
    )

    simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1"])
    assert "H1" in simulator.previous_strain

    second = simulator.generate(duration=4.0, sampling_frequency=512.0, detectors=["H1"])
    assert second["H1"].shape == (2048,)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"detectors": []}, "detectors must contain at least one detector"),
        ({"low_frequency_cutoff": -1.0}, "low_frequency_cutoff must be non-negative"),
        (
            {"low_frequency_cutoff": 20.0, "high_frequency_cutoff": 20.0},
            "high_frequency_cutoff must be greater than low_frequency_cutoff",
        ),
        (
            {"high_frequency_cutoff": 200.0},
            "high_frequency_cutoff must not exceed the Nyquist frequency",
        ),
    ],
)
def test_colored_simulator_validates_runtime_arguments(
    tmp_path: Path,
    kwargs: dict[str, float | list[str]],
    message: str,
) -> None:
    """ColoredNoiseSimulator rejects invalid runtime and cutoff arguments."""
    psd_path = _write_psd_file(tmp_path / "invalid_runtime_psd.txt")
    base_kwargs: dict[str, object] = {
        "psd_file": psd_path,
        "detectors": ["H1"],
        "sampling_frequency": 256.0,
        "duration": 4.0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ColoredNoiseSimulator(**base_kwargs)


def test_colored_simulator_rejects_empty_psd_schedule() -> None:
    """Empty time-varying schedules are rejected."""
    with pytest.raises(ValueError, match="must contain at least one anchor"):
        ColoredNoiseSimulator(psd_schedule=[], detectors=["H1"], sampling_frequency=256.0)


def test_colored_simulator_requires_psd_source() -> None:
    """Either psd_file or psd_schedule must be configured."""
    with pytest.raises(ValueError, match="Either psd_file or psd_schedule must be provided"):
        ColoredNoiseSimulator(detectors=["H1"], sampling_frequency=256.0)


def test_colored_simulator_rejects_mixed_psd_file_and_schedule(tmp_path: Path) -> None:
    """Single PSD and schedule inputs are mutually exclusive."""
    psd_path = _write_psd_file(tmp_path / "mixed_psd.txt")
    with pytest.raises(ValueError, match="mutually exclusive"):
        ColoredNoiseSimulator(
            psd_file=psd_path, psd_schedule=[(0.0, psd_path)], detectors=["H1"], sampling_frequency=256.0
        )


def test_colored_simulator_rejects_duplicate_schedule_offsets(tmp_path: Path) -> None:
    """Schedule offsets must be distinct."""
    psd_a = _write_psd_file(tmp_path / "a_psd.txt")
    psd_b = _write_psd_file(tmp_path / "b_psd.txt")
    with pytest.raises(ValueError, match="must use distinct GPS offsets"):
        ColoredNoiseSimulator(psd_schedule=[(0.0, psd_a), (0.0, psd_b)], detectors=["H1"], sampling_frequency=256.0)


def test_colored_simulator_rejects_unsorted_schedule_offsets(tmp_path: Path) -> None:
    """Schedule offsets must be sorted."""
    psd_a = _write_psd_file(tmp_path / "a_psd_unsorted.txt")
    psd_b = _write_psd_file(tmp_path / "b_psd_unsorted.txt")
    with pytest.raises(ValueError, match="must be sorted by GPS offset"):
        ColoredNoiseSimulator(psd_schedule=[(1.0, psd_a), (0.0, psd_b)], detectors=["H1"], sampling_frequency=256.0)


def test_colored_reset_handles_empty_psd_anchor_cache(tmp_path: Path) -> None:
    """Reset tolerates an empty anchor cache without touching _psd."""
    psd_path = _write_psd_file(tmp_path / "reset_psd.txt")
    simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, seed=2)
    simulator._psd_anchors = []
    simulator.reset()
    assert simulator._generated_samples == 0


def test_colored_simulator_rejects_empty_frequency_mask(tmp_path: Path) -> None:
    """Initialization fails when cutoff band contains no FFT bins."""
    psd_path = _write_psd_file(tmp_path / "empty_mask_psd.txt")
    with pytest.raises(ValueError, match="contains no simulation bins"):
        ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
            low_frequency_cutoff=0.01,
            high_frequency_cutoff=0.02,
        )


def test_colored_simulator_rejects_missing_psd_file(tmp_path: Path) -> None:
    """Initialization fails when the PSD file does not exist."""
    with pytest.raises(FileNotFoundError, match="PSD file not found"):
        ColoredNoiseSimulator(
            psd_file=tmp_path / "missing_psd.txt",
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_colored_simulator_rejects_unsupported_psd_suffix(tmp_path: Path) -> None:
    """Initialization fails for unsupported PSD file formats."""
    psd_path = tmp_path / "psd.dat"
    np.savetxt(psd_path, np.column_stack((np.array([0.0, 1.0]), np.array([1.0, 1.0]))))
    with pytest.raises(ValueError, match="Unsupported PSD file format"):
        ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_colored_simulator_rejects_wrong_psd_shape(tmp_path: Path) -> None:
    """Initialization fails when PSD data are not shape (N, 2)."""
    psd_path = tmp_path / "bad_shape.npy"
    np.save(psd_path, np.ones((8,)))
    with pytest.raises(ValueError, match="PSD file must have shape"):
        ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
        )
