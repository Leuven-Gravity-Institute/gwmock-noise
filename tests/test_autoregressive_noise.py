"""Tests for the autoregressive noise simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.simulators import ARNoiseSimulator


def _write_psd_file(path: Path, *, sampling_frequency: float = 256.0, n_points: int = 1025) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, n_points)
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


def test_generated_psd_matches_input_psd_within_tolerance(tmp_path: Path) -> None:
    """Averaged AR realizations preserve the target PSD shape."""
    psd_path = _write_psd_file(tmp_path / "flat_psd.txt")
    sampling_frequency = 256.0
    estimated_psds = []

    for seed in range(24):
        simulator = ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=sampling_frequency,
            seed=seed,
            order=64,
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
    """AR state carries across generate() calls without a visible jump."""
    psd_path = _write_psd_file(tmp_path / "continuity_psd.txt")
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        seed=1234,
        order=64,
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
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
        order=64,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1", "L1"], seed=99)
    simulator.reset()
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=["H1", "L1"], seed=99)

    np.testing.assert_allclose(first["H1"], second["H1"])
    np.testing.assert_allclose(first["L1"], second["L1"])
    assert not np.allclose(first["H1"], first["L1"])


def test_ar_model_fit_completes_under_five_seconds_for_4096_point_psd(tmp_path: Path) -> None:
    """The fitter meets the issue timing requirement for a 4096-point PSD."""
    psd_path = _write_psd_file(
        tmp_path / "fit_timing_psd.txt",
        sampling_frequency=4096.0,
        n_points=4096,
    )
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=4096.0,
        order=256,
    )

    assert simulator.metadata["autoregressive_noise"]["fit_time_seconds"] < 5.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"detectors": []}, "detectors must contain at least one detector"),
        ({"order": 0}, "order must be greater than zero"),
        ({"block_size": 0}, "block_size must be greater than zero"),
        ({"regularization": -1.0}, "regularization must be non-negative"),
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
def test_ar_simulator_validates_runtime_arguments(
    tmp_path: Path,
    kwargs: dict[str, float | int | list[str]],
    message: str,
) -> None:
    """ARNoiseSimulator rejects invalid runtime and cutoff arguments."""
    psd_path = _write_psd_file(tmp_path / "invalid_runtime_psd.txt")
    base_kwargs: dict[str, object] = {
        "psd_file": psd_path,
        "detectors": ["H1"],
        "sampling_frequency": 256.0,
        "duration": 4.0,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        ARNoiseSimulator(**base_kwargs)


def test_ar_simulator_rejects_missing_psd_file(tmp_path: Path) -> None:
    """Initialization fails when the PSD file does not exist."""
    with pytest.raises(FileNotFoundError, match="PSD file not found"):
        ARNoiseSimulator(
            psd_file=tmp_path / "missing_psd.txt",
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_ar_simulator_rejects_unsupported_psd_suffix(tmp_path: Path) -> None:
    """Initialization fails for unsupported PSD file formats."""
    psd_path = tmp_path / "psd.dat"
    np.savetxt(psd_path, np.column_stack((np.array([0.0, 1.0]), np.array([1.0, 1.0]))))
    with pytest.raises(ValueError, match="Unsupported PSD file format"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_ar_simulator_rejects_wrong_psd_shape(tmp_path: Path) -> None:
    """Initialization fails when PSD data are not shape (N, 2)."""
    psd_path = tmp_path / "bad_shape.npy"
    np.save(psd_path, np.ones((8,)))
    with pytest.raises(ValueError, match="PSD file must have shape"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_ar_simulator_rejects_duplicate_detectors(tmp_path: Path) -> None:
    """Runtime validation rejects duplicate detector names."""
    psd_path = _write_psd_file(tmp_path / "duplicate_detectors_psd.txt")
    with pytest.raises(ValueError, match="detectors must not contain duplicate names"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1", "H1"],
            sampling_frequency=256.0,
        )


def test_ar_generate_with_order_one_updates_state(tmp_path: Path) -> None:
    """Order=1 follows the no-shift update branch in _generate_block."""
    psd_path = _write_psd_file(tmp_path / "order_one_psd.txt")
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=1,
        seed=1,
    )
    result = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1"])
    assert result["H1"].shape == (512,)
    assert simulator._state["H1"].shape == (1,)


def test_ar_generate_reconfigures_fit_when_sampling_frequency_changes(tmp_path: Path) -> None:
    """Changing runtime sampling frequency triggers refit/reset branch."""
    psd_path = _write_psd_file(tmp_path / "reconfigure_psd.txt", sampling_frequency=512.0)
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=16,
        seed=5,
    )
    first = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1"])
    second = simulator.generate(duration=2.0, sampling_frequency=512.0, detectors=["H1"])
    assert first["H1"].shape == (512,)
    assert second["H1"].shape == (1024,)


def test_ar_generate_uses_block_chunking_for_long_requests(tmp_path: Path) -> None:
    """Long generation requests use block concatenation path."""
    psd_path = _write_psd_file(tmp_path / "chunking_psd.txt")
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=8,
        block_size=64,
        seed=7,
    )
    # 300 > block_size so while/concatenate path is exercised.
    result = simulator.generate(duration=300 / 256.0, sampling_frequency=256.0, detectors=["H1"])
    assert result["H1"].shape == (300,)


def test_ar_simulator_supports_zero_regularization_and_system_seed(tmp_path: Path) -> None:
    """Fit works with regularization=0 and seed=None RNG initialization."""
    psd_path = _write_psd_file(tmp_path / "zero_reg_psd.txt")
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
        order=16,
        regularization=0.0,
        seed=None,
    )
    result = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1", "L1"])
    assert result["H1"].shape == (512,)
    assert result["L1"].shape == (512,)


def test_ar_simulator_rejects_empty_frequency_band_during_fit(tmp_path: Path) -> None:
    """Fit rejects cutoffs that leave no FFT bins."""
    psd_path = _write_psd_file(tmp_path / "empty_band_psd.txt")
    with pytest.raises(ValueError, match="contains no simulation bins"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
            low_frequency_cutoff=0.01,
            high_frequency_cutoff=0.02,
        )


def test_ar_simulator_rejects_zero_variance_target_psd(tmp_path: Path) -> None:
    """Fit rejects PSDs that integrate to zero variance."""
    psd_path = _write_psd_file(tmp_path / "zero_psd.txt")
    data = np.loadtxt(psd_path)
    data[:, 1] = 0.0
    np.savetxt(psd_path, data)
    with pytest.raises(ValueError, match="zero variance"):
        ARNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, order=8)


def test_ar_simulator_rejects_tiny_duration_rounding_to_zero_samples(tmp_path: Path) -> None:
    """Generate rejects requests that round to zero samples."""
    psd_path = _write_psd_file(tmp_path / "tiny_duration_psd.txt")
    simulator = ARNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, order=8)
    with pytest.raises(ValueError, match="must produce at least one sample"):
        simulator.generate(duration=1.0e-6, sampling_frequency=256.0, detectors=["H1"])


def test_ar_simulator_rejects_nonpositive_innovation_variance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit rejects nonpositive innovation variance."""
    psd_path = _write_psd_file(tmp_path / "innovation_psd.txt")
    monkeypatch.setattr(
        "gwmock_noise.simulators.autoregressive.np.linalg.solve",
        lambda a, b: np.array([-2.0]),
    )
    monkeypatch.setattr(
        "gwmock_noise.simulators.autoregressive.np.roots",
        lambda a: np.array([]),
    )
    monkeypatch.setattr(
        "gwmock_noise.simulators.autoregressive.np.dot",
        lambda a, b: -10.0,
    )
    with pytest.raises(ValueError, match="Innovation variance must be positive"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
            order=1,
            low_frequency_cutoff=0.0,
            high_frequency_cutoff=128.0,
        )


def test_ar_simulator_rejects_unstable_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit rejects unstable AR roots."""
    psd_path = _write_psd_file(tmp_path / "unstable_psd.txt")
    monkeypatch.setattr(
        "gwmock_noise.simulators.autoregressive.np.linalg.solve",
        lambda a, b: np.array([0.0]),
    )
    monkeypatch.setattr(
        "gwmock_noise.simulators.autoregressive.np.roots",
        lambda a: np.array([1.1]),
    )
    with pytest.raises(ValueError, match="Fitted AR model is unstable"):
        ARNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, order=1)


def test_ar_generate_reconfigures_state_for_detector_reordering(tmp_path: Path) -> None:
    """Detector reordering triggers reconfigure_state reset path."""
    psd_path = _write_psd_file(tmp_path / "reorder_psd.txt")
    simulator = ARNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
        order=8,
        seed=9,
    )
    first = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1", "L1"])
    second = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["L1", "H1"])
    assert first["H1"].shape == (512,)
    assert second["H1"].shape == (512,)
