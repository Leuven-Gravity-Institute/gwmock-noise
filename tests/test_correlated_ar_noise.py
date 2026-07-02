"""Tests for the correlated autoregressive / VMA noise simulator."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.simulators import CorrelatedARNoiseSimulator, correlated_ar
from gwmock_noise.simulators.colored import PSD_WINDOW_WIDTH_HZ
from gwmock_noise.simulators.correlated import parse_csd_file_map

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
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Create PSD and CSD files for a detector network."""
    psd_files = {detector: _write_psd_file(tmp_path / f"{detector}_psd.txt", value=psd_value) for detector in detectors}
    csd_files = {
        f"{pair[0]}-{pair[1]}": _write_csd_file(tmp_path / f"{pair[0]}_{pair[1]}_csd.npy", value=csd_value)
        for pair in combinations(sorted(detectors), 2)
    }
    return psd_files, csd_files


def test_correlated_ar_simulator_is_importable_from_top_level_package() -> None:
    """CorrelatedARNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.CorrelatedARNoiseSimulator is CorrelatedARNoiseSimulator


def test_generated_psd_and_csd_match_inputs_within_tolerance(tmp_path: Path) -> None:
    """Averaged realizations preserve both PSD and CSD levels."""
    detectors = ["H1", "L1"]
    sampling_frequency = 256.0
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    estimated_psds = []
    estimated_csds = []

    for seed in range(24):
        simulator = CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files=csd_files,
            detectors=detectors,
            sampling_frequency=sampling_frequency,
            seed=seed,
            order=64,
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
    assert np.median(np.abs(mean_csd.imag[band])) < 0.25 * FLAT_CSD


@pytest.mark.parametrize("detectors", [["H1"], ["H1", "L1"], ["H1", "L1", "V1"]])
def test_correlated_ar_simulator_supports_one_two_and_three_detectors(
    tmp_path: Path,
    detectors: list[str],
) -> None:
    """CorrelatedARNoiseSimulator supports 1-3 detectors without code changes."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=123,
        order=32,
    )

    realization = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)

    assert set(realization) == set(detectors)
    assert all(strain.shape == (1024,) for strain in realization.values())


def test_consecutive_generate_calls_are_continuous_across_detectors(tmp_path: Path) -> None:
    """Persistent innovation state avoids detector boundary jumps."""
    detectors = ["H1", "L1", "V1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=1234,
        order=64,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)

    for detector in detectors:
        combined = np.concatenate((first[detector], second[detector]))
        jumps = np.abs(np.diff(combined))
        boundary_jump = jumps[first[detector].size - 1]
        assert boundary_jump <= np.quantile(jumps, 0.995)


def test_generate_is_deterministic_after_reset(tmp_path: Path) -> None:
    """Resetting and reusing the same seed reproduces the same realization."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        order=64,
    )

    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors, seed=99)
    simulator.reset()
    second = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors, seed=99)

    np.testing.assert_allclose(first["H1"], second["H1"])
    np.testing.assert_allclose(first["L1"], second["L1"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"detectors": []}, "detectors must contain at least one detector"),
        ({"detectors": ["H1", "H1"]}, "detectors must not contain duplicates"),
        ({"order": -1}, "order must be non-negative"),
        ({"block_size": 0}, "block_size must be greater than zero"),
        ({"low_frequency_cutoff": -1.0}, "low_frequency_cutoff must be non-negative"),
        ({"regularization_epsilon": 0.0}, "regularization_epsilon must be greater than zero"),
        (
            {"low_frequency_cutoff": 20.0, "high_frequency_cutoff": 20.0},
            "high_frequency_cutoff must be greater than low_frequency_cutoff",
        ),
        ({"high_frequency_cutoff": 200.0}, "high_frequency_cutoff must not exceed the Nyquist frequency"),
    ],
)
def test_correlated_ar_simulator_validates_runtime_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """CorrelatedARNoiseSimulator rejects invalid runtime and cutoff parameters."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    init_kwargs: dict[str, object] = {
        "psd_files": psd_files,
        "csd_files": csd_files,
        "detectors": ["H1", "L1"],
        "sampling_frequency": 256.0,
        "duration": 4.0,
    }
    init_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        CorrelatedARNoiseSimulator(**init_kwargs)


def test_correlated_ar_simulator_rejects_psd_key_mismatch(tmp_path: Path) -> None:
    """PSD key set must match configured detectors exactly."""
    psd_files = {"H1": _write_psd_file(tmp_path / "h1_psd.txt")}
    with pytest.raises(ValueError, match="psd_files keys must exactly match detectors"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )


def test_correlated_ar_simulator_accepts_tuple_key_csd_mapping(tmp_path: Path) -> None:
    """Tuple-key CSD maps are accepted after normalization."""
    detectors = ["H1", "L1"]
    psd_files, _ = _build_spectral_inputs(tmp_path, detectors)
    tuple_csd_files = {("H1", "L1"): _write_csd_file(tmp_path / "pair.npy")}
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=tuple_csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
    )

    assert simulator.metadata["correlated_autoregressive_noise"]["csd_files"] == {
        "H1-L1": str(tuple_csd_files[("H1", "L1")])
    }


def test_correlated_ar_simulator_rejects_invalid_csd_tuple_keys(tmp_path: Path) -> None:
    """Tuple-key CSD validation rejects malformed detector pairs."""
    psd_files, _ = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    csd_path = _write_csd_file(tmp_path / "pair.npy")

    with pytest.raises(ValueError, match="exactly two detector names"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1",): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="two distinct detectors"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "H1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="reference configured detectors"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "V1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="Duplicate CSD file mapping"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "L1"): csd_path, ("L1", "H1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )


def test_correlated_ar_simulator_reuses_parse_csd_file_map_contract(tmp_path: Path) -> None:
    """String-key CSD mappings stay compatible with the shared parser."""
    csd_mapping = {"H1-L1": tmp_path / "pair.npy"}
    assert parse_csd_file_map(csd_mapping) == {("H1", "L1"): tmp_path / "pair.npy"}


def test_correlated_ar_simulator_rejects_empty_frequency_band(tmp_path: Path) -> None:
    """Cutoff band with no bins is rejected."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    with pytest.raises(ValueError, match="contains no simulation bins"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files=csd_files,
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
            low_frequency_cutoff=0.01,
            high_frequency_cutoff=0.02,
        )


def test_generate_block_requires_initialized_rng(tmp_path: Path) -> None:
    """Internal block generation requires RNG initialization."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
    )
    simulator._rng = None
    with pytest.raises(RuntimeError, match="not initialized"):
        simulator._generate_block(16)


def test_generate_reconfigures_when_runtime_changes(tmp_path: Path) -> None:
    """Runtime changes trigger filter refit."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=7,
        order=32,
    )
    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)
    second = simulator.generate(duration=4.0, sampling_frequency=512.0, detectors=detectors)
    assert first["H1"].shape == (1024,)
    assert second["H1"].shape == (2048,)


def test_generate_rejects_detector_subset_at_runtime(tmp_path: Path) -> None:
    """Subset detector requests are rejected with a clear error."""
    detectors = ["H1", "L1", "V1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        order=16,
    )
    with pytest.raises(ValueError, match="subset or superset"):
        simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1", "L1"])


def test_generate_rejects_detector_superset_at_runtime(tmp_path: Path) -> None:
    """Superset detector requests are rejected with a clear error."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        order=16,
    )
    with pytest.raises(ValueError, match="subset or superset"):
        simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1", "L1", "V1"])


def test_generate_accepts_detector_reorder_only(tmp_path: Path) -> None:
    """Same detector set in a different order still runs (refit path)."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        order=16,
    )
    out = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["L1", "H1"])
    assert set(out) == {"H1", "L1"}
    assert out["H1"].shape == (512,)


def test_correlated_ar_rejects_csd_pair_outside_detector_set_after_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normalization re-checks parsed CSD keys against configured detectors."""
    psd_files, _ = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    fake_csd = {("H1", "V1"): tmp_path / "h1_v1.npy"}

    monkeypatch.setattr("gwmock_noise.simulators.correlated_ar.parse_csd_file_map", lambda _: fake_csd)
    with pytest.raises(ValueError, match="configured detectors"):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files={"H1-L1": tmp_path / "h1_l1.npy"},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )


def test_correlated_ar_rejects_too_few_spectral_points(tmp_path: Path) -> None:
    """Model fitting requires at least two spectral samples."""
    psd_path = tmp_path / "single_point_psd.npy"
    np.save(psd_path, np.array([[0.0, 1.0e-3]], dtype=float))
    with pytest.raises(ValueError, match="at least two frequency samples"):
        CorrelatedARNoiseSimulator(
            psd_files={"H1": psd_path},
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_correlated_ar_order_zero_updates_empty_state(tmp_path: Path) -> None:
    """Order-zero fit keeps an empty continuity state after generation."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
        order=0,
        seed=4,
    )
    out = simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=["H1", "L1"])
    assert out["H1"].shape == (512,)
    assert simulator._state.shape == (0, 2)


def test_correlated_ar_rejects_tiny_duration_rounding_to_zero_samples(tmp_path: Path) -> None:
    """Generate rejects requests that round to zero samples."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        order=16,
    )
    with pytest.raises(ValueError, match="must produce at least one sample"):
        simulator.generate(duration=1.0e-6, sampling_frequency=256.0, detectors=detectors)


def test_regularized_cholesky_falls_back_on_factorization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinAlgError path falls back to diagonal square-root factor."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedARNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
    )

    def _raise_linalg_error(_: np.ndarray) -> np.ndarray:
        raise np.linalg.LinAlgError("forced")

    monkeypatch.setattr(np.linalg, "cholesky", _raise_linalg_error)
    matrix = np.array([[1.0, 0.2], [0.2, 0.5]], dtype=np.complex128)
    factor = simulator._regularized_cholesky(matrix)

    assert factor.shape == (2, 2)
    assert np.allclose(factor, np.diag(np.diag(factor)))


def test_fit_model_uses_absolute_width_taper_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The VMA fit's PSD/CSD taper alpha scales with bandwidth to keep a fixed Hz-wide edge."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    captured_alphas: list[float] = []
    original_tukey_window = correlated_ar._tukey_window

    def _spy_tukey_window(length: int, alpha: float = 5e-3) -> np.ndarray:
        captured_alphas.append(alpha)
        return original_tukey_window(length, alpha=alpha)

    monkeypatch.setattr(correlated_ar, "_tukey_window", _spy_tukey_window)

    for low, high in ((8.0, 96.0), (8.0, 48.0)):
        CorrelatedARNoiseSimulator(
            psd_files=psd_files,
            csd_files=csd_files,
            detectors=detectors,
            sampling_frequency=256.0,
            order=8,
            low_frequency_cutoff=low,
            high_frequency_cutoff=high,
        )

    assert captured_alphas == pytest.approx(
        [2.0 * PSD_WINDOW_WIDTH_HZ / (96.0 - 8.0), 2.0 * PSD_WINDOW_WIDTH_HZ / (48.0 - 8.0)]
    )
