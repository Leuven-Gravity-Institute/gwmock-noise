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
        components=[
            {
                "simulator": "correlated",
                "psd_files": psd_files,
                "csd_files": csd_config,
                "low_frequency_cutoff": 8.0,
                "high_frequency_cutoff": 96.0,
            }
        ],
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

    for seed in range(32):
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


def test_parse_csd_file_map_rejects_malformed_keys(tmp_path: Path) -> None:
    """CSD string-key parser validates malformed keys."""
    with pytest.raises(ValueError, match="DET1-DET2"):
        parse_csd_file_map({"H1": tmp_path / "a.npy"})

    with pytest.raises(ValueError, match="two distinct detectors"):
        parse_csd_file_map({"H1-H1": tmp_path / "a.npy"})


def test_parse_csd_file_map_accepts_empty_mapping() -> None:
    """Empty CSD map is normalized to an empty dict."""
    assert parse_csd_file_map({}) == {}


def test_parse_csd_file_map_rejects_duplicate_normalized_pairs(tmp_path: Path) -> None:
    """CSD parser rejects duplicate keys that normalize to the same pair."""
    with pytest.raises(ValueError, match="Duplicate CSD file mapping"):
        parse_csd_file_map(
            {
                "H1-L1": tmp_path / "a.npy",
                "L1-H1": tmp_path / "b.npy",
            }
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"detectors": []}, "detectors must contain at least one detector"),
        ({"detectors": ["H1", "H1"]}, "detectors must not contain duplicates"),
        ({"low_frequency_cutoff": -1.0}, "low_frequency_cutoff must be non-negative"),
        ({"regularization_epsilon": 0.0}, "regularization_epsilon must be greater than zero"),
        (
            {"low_frequency_cutoff": 20.0, "high_frequency_cutoff": 20.0},
            "high_frequency_cutoff must be greater than low_frequency_cutoff",
        ),
        ({"high_frequency_cutoff": 200.0}, "high_frequency_cutoff must not exceed the Nyquist frequency"),
    ],
)
def test_correlated_simulator_validates_runtime_arguments(
    tmp_path: Path,
    kwargs: dict[str, object],
    message: str,
) -> None:
    """CorrelatedNoiseSimulator rejects invalid runtime/cutoff parameters."""
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
        CorrelatedNoiseSimulator(**init_kwargs)


def test_correlated_simulator_rejects_psd_key_mismatch(tmp_path: Path) -> None:
    """PSD key set must match configured detectors exactly."""
    psd_files = {"H1": _write_psd_file(tmp_path / "h1_psd.txt")}
    with pytest.raises(ValueError, match="psd_files keys must exactly match detectors"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )


def test_correlated_simulator_rejects_invalid_csd_tuple_keys(tmp_path: Path) -> None:
    """Tuple-key CSD validation rejects malformed detector pairs."""
    psd_files, _ = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    csd_path = _write_csd_file(tmp_path / "pair.npy")

    with pytest.raises(ValueError, match="exactly two detector names"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1",): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="two distinct detectors"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "H1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="reference configured detectors"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "V1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    with pytest.raises(ValueError, match="Duplicate CSD file mapping"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files={("H1", "L1"): csd_path, ("L1", "H1"): csd_path},
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )


def test_correlated_simulator_rejects_empty_frequency_band(tmp_path: Path) -> None:
    """Cutoff band with no bins is rejected."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    with pytest.raises(ValueError, match="contains no simulation bins"):
        CorrelatedNoiseSimulator(
            psd_files=psd_files,
            csd_files=csd_files,
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
            low_frequency_cutoff=0.01,
            high_frequency_cutoff=0.02,
        )


def test_generate_realization_chunk_requires_initialized_rng(tmp_path: Path) -> None:
    """Internal chunk generation requires RNG initialization."""
    psd_files, csd_files = _build_spectral_inputs(tmp_path, ["H1", "L1"])
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
    )
    simulator._rng = None
    with pytest.raises(RuntimeError, match="not initialized"):
        simulator._generate_realization_chunk()


def test_generate_reconfigures_when_runtime_changes(tmp_path: Path) -> None:
    """Runtime changes trigger spectral reconfiguration path."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=7,
    )
    first = simulator.generate(duration=4.0, sampling_frequency=256.0, detectors=detectors)
    second = simulator.generate(duration=4.0, sampling_frequency=512.0, detectors=detectors)
    assert first["H1"].shape == (1024,)
    assert second["H1"].shape == (2048,)


def test_previous_strain_property_exposes_stitcher_buffers(tmp_path: Path) -> None:
    """previous_strain property proxies through to the stitcher state."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        seed=3,
    )
    simulator.generate(duration=2.0, sampling_frequency=256.0, detectors=detectors)
    assert set(simulator.previous_strain) == set(detectors)


def test_regularized_cholesky_falls_back_on_factorization_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LinAlgError path falls back to diagonal square-root factor."""
    detectors = ["H1", "L1"]
    psd_files, csd_files = _build_spectral_inputs(tmp_path, detectors)
    simulator = CorrelatedNoiseSimulator(
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
