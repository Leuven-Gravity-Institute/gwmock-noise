"""Tests for the Schumann-resonance correlated-noise simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.signal import csd

from gwmock_noise.diagnostics.psd import estimate_psd
from gwmock_noise.simulators import NoiseSimulator, SchumannNoiseSimulator, schumann
from gwmock_noise.simulators.colored import PSD_WINDOW_WIDTH_HZ

HANFORD_POSITION = (46.455, -119.408)
LIVINGSTON_POSITION = (30.563, -90.774)
FUNDAMENTAL_FREQUENCY_HZ = 7.83
EXPECTED_HL_COHERENCE = 0.8892289587148472


def _write_coupling_file(path: Path, *, value: float = 1.0) -> Path:
    """Write a flat magnetic coupling transfer function."""
    frequencies = np.linspace(0.0, 128.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _estimate_one_sided_csd(
    strain_a: np.ndarray,
    strain_b: np.ndarray,
    sampling_frequency: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the one-sided cross-spectral density on the Welch grid."""
    frequencies, spectral_density = csd(
        strain_a,
        strain_b,
        fs=sampling_frequency,
        window="hann",
        nperseg=int(16.0 * sampling_frequency),
        noverlap=int(8.0 * sampling_frequency),
        detrend=False,
        return_onesided=True,
        scaling="density",
    )
    return np.asarray(frequencies, dtype=float), np.asarray(spectral_density, dtype=complex)


def _build_simulator(tmp_path: Path, detectors: list[str]) -> SchumannNoiseSimulator:
    """Create a Schumann simulator with flat coupling files."""
    positions = {"H1": HANFORD_POSITION, "L1": LIVINGSTON_POSITION, "V1": (43.63, 10.5)}
    coupling_files = {
        detector: _write_coupling_file(tmp_path / f"{detector}_coupling.txt", value=1.0e-3) for detector in detectors
    }
    return SchumannNoiseSimulator(
        detectors=detectors,
        positions={detector: positions[detector] for detector in detectors},
        coupling_files=coupling_files,
        sampling_frequency=256.0,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=40.0,
        seed=11,
        # 8 s window (df = 0.125 Hz) to resolve the narrow Schumann resonances.
        window_duration=8.0,
    )


def test_schumann_simulator_satisfies_noise_protocol(tmp_path: Path) -> None:
    """SchumannNoiseSimulator satisfies the runtime-checkable protocol."""
    simulator = _build_simulator(tmp_path, ["H1", "L1"])
    assert isinstance(simulator, NoiseSimulator)


def test_output_psd_has_schumann_peaks(tmp_path: Path) -> None:
    """The averaged output PSD peaks near the first four Schumann modes."""
    simulator = _build_simulator(tmp_path, ["H1"])
    estimated_psds = []
    for seed in range(8):
        realization = simulator.generate(
            duration=64.0,
            sampling_frequency=256.0,
            detectors=["H1"],
            seed=seed,
        )["H1"]
        frequencies, estimated_psd = estimate_psd(
            realization,
            sampling_frequency=256.0,
            segment_duration=16.0,
        )
        estimated_psds.append(estimated_psd)

    mean_psd = np.mean(np.stack(estimated_psds), axis=0)

    for expected_frequency in (7.8, 14.0, 20.0, 26.0):
        window = np.abs(frequencies - expected_frequency) <= 2.0
        peak_frequency = frequencies[window][np.argmax(mean_psd[window])]
        assert peak_frequency == pytest.approx(expected_frequency, abs=0.5)


def test_hanford_livingston_coherence_matches_isotropic_reference(tmp_path: Path) -> None:
    """Two-detector coherence matches the isotropic Schumann approximation."""
    simulator = _build_simulator(tmp_path, ["H1", "L1"])
    psds_h1 = []
    psds_l1 = []
    csds = []

    for seed in range(16):
        realization = simulator.generate(
            duration=64.0,
            sampling_frequency=256.0,
            detectors=["H1", "L1"],
            seed=seed,
        )
        _, psd_h1 = estimate_psd(realization["H1"], sampling_frequency=256.0, segment_duration=16.0)
        _, psd_l1 = estimate_psd(realization["L1"], sampling_frequency=256.0, segment_duration=16.0)
        frequencies, csd_estimate = _estimate_one_sided_csd(realization["H1"], realization["L1"], 256.0)
        psds_h1.append(psd_h1)
        psds_l1.append(psd_l1)
        csds.append(csd_estimate)

    mean_psd_h1 = np.mean(np.stack(psds_h1), axis=0)
    mean_psd_l1 = np.mean(np.stack(psds_l1), axis=0)
    mean_csd = np.mean(np.stack(csds), axis=0)
    index = int(np.argmin(np.abs(frequencies - FUNDAMENTAL_FREQUENCY_HZ)))
    measured_coherence = np.abs(mean_csd[index]) / np.sqrt(mean_psd_h1[index] * mean_psd_l1[index])

    assert measured_coherence == pytest.approx(EXPECTED_HL_COHERENCE, rel=0.2)


def test_theoretical_coherence_matches_reference_value(tmp_path: Path) -> None:
    """The built-in isotropic approximation reproduces the expected H1-L1 target."""
    simulator = _build_simulator(tmp_path, ["H1", "L1"])
    assert simulator.theoretical_coherence(FUNDAMENTAL_FREQUENCY_HZ, "H1", "L1") == pytest.approx(
        EXPECTED_HL_COHERENCE,
        rel=1.0e-6,
    )


def test_regularized_cholesky_tracks_a_physically_small_target_at_schumann_scale(tmp_path: Path) -> None:
    """The Cholesky factor must track the target diagonal's own scale, not a fixed absolute floor.

    The existing fixtures' coupling value (``1.0e-3``) sits well above the
    regularization epsilon's implicit scale, so they cannot exercise the
    regularization path at all. A published LIGO magnetic-field-to-strain
    coupling function is of order ``1e-23`` strain/pT (Coughlin et al., Phys.
    Rev. D 104, 122006 (2021)), which puts the Schumann spectral matrix's
    diagonal at Schumann-relevant scales (~1e-46 or below) far below that
    scale, where a ``max(diagonal, 1.0)`` floor turns the regularization
    epsilon into an absolute floor that overwrites the physical scale.
    """
    coupling_value = 1.0e-23
    coupling_path = _write_coupling_file(tmp_path / "H1_coupling.txt", value=coupling_value)

    simulator = SchumannNoiseSimulator(
        detectors=["H1"],
        positions={"H1": HANFORD_POSITION},
        coupling_files={"H1": coupling_path},
        sampling_frequency=256.0,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=40.0,
        seed=1,
        window_duration=8.0,
    )

    expected_diagonal = simulator._schumann_psd * (coupling_value**2) * (0.5 / simulator._delta_frequency)
    observed_diagonal = np.real(np.sum(np.abs(simulator._cholesky_factors[:, 0, :]) ** 2, axis=1))

    # Exclude the taper's zeroed band edges, where the target itself is (correctly) zero.
    band = expected_diagonal > (expected_diagonal.max() * 1.0e-6)
    assert observed_diagonal[band] == pytest.approx(expected_diagonal[band], rel=1.0e-6, abs=0.0)


def test_configure_spectral_factors_uses_absolute_width_taper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The coupling/PSD taper alpha scales with bandwidth to keep a fixed Hz-wide edge."""
    captured_alphas: list[float] = []
    original_tukey_window = schumann._tukey_window

    def _spy_tukey_window(length: int, alpha: float = 5e-3) -> np.ndarray:
        captured_alphas.append(alpha)
        return original_tukey_window(length, alpha=alpha)

    monkeypatch.setattr(schumann, "_tukey_window", _spy_tukey_window)

    positions = {"H1": HANFORD_POSITION, "L1": LIVINGSTON_POSITION}
    coupling_files = {
        detector: _write_coupling_file(tmp_path / f"{detector}_coupling.txt", value=1.0e-3) for detector in positions
    }
    for high_frequency_cutoff in (40.0, 60.0):
        SchumannNoiseSimulator(
            detectors=list(positions),
            positions=positions,
            coupling_files=coupling_files,
            sampling_frequency=256.0,
            low_frequency_cutoff=2.0,
            high_frequency_cutoff=high_frequency_cutoff,
            seed=11,
            window_duration=8.0,
        )

    assert captured_alphas == pytest.approx(
        [2.0 * PSD_WINDOW_WIDTH_HZ / (40.0 - 2.0), 2.0 * PSD_WINDOW_WIDTH_HZ / (60.0 - 2.0)]
    )
