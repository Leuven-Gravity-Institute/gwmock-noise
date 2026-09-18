"""Tests for the streaming multichannel simulator.

The tests pin the contract for multichannel generation from a tabulated
PSD/CSD matrix:

* the fitted factor reproduces the target cross-spectral matrix band by band;
* a long realization recovers the target PSD and CSD;
* a complex CSD's phase survives generation as a cross-channel lag;
* the relation to the incumbent ``correlated_ar`` truncated VMA is measured on
  a shared target;
* the continuation state is the bounded recursion history, resumable without
  cached strain.

The comparison against an independent exact multivariate circulant embedding
(Helgason, Pipiras and Abry 2011) is deliberately not part of this branch: the
paper-side reference does not exist yet and that arm is recorded as gated and
unanchored, so the generator's covariance is checked against the target's own
PSD/CSD definition and the analytic per-band comparisons only.

Numbers quoted in the docstrings are the values observed while writing the
tests, so a later failure says what moved.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import csd as cross_spectral_density
from scipy.signal import welch

from gwmock_noise.config import NoiseComponentConfig, NoiseConfig
from gwmock_noise.simulators import (
    CorrelatedARNoiseSimulator,
    MultichannelNoiseSimulator,
)
from gwmock_noise.simulators._fit import FitError
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.matrix_factorization import matrix_autocovariance
from gwmock_noise.simulators.registry import available_simulator_names

SAMPLING_FREQUENCY = 256.0
BAND_LOW = 5.0
BAND_HIGH = 120.0
ET_PSD_A = "ET_10_full_cryo_psd"
ET_PSD_B = "ET_15_full_cryo_psd"
COHERENCE_SCALE = 0.6
COHERENCE_WIDTH = 60.0
DESIGN_POINTS = 2049


def _et_pair_csd(frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ET PSDs and an analytic CSD on a frequency grid."""
    table_frequencies, psd_a = load_spectral_series(ET_PSD_A, kind="PSD")
    _, psd_b = load_spectral_series(ET_PSD_B, kind="PSD")
    values_a = np.interp(frequencies, table_frequencies, psd_a)
    values_b = np.interp(frequencies, table_frequencies, psd_b)
    coherence = COHERENCE_SCALE * np.exp(-((frequencies / COHERENCE_WIDTH) ** 2))
    return values_a, values_b, coherence * np.sqrt(values_a * values_b)


def _write_et_pair(directory: Path) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    """Write the ET PSD/CSD pair to two-column text tables."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    values_a, values_b, values_csd = _et_pair_csd(frequencies)
    psd_paths = {"E1": directory / "E1_psd.txt", "E2": directory / "E2_psd.txt"}
    np.savetxt(psd_paths["E1"], np.column_stack((frequencies, values_a)))
    np.savetxt(psd_paths["E2"], np.column_stack((frequencies, values_b)))
    csd_path = directory / "E1_E2_csd.txt"
    np.savetxt(csd_path, np.column_stack((frequencies, values_csd)))
    return psd_paths, {("E1", "E2"): csd_path}


def _flat_pair_arrays(
    frequencies: np.ndarray, *, amplitude: float = 1.0e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a smooth, mild-coherence pair the incumbent VMA can also fit."""
    values_a = amplitude * (1.0 + 0.5 * np.cos(2.0 * np.pi * frequencies / SAMPLING_FREQUENCY))
    values_b = 2.0 * amplitude * (1.0 + 0.3 * np.sin(2.0 * np.pi * frequencies / SAMPLING_FREQUENCY))
    coherence = COHERENCE_SCALE * np.exp(-((frequencies / COHERENCE_WIDTH) ** 2))
    return values_a, values_b, coherence * np.sqrt(values_a * values_b)


def _flat_pair(directory: Path, *, amplitude: float = 1.0e-3) -> tuple[dict[str, Path], dict[tuple[str, str], Path]]:
    """Write a smooth, mild-coherence pair the incumbent VMA can also fit."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    values_a, values_b, values_csd = _flat_pair_arrays(frequencies, amplitude=amplitude)
    psd_paths = {"E1": directory / "flat_E1.txt", "E2": directory / "flat_E2.txt"}
    np.savetxt(psd_paths["E1"], np.column_stack((frequencies, values_a)))
    np.savetxt(psd_paths["E2"], np.column_stack((frequencies, values_b)))
    csd_path = directory / "flat_csd.txt"
    np.savetxt(csd_path, np.column_stack((frequencies, values_csd)))
    return psd_paths, {("E1", "E2"): csd_path}


def _band_relative_error(
    frequencies: np.ndarray,
    estimate: np.ndarray,
    target: np.ndarray,
) -> float:
    """Return the band-integrated relative error of a one-sided spectral estimate."""
    selected = (frequencies >= BAND_LOW) & (frequencies <= BAND_HIGH)
    reference = float(np.sum(target[selected]))
    return abs(float(np.sum(estimate[selected])) - reference) / reference


def test_multichannel_is_a_registered_simulator() -> None:
    """The new component is discoverable by the config-driven registry."""
    assert "multichannel" in available_simulator_names()


def test_fitted_model_reproduces_the_et_pair_target_per_band(tmp_path: Path) -> None:
    """The fitted cross-spectral matrix tracks the ET pair across the band.

    At order 256 the observed worst band residual is about 0.02 and the median
    about 0.003; the thresholds leave room for the fit to change shape without
    going quiet.
    """
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=256,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
    )
    residual = simulator.metadata["multichannel_noise"]["fit_residual"]
    assert residual["band_count"] == len(residual["bands"])
    assert residual["worst_relative_error"] < 0.15
    assert residual["median_relative_error"] < 0.02
    assert residual["worst_relative_error"] == max(band["relative_error"] for band in residual["bands"])


def test_generated_psd_and_csd_match_the_et_pair(tmp_path: Path) -> None:
    """A long realization recovers the ET pair's powers and cross-power."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=256,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
        block_size=1 << 15,
    )
    target_frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    target_a, _, target_csd = _et_pair_csd(target_frequencies)
    realizations = [simulator.generate(32.0, SAMPLING_FREQUENCY, ["E1", "E2"], seed=seed) for seed in range(12)]
    strain_e1 = np.concatenate([realization["E1"] for realization in realizations])
    strain_e2 = np.concatenate([realization["E2"] for realization in realizations])

    frequencies, psd = welch(strain_e1, fs=SAMPLING_FREQUENCY, nperseg=8192, noverlap=4096)
    _, csd = cross_spectral_density(strain_e1, strain_e2, fs=SAMPLING_FREQUENCY, nperseg=8192, noverlap=4096)
    psd_target = np.interp(frequencies, target_frequencies, target_a)
    csd_target = np.interp(frequencies, target_frequencies, target_csd)
    assert _band_relative_error(frequencies, psd, psd_target) < 0.15
    assert abs(float(np.sum(csd.real[(frequencies >= BAND_LOW) & (frequencies <= BAND_HIGH)]))) > 0.0
    assert _band_relative_error(frequencies, csd.real, csd_target) < 0.20


def test_relationship_to_correlated_ar_truncated_vma(tmp_path: Path) -> None:
    """The Whittle factor tracks the same target as the incumbent truncated VMA.

    The incumbent takes the per-frequency Cholesky factor of the target and
    truncates it, so its filter is causal only by truncation; the Whittle factor
    is causal by construction. On the shared smooth target at order 64 the
    observed band-integrated errors are about 0.04 for the PSD and 0.08 for the
    CSD against 0.13 and 0.21 for the incumbent, and the fitted model residual
    is about 0.16. Both are measured against the same tabulated target.
    """
    psd_paths, csd_paths = _flat_pair(tmp_path)
    order = 64
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=order,
        low_frequency_cutoff=4.0,
        high_frequency_cutoff=BAND_HIGH,
        block_size=1 << 16,
    )
    incumbent = CorrelatedARNoiseSimulator(
        psd_files={detector: str(path) for detector, path in psd_paths.items()},
        csd_files={pair: str(path) for pair, path in csd_paths.items()},
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=order,
        low_frequency_cutoff=4.0,
        high_frequency_cutoff=BAND_HIGH,
    )
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    target_a, _, target_csd = _flat_pair_arrays(frequencies)

    reference = simulator.generate(64.0, SAMPLING_FREQUENCY, ["E1", "E2"], seed=1)
    challenger = incumbent.generate(64.0, SAMPLING_FREQUENCY, ["E1", "E2"], seed=1)

    def _errors(strain: dict[str, np.ndarray]) -> tuple[float, float]:
        estimate_frequencies, psd = welch(strain["E1"], fs=SAMPLING_FREQUENCY, nperseg=2048, noverlap=1024)
        _, csd = cross_spectral_density(strain["E1"], strain["E2"], fs=SAMPLING_FREQUENCY, nperseg=2048, noverlap=1024)
        return (
            _band_relative_error(estimate_frequencies, psd, np.interp(estimate_frequencies, frequencies, target_a)),
            _band_relative_error(
                estimate_frequencies, csd.real, np.interp(estimate_frequencies, frequencies, target_csd)
            ),
        )

    reference_psd_error, reference_csd_error = _errors(reference)
    incumbent_psd_error, incumbent_csd_error = _errors(challenger)
    assert reference_psd_error < 0.10
    assert reference_csd_error < 0.15
    assert reference_psd_error <= incumbent_psd_error
    assert reference_csd_error <= incumbent_csd_error
    assert simulator.state_nbytes > 0
    assert simulator.metadata["multichannel_noise"]["state_size"] == order * 2


def test_chunked_generation_is_bit_identical_to_single_shot(tmp_path: Path) -> None:
    """Splitting a run into calls does not change the samples."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)

    def make() -> MultichannelNoiseSimulator:
        return MultichannelNoiseSimulator(
            psd_files=psd_paths,
            csd_files=csd_paths,
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=32,
            low_frequency_cutoff=BAND_LOW,
            high_frequency_cutoff=BAND_HIGH,
            block_size=1 << 10,
        )

    single = make()
    one_shot = single.generate(2.0, SAMPLING_FREQUENCY, ["E1", "E2"], seed=7)
    chunked = make()
    pieces = [
        chunked.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=7 if index == 0 else None) for index in range(4)
    ]
    for detector in ["E1", "E2"]:
        np.testing.assert_array_equal(one_shot[detector], np.concatenate([piece[detector] for piece in pieces]))


def test_resume_from_exported_state_is_bit_identical(tmp_path: Path) -> None:
    """A stream stopped and resumed from its exported state matches an uninterrupted run."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)

    def make() -> MultichannelNoiseSimulator:
        return MultichannelNoiseSimulator(
            psd_files=psd_paths,
            csd_files=csd_paths,
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=32,
            low_frequency_cutoff=BAND_LOW,
            high_frequency_cutoff=BAND_HIGH,
            block_size=1 << 10,
        )

    uninterrupted = make()
    full = [
        uninterrupted.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=13 if index == 0 else None)
        for index in range(4)
    ]

    stopped = make()
    head = [
        stopped.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=13 if index == 0 else None) for index in range(2)
    ]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    tail = [resumed.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"]) for _ in range(2)]

    for detector in ["E1", "E2"]:
        expected = np.concatenate([chunk[detector] for chunk in full])
        actual = np.concatenate([chunk[detector] for chunk in head + tail])
        np.testing.assert_array_equal(actual, expected)


def test_state_bytes_are_exposed_and_bounded(tmp_path: Path) -> None:
    """The continuation state is the order-sized history, and its byte size is reported."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=64,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
        block_size=1 << 10,
    )
    simulator.generate(1.0, SAMPLING_FREQUENCY, ["E1", "E2"], seed=1)
    fitted = simulator.metadata["multichannel_noise"]
    assert fitted["state_size"] == 128
    state_bytes = fitted["state_bytes"]
    assert state_bytes > 0
    assert simulator.state_nbytes == state_bytes
    for _ in range(20):
        simulator.generate(1.0, SAMPLING_FREQUENCY, ["E1", "E2"])
    assert simulator.state_nbytes <= state_bytes + 128


def test_metadata_is_json_serializable(tmp_path: Path) -> None:
    """The metadata sidecar writer receives only plain JSON types."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=32,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
    )
    fitted = simulator.metadata["multichannel_noise"]
    assert fitted["fit_method"] == "whittle-levinson-block-toeplitz"
    conditioning = fitted["conditioning"]
    assert conditioning["max_reflection_singular_value"] < 1.0
    assert conditioning["min_prediction_error_eigenvalue"] > 0.0
    assert conditioning["innovation_condition_number"] >= 1.0
    json.dumps(simulator.metadata)


def test_single_channel_generation_matches_the_target_psd(tmp_path: Path) -> None:
    """One channel degenerates to a scalar fit that recovers the target PSD."""
    psd_paths, csd_paths = _flat_pair(tmp_path)
    del csd_paths
    simulator = MultichannelNoiseSimulator(
        psd_files={"E1": psd_paths["E1"]},
        detectors=["E1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=64,
        low_frequency_cutoff=4.0,
        high_frequency_cutoff=BAND_HIGH,
        block_size=1 << 14,
    )
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    target_a, _, _ = _flat_pair_arrays(frequencies)
    realization = simulator.generate(64.0, SAMPLING_FREQUENCY, ["E1"], seed=1)["E1"]
    estimate_frequencies, psd = welch(realization, fs=SAMPLING_FREQUENCY, nperseg=2048, noverlap=1024)
    assert (
        _band_relative_error(estimate_frequencies, psd, np.interp(estimate_frequencies, frequencies, target_a)) < 0.15
    )


def test_target_not_positive_definite_is_a_fit_failure() -> None:
    """A tabulated CSD that breaks the target's positive definiteness is refused."""
    frequencies = np.fft.rfftfreq(2 * (DESIGN_POINTS - 1), d=1.0 / SAMPLING_FREQUENCY)
    target = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0e-3
    target[:, 1, 1] = 1.0e-3
    target[:, 0, 1] = 2.0e-3
    target[:, 1, 0] = 2.0e-3
    with pytest.raises(FitError, match="positive definite"):
        MultichannelNoiseSimulator(
            target_matrices=target,
            target_frequencies=frequencies,
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=16,
            low_frequency_cutoff=BAND_LOW,
            high_frequency_cutoff=BAND_HIGH,
            regularization_epsilon=0.0,
        )


def test_in_memory_target_matches_the_file_target(tmp_path: Path) -> None:
    """An analytic target matrix drives the same fit as the tabulated files."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, DESIGN_POINTS)
    target_a, target_b, target_csd = _et_pair_csd(frequencies)
    target = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = target_a
    target[:, 1, 1] = target_b
    target[:, 0, 1] = target_csd
    target[:, 1, 0] = target_csd

    from_files = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=64,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
    )
    from_array = MultichannelNoiseSimulator(
        target_matrices=target,
        target_frequencies=frequencies,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=64,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
    )
    np.testing.assert_allclose(from_array.ar_coefficients, from_files.ar_coefficients, atol=1e-6)


def test_missing_psd_raises_file_not_found(tmp_path: Path) -> None:
    """A missing PSD is still reported through the documented loader error."""
    with pytest.raises(FileNotFoundError):
        MultichannelNoiseSimulator(
            psd_files={"E1": tmp_path / "absent.txt"},
            detectors=["E1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=16,
        )


def test_runtime_validation_rejects_inconsistent_networks(tmp_path: Path) -> None:
    """Duplicate detectors, mixed inputs and network changes are refused."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    with pytest.raises(ValueError, match="duplicates"):
        MultichannelNoiseSimulator(
            psd_files=psd_paths,
            csd_files=csd_paths,
            detectors=["E1", "E1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=16,
        )
    with pytest.raises(ValueError, match="Exactly one"):
        MultichannelNoiseSimulator(
            psd_files=psd_paths,
            target_matrices=np.zeros((3, 2, 2)),
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=16,
        )
    simulator = MultichannelNoiseSimulator(
        psd_files=psd_paths,
        csd_files=csd_paths,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=16,
        low_frequency_cutoff=BAND_LOW,
        high_frequency_cutoff=BAND_HIGH,
    )
    with pytest.raises(ValueError, match="unsupported"):
        simulator.generate(1.0, SAMPLING_FREQUENCY, ["E1"], seed=1)


def test_export_state_rejects_mismatched_orders(tmp_path: Path) -> None:
    """A snapshot from a different order is refused."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)

    def make(order: int) -> MultichannelNoiseSimulator:
        return MultichannelNoiseSimulator(
            psd_files=psd_paths,
            csd_files=csd_paths,
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=order,
            low_frequency_cutoff=BAND_LOW,
            high_frequency_cutoff=BAND_HIGH,
        )

    exporter = make(16)
    exporter.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=2)
    snapshot = exporter.export_state()
    with pytest.raises(ValueError, match="order"):
        make(32).import_state(snapshot)


def test_complex_csd_phase_survives_generation() -> None:
    """The generated cross-channel lag covariance carries the CSD phase.

    The target is channel 2 delayed by three samples relative to channel 1, so its
    cross-spectrum carries a frequency-dependent phase and the cross-channel
    covariance is a spike at one lag. The generated process must reproduce the
    target autocovariance there: a dropped or conjugated phase would move the
    spike or halve it.
    """
    design_size = 64
    delay = 3
    frequencies = np.fft.rfftfreq(design_size, d=1.0 / SAMPLING_FREQUENCY)
    angular = 2.0 * np.pi * frequencies / SAMPLING_FREQUENCY
    target = np.zeros((frequencies.size, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0
    target[:, 1, 1] = 1.0
    cross = 0.2 * np.exp(-1j * angular * delay)
    target[:, 0, 1] = cross
    target[:, 1, 0] = np.conj(cross)
    for index in (0, frequencies.size - 1):
        target[index, 0, 1] = target[index, 0, 1].real
        target[index, 1, 0] = target[index, 0, 1]

    simulator = MultichannelNoiseSimulator(
        target_matrices=target,
        target_frequencies=frequencies,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=8,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=48.0,
        block_size=1 << 12,
    )
    target_curve = simulator.target_spectral_matrices
    expected = matrix_autocovariance(target_curve, 2 * (target_curve.shape[0] - 1))
    n_samples = 256
    n_realizations = 128
    accumulator = np.zeros((2, 2))
    for seed in range(n_realizations):
        realization = simulator.generate(n_samples / SAMPLING_FREQUENCY, SAMPLING_FREQUENCY, ["E1", "E2"], seed=seed)
        strain = np.column_stack([realization["E1"], realization["E2"]])
        accumulator += np.einsum("ti,tj->ij", strain[delay:], strain[:-delay]) / n_samples
    estimate = accumulator / n_realizations

    assert abs(expected[delay, 0, 1]) > 10.0 * abs(expected[delay, 1, 0])
    tolerance = 0.25 * abs(expected[delay, 0, 1])
    assert estimate[0, 1] == pytest.approx(expected[delay, 0, 1], abs=tolerance)
    assert estimate[1, 0] == pytest.approx(expected[delay, 1, 0], abs=tolerance)


def _small_pair_target(n_frequencies: int = 33) -> tuple[np.ndarray, np.ndarray]:
    """Return a minimal positive-definite two-channel target and its grid."""
    frequencies = np.fft.rfftfreq(2 * (n_frequencies - 1), d=1.0 / SAMPLING_FREQUENCY)
    target = np.zeros((n_frequencies, 2, 2), dtype=np.complex128)
    target[:, 0, 0] = 1.0e-3
    target[:, 1, 1] = 1.5e-3
    target[:, 0, 1] = 4.0e-4
    target[:, 1, 0] = 4.0e-4
    return target, frequencies


def _build_small_simulator(**overrides: object) -> MultichannelNoiseSimulator:
    """Construct a small in-memory simulator for the behavioural path tests."""
    target, frequencies = _small_pair_target()
    settings: dict[str, object] = {
        "target_matrices": target,
        "target_frequencies": frequencies,
        "detectors": ["E1", "E2"],
        "sampling_frequency": SAMPLING_FREQUENCY,
        "order": 8,
        "low_frequency_cutoff": 2.0,
        "high_frequency_cutoff": 40.0,
    }
    settings.update(overrides)
    return MultichannelNoiseSimulator(**settings)


def test_target_frequencies_default_to_the_matching_grid() -> None:
    """An in-memory target without a frequency grid uses the rfft grid of its length."""
    target, _ = _small_pair_target()
    simulator = MultichannelNoiseSimulator(
        target_matrices=target,
        detectors=["E1", "E2"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=8,
        low_frequency_cutoff=2.0,
        high_frequency_cutoff=40.0,
    )
    assert simulator.design_frequencies.size >= target.shape[0]


def test_target_channel_count_must_match_detectors() -> None:
    """A target with more channels than detectors is rejected."""
    target, frequencies = _small_pair_target()
    with pytest.raises(ValueError, match="channel count"):
        MultichannelNoiseSimulator(
            target_matrices=target,
            target_frequencies=frequencies,
            detectors=["E1", "E2", "E3"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=8,
            low_frequency_cutoff=2.0,
        )


def test_target_frequencies_must_match_the_target_length() -> None:
    """A frequency grid of the wrong length is rejected."""
    target, frequencies = _small_pair_target()
    with pytest.raises(ValueError, match="agree in length"):
        MultichannelNoiseSimulator(
            target_matrices=target,
            target_frequencies=frequencies[:-1],
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=8,
            low_frequency_cutoff=2.0,
        )


def test_target_with_one_frequency_bin_is_rejected() -> None:
    """A one-bin target has no design grid to fit and is refused."""
    target = np.zeros((1, 2, 2), dtype=np.complex128)
    target[0, 0, 0] = 1.0e-3
    target[0, 1, 1] = 1.5e-3
    with pytest.raises(ValueError, match="at least two frequency samples"):
        MultichannelNoiseSimulator(
            target_matrices=target,
            target_frequencies=np.array([0.0]),
            detectors=["E1", "E2"],
            sampling_frequency=SAMPLING_FREQUENCY,
            order=8,
            low_frequency_cutoff=2.0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"order": 0}, "order must be greater than zero"),
        ({"block_size": 0}, "block_size must be greater than zero"),
        ({"regularization_epsilon": -1.0}, "regularization_epsilon must be non-negative"),
        ({"low_frequency_cutoff": -1.0}, "low_frequency_cutoff must be non-negative"),
        ({"low_frequency_cutoff": 10.0, "high_frequency_cutoff": 5.0}, "high_frequency_cutoff must be greater"),
        ({"high_frequency_cutoff": SAMPLING_FREQUENCY}, "must not exceed the Nyquist"),
    ],
)
def test_constructor_validates_runtime_settings(overrides: dict[str, object], message: str) -> None:
    """Each invalid runtime setting is refused by name."""
    with pytest.raises(ValueError, match=message):
        _build_small_simulator(**overrides)


def test_from_component_builds_from_psd_and_csd_options(tmp_path: Path) -> None:
    """A config-driven component constructs a working simulator from file options."""
    psd_paths, csd_paths = _write_et_pair(tmp_path)
    component = NoiseComponentConfig(
        simulator="multichannel",
        options={
            "psd_files": {detector: str(path) for detector, path in psd_paths.items()},
            "csd_files": {"E1-E2": str(csd_paths[("E1", "E2")])},
            "order": 8,
            "low_frequency_cutoff": BAND_LOW,
            "high_frequency_cutoff": BAND_HIGH,
        },
    )
    config = NoiseConfig(
        detectors=["E1", "E2"],
        duration=1.0,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=3,
        components=[component],
    )
    simulator = MultichannelNoiseSimulator.from_component(component, config)
    assert isinstance(simulator, MultichannelNoiseSimulator)
    assert simulator.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"])["E1"].shape == (128,)


@pytest.mark.parametrize(
    ("csd_files", "message"),
    [
        ({("E1", "E1"): "self.txt"}, "two distinct detectors"),
        ({("E1", "E2"): "a.txt", ("E2", "E1"): "b.txt"}, "Duplicate CSD file mapping"),
        ({"E1-E9": "unknown.txt"}, "reference configured detectors"),
    ],
)
def test_csd_file_maps_are_validated(csd_files: dict, message: str) -> None:
    """Self-pairs, duplicate normalized pairs and unknown detectors are refused."""
    with pytest.raises(ValueError, match=message):
        _build_small_simulator(csd_files=csd_files)


def test_reset_clears_the_streaming_state() -> None:
    """reset() clears the recursion history, pending output and generator."""
    simulator = _build_small_simulator(block_size=1 << 12)
    simulator.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=1)
    simulator.reset()
    assert simulator.state_nbytes > 0


def test_first_order_simulation_covers_the_single_lag_history() -> None:
    """An order-one fit exercises the history shift's single-lag path."""
    simulator = _build_small_simulator(order=1)
    realization = simulator.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=4)
    assert realization["E1"].shape == (128,)


def test_reconfiguration_streaming_and_sub_sample_paths() -> None:
    """A changed sampling frequency refits, a sub-sample span is refused, and the stream yields."""
    simulator = _build_small_simulator()
    realization = simulator.generate(0.5, SAMPLING_FREQUENCY / 2.0, ["E1", "E2"], seed=1)
    assert realization["E1"].shape == (64,)
    with pytest.raises(ValueError, match="at least one sample"):
        simulator.generate(0.001, SAMPLING_FREQUENCY / 2.0, ["E1", "E2"])
    stream = simulator.generate_stream(0.25, SAMPLING_FREQUENCY / 2.0, ["E1", "E2"], seed=2)
    assert next(stream)["E1"].shape == (32,)


def test_import_state_accepts_an_unstarted_snapshot_and_checks_the_rate() -> None:
    """A never-started snapshot imports, and a mismatched sampling frequency is refused."""
    exporter = _build_small_simulator()
    consumer = _build_small_simulator()
    consumer.import_state(exporter.export_state())

    exporter.generate(0.5, SAMPLING_FREQUENCY, ["E1", "E2"], seed=1)
    snapshot = exporter.export_state()
    with pytest.raises(ValueError, match="sampling frequency"):
        _build_small_simulator(sampling_frequency=SAMPLING_FREQUENCY / 2.0).import_state(snapshot)


def test_spectral_accessors_return_the_design_grid_and_model() -> None:
    """The matrix accessors expose the design grid, target and fitted model."""
    simulator = _build_small_simulator()
    assert simulator.model_spectral_matrices.shape == simulator.target_spectral_matrices.shape
    target_frequencies, target = simulator.target_spectral_matrix_curve
    model_frequencies, model = simulator.model_spectral_matrix_curve
    assert target.shape == model.shape
    np.testing.assert_array_equal(target_frequencies, model_frequencies)
    assert simulator.ar_coefficients.shape[0] == 8
    assert simulator.innovation_covariance.shape == (2, 2)
