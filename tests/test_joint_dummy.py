"""Executable contract tests for :class:`JointDummyCorrelatedSimulator`.

Each test pins one facet the protocol promises: array shape/dtype, channel
units, deterministic seeding, RNG continuation across streamed chunks, the
one-sided-PSD Fourier convention, and provenance/metadata content. None of
these are prose claims -- every one is asserted directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise.simulators.joint_dummy import DEFAULT_SAMPLE_VARIANCE, JointDummyCorrelatedSimulator
from gwmock_noise.simulators.joint_protocol import ChannelDomain, ChannelKind


def _make_simulator(**overrides: object) -> JointDummyCorrelatedSimulator:
    kwargs: dict[str, object] = {
        "detectors": ["H1", "L1"],
        "witnesses": ["SEIS1"],
        "sampling_frequency": 64.0,
        "duration": 2.0,
    }
    kwargs.update(overrides)
    return JointDummyCorrelatedSimulator(**kwargs)  # type: ignore[arg-type]


# --- construction validation -------------------------------------------------


def test_rejects_empty_detectors() -> None:
    """Construction requires at least one strain channel."""
    with pytest.raises(ValueError, match="at least one detector"):
        JointDummyCorrelatedSimulator(detectors=[], witnesses=["SEIS1"])


def test_rejects_empty_witnesses() -> None:
    """Construction requires at least one witness channel."""
    with pytest.raises(ValueError, match="at least one witness"):
        JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=[])


def test_rejects_overlapping_channel_names() -> None:
    """Strain and witness channel names must be disjoint."""
    with pytest.raises(ValueError, match="must not share channel names"):
        JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=["H1"])


def test_rejects_duplicate_detectors() -> None:
    """Detector names must be unique."""
    with pytest.raises(ValueError, match="must not contain duplicates"):
        JointDummyCorrelatedSimulator(detectors=["H1", "H1"], witnesses=["SEIS1"])


@pytest.mark.parametrize("coupling", [-0.1, 1.0, 1.5])
def test_rejects_out_of_range_coupling(coupling: float) -> None:
    """Coupling must satisfy 0.0 <= coupling < 1.0."""
    with pytest.raises(ValueError, match="coupling must satisfy"):
        JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=["SEIS1"], coupling=coupling)


def test_changing_detector_set_is_rejected() -> None:
    """generate_joint refuses a detector set that differs from construction."""
    simulator = _make_simulator()
    with pytest.raises(ValueError, match="Changing the detector set"):
        simulator.generate_joint(1.0, 64.0, ["H1", "V1"], ["SEIS1"])


def test_changing_witness_set_is_rejected() -> None:
    """generate_joint refuses a witness set that differs from construction."""
    simulator = _make_simulator()
    with pytest.raises(ValueError, match="Changing the witness set"):
        simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS2"])


def test_reordering_channels_is_accepted() -> None:
    """generate_joint accepts a permutation of the constructed channel sets."""
    simulator = _make_simulator()
    realization = simulator.generate_joint(1.0, 64.0, ["L1", "H1"], ["SEIS1"])
    assert set(realization.strain) == {"H1", "L1"}


# --- shape / dtype / units (executable contracts) ----------------------------


def test_generate_joint_produces_simultaneous_strain_and_witness_output() -> None:
    """One call returns both strain and witness channels together."""
    simulator = _make_simulator()
    realization = simulator.generate_joint(2.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=1)
    assert set(realization.strain) == {"H1", "L1"}
    assert set(realization.witness) == {"SEIS1"}


def test_array_shapes_match_requested_duration() -> None:
    """Every channel array has exactly round(duration * sampling_frequency) samples."""
    simulator = _make_simulator()
    realization = simulator.generate_joint(1.5, 64.0, ["H1", "L1"], ["SEIS1"], seed=2)
    expected_samples = round(1.5 * 64.0)
    for array in (*realization.strain.values(), *realization.witness.values()):
        assert array.shape == (expected_samples,)


def test_array_dtype_is_float64() -> None:
    """Every channel array is float64, matching channel_metadata's declared dtype."""
    simulator = _make_simulator()
    realization = simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=3)
    for name, array in {**realization.strain, **realization.witness}.items():
        assert array.dtype == np.float64
        assert realization.channel_metadata[name].dtype == "float64"


def test_channel_metadata_declares_kind_unit_and_domain() -> None:
    """channel_metadata distinguishes strain from witness channels by kind and unit."""
    simulator = _make_simulator(strain_unit="strain", witness_unit="counts")
    realization = simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=4)
    for name in ("H1", "L1"):
        metadata = realization.channel_metadata[name]
        assert metadata.kind is ChannelKind.STRAIN
        assert metadata.unit == "strain"
        assert metadata.domain is ChannelDomain.TIME
        assert metadata.sampling_frequency == 64.0
    witness_metadata = realization.channel_metadata["SEIS1"]
    assert witness_metadata.kind is ChannelKind.WITNESS
    assert witness_metadata.unit == "counts"


# --- deterministic seeds / RNG continuation ----------------------------------


def test_same_seed_reproduces_the_same_realization() -> None:
    """Two independently constructed simulators seeded alike produce identical output."""
    first = _make_simulator().generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=99)
    second = _make_simulator().generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=99)
    for name in ("H1", "L1"):
        np.testing.assert_array_equal(first.strain[name], second.strain[name])
    np.testing.assert_array_equal(first.witness["SEIS1"], second.witness["SEIS1"])


def test_different_seeds_produce_different_realizations() -> None:
    """Distinct seeds are not degenerate."""
    first = _make_simulator().generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=1)
    second = _make_simulator().generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=2)
    assert not np.array_equal(first.strain["H1"], second.strain["H1"])


def test_streamed_chunks_continue_one_seeded_realization() -> None:
    """Streaming continuity: chunks concatenate to the same seeded single-shot call.

    This is the same continuation contract
    ``NoiseSimulator.generate_stream`` makes for strain-only output, applied
    here to both strain and witness channels at once.
    """
    long_simulator = _make_simulator(duration=1.0)
    stream_simulator = _make_simulator(duration=1.0)

    expected = long_simulator.generate_joint(4.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=42)

    stream = stream_simulator.generate_joint_stream(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=42)
    chunks = [next(stream) for _ in range(4)]

    for name in ("H1", "L1"):
        actual = np.concatenate([chunk.strain[name] for chunk in chunks])
        np.testing.assert_array_equal(actual, expected.strain[name])
    actual_witness = np.concatenate([chunk.witness["SEIS1"] for chunk in chunks])
    np.testing.assert_array_equal(actual_witness, expected.witness["SEIS1"])


def test_generate_joint_after_stream_reseed_breaks_continuity_deliberately() -> None:
    """Passing an explicit seed again restarts the generator from that seed."""
    simulator = _make_simulator(duration=1.0)
    first = simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=7)
    reseeded = simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=7)
    np.testing.assert_array_equal(first.strain["H1"], reseeded.strain["H1"])


# --- covariance access / Fourier convention ----------------------------------


def test_covariance_matrix_is_hermitian_at_every_frequency() -> None:
    """The reported cross-spectral density is Hermitian at every frequency bin."""
    simulator = _make_simulator()
    covariance = simulator.covariance()
    np.testing.assert_allclose(covariance.matrix, covariance.matrix.conj().swapaxes(-1, -2))


def test_covariance_frequency_grid_matches_rfftfreq_convention() -> None:
    """The frequency grid is the one-sided, non-negative numpy.fft.rfftfreq grid."""
    simulator = _make_simulator(sampling_frequency=64.0)
    covariance = simulator.covariance()
    assert covariance.frequencies[0] == 0.0
    assert covariance.frequencies[-1] == pytest.approx(32.0)
    assert np.all(np.diff(covariance.frequencies) > 0)


def test_covariance_matches_the_exact_closed_form_value() -> None:
    """covariance() returns 2 * Sigma / fs exactly, at every reported frequency.

    This is an analytic identity for a temporally white process, not a
    statistical estimate, so it is checked exactly rather than within a
    Monte Carlo tolerance.
    """
    coupling = 0.4
    sampling_frequency = 128.0
    simulator = _make_simulator(coupling=coupling, sampling_frequency=sampling_frequency)
    covariance = simulator.covariance()

    n = len(covariance.channel_order)
    expected = np.full((n, n), coupling, dtype=complex)
    np.fill_diagonal(expected, DEFAULT_SAMPLE_VARIANCE)
    expected *= 2.0 / sampling_frequency

    for bin_index in range(covariance.matrix.shape[0]):
        np.testing.assert_allclose(covariance.matrix[bin_index], expected)


def test_empirical_correlation_matches_the_analytic_coupling() -> None:
    """A long realization's sample correlation lands near the analytic coupling.

    This anchors the closed-form covariance() contract to what generate_joint
    actually draws, with a Monte Carlo tolerance appropriate to the sample
    size (n = 128,000 gives a standard error on the correlation coefficient
    of about (1 - rho^2) / sqrt(n) ~ 0.0026 at rho = 0.3; 0.05 is a
    comfortable multiple of that).
    """
    simulator = _make_simulator(coupling=0.3, sampling_frequency=64.0, duration=2000.0)
    realization = simulator.generate_joint(2000.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=2024)
    correlation = np.corrcoef(realization.strain["H1"], realization.witness["SEIS1"])[0, 1]
    assert correlation == pytest.approx(0.3, abs=0.05)


# --- provenance / metadata ----------------------------------------------------


def test_provenance_records_seed_and_backend_identity() -> None:
    """Provenance names the backend, package, seed and channel order actually used."""
    simulator = _make_simulator()
    realization = simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=17)
    assert realization.provenance["backend"] == "joint_dummy_correlated"
    assert realization.provenance["package"] == "gwmock_noise"
    assert realization.provenance["seed"] == 17
    assert realization.provenance["channel_order"] == ["H1", "L1", "SEIS1"]


def test_metadata_reports_current_runtime_state() -> None:
    """The metadata property reflects the most recent generate_joint call."""
    simulator = _make_simulator()
    simulator.generate_joint(1.0, 64.0, ["H1", "L1"], ["SEIS1"], seed=5)
    metadata = simulator.metadata
    assert metadata["implementation"] == "joint_dummy_correlated"
    assert metadata["duration"] == 1.0
    assert metadata["sampling_frequency"] == 64.0
    assert set(metadata["detectors"]) == {"H1", "L1"}
    assert metadata["witnesses"] == ["SEIS1"]
    assert metadata["seed"] == 5
