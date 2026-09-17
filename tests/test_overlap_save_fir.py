"""Tests for the minimum-phase overlap-save FIR colouring simulator."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import OverlapSaveFirSimulator, open_stream
from gwmock_noise.config import NoiseComponentConfig, NoiseConfig
from gwmock_noise.simulators.overlap_save import (
    MAX_FILTER_LENGTH,
    MIN_FILTER_LENGTH,
    OverlapSaveFilter,
    design_colouring_filter,
    minimum_phase_from_magnitude,
    validate_filter_length,
)
from gwmock_noise.simulators.registry import available_simulator_names

SAMPLING_FREQUENCY = 256.0
DESIGN_SIZE = 2048
LOW_FREQUENCY = 8.0
HIGH_FREQUENCY = 120.0


def _band_limited_target(
    *,
    design_size: int = DESIGN_SIZE,
    sampling_frequency: float = SAMPLING_FREQUENCY,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth band-limited one-sided PSD and its frequency grid."""
    grid = np.fft.rfftfreq(design_size, d=1.0 / sampling_frequency)
    target = np.zeros_like(grid)
    band = (grid >= LOW_FREQUENCY) & (grid <= HIGH_FREQUENCY)
    target[band] = 1.0e-3 * (1.0 + 0.3 * np.sin(grid[band] / 5.0))
    return target, grid


def _write_flat_psd(path: Path, *, sampling_frequency: float = SAMPLING_FREQUENCY, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 129)
    np.savetxt(path, np.column_stack((frequencies, np.full_like(frequencies, value))))
    return path


def _spectral_error(taps: np.ndarray, target: np.ndarray, *, sampling_frequency: float) -> float:
    """Return the relative spectral error of a filter against a target PSD."""
    design_size = 2 * (target.size - 1)
    achieved = 2.0 * np.abs(np.fft.rfft(taps, n=design_size)) ** 2 / sampling_frequency
    band = target > 0.0
    return float(np.linalg.norm(achieved[band] - target[band]) / np.linalg.norm(target[band]))


@pytest.mark.parametrize("exponent", range(4, 17))
def test_validate_filter_length_accepts_documented_sweep(exponent: int) -> None:
    """Every power of two in the documented sweep is a valid truncation control."""
    validate_filter_length(1 << exponent)


@pytest.mark.parametrize("filter_length", [0, 100, 24, 1 << 17])
def test_validate_filter_length_rejects_off_grid_values(filter_length: int) -> None:
    """Filter lengths off the documented power-of-two grid are rejected."""
    with pytest.raises(ValueError, match=r"power of two|must lie in"):
        validate_filter_length(filter_length)


def test_design_colouring_filter_returns_requested_length() -> None:
    """The designed filter has exactly ``filter_length`` taps and is deterministic."""
    target, _ = _band_limited_target()
    first = design_colouring_filter(target, delta_frequency=0.125, filter_length=64, minimum_phase=True)
    second = design_colouring_filter(target, delta_frequency=0.125, filter_length=64, minimum_phase=True)
    assert first.shape == (64,)
    assert np.array_equal(first, second)


def test_design_colouring_filter_matches_target_variance() -> None:
    """The taps are scaled so the filtered white noise carries the target variance."""
    target, _ = _band_limited_target()
    delta_frequency = SAMPLING_FREQUENCY / DESIGN_SIZE
    taps = design_colouring_filter(target, delta_frequency=delta_frequency, filter_length=256, minimum_phase=True)
    expected = delta_frequency * float(np.sum(target))
    assert np.sum(taps**2) == pytest.approx(expected, rel=1e-9)


def test_design_colouring_filter_error_decreases_with_filter_length() -> None:
    """The spectral error is monotonic in the truncation control ``L_f``."""
    target, _ = _band_limited_target()
    delta_frequency = SAMPLING_FREQUENCY / DESIGN_SIZE
    errors = [
        _spectral_error(
            design_colouring_filter(target, delta_frequency=delta_frequency, filter_length=length, minimum_phase=True),
            target,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        for length in (16, 32, 64, 128, 256)
    ]
    assert all(later < earlier for earlier, later in pairwise(errors))


def test_minimum_phase_from_magnitude_preserves_magnitude_response() -> None:
    """The cepstral factorisation reproduces the magnitude it is given."""
    target, _ = _band_limited_target()
    windowed = design_colouring_filter(
        target, delta_frequency=SAMPLING_FREQUENCY / DESIGN_SIZE, filter_length=64, minimum_phase=False
    )
    magnitude = np.abs(np.fft.rfft(windowed, n=DESIGN_SIZE))
    reconstructed = np.abs(np.fft.rfft(minimum_phase_from_magnitude(magnitude), n=DESIGN_SIZE))
    np.testing.assert_allclose(reconstructed, magnitude, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("filter_length", [16, 32, 64, 128])
def test_design_minimum_phase_keeps_windowed_magnitude_response(filter_length: int) -> None:
    """Truncating the minimum-phase design keeps the windowed design's magnitude response."""
    target, _ = _band_limited_target()
    delta_frequency = SAMPLING_FREQUENCY / DESIGN_SIZE
    minimum_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=filter_length, minimum_phase=True
    )
    linear_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=filter_length, minimum_phase=False
    )
    min_response = np.abs(np.fft.rfft(minimum_phase, n=DESIGN_SIZE))
    linear_response = np.abs(np.fft.rfft(linear_phase, n=DESIGN_SIZE))
    min_response /= np.linalg.norm(min_response)
    linear_response /= np.linalg.norm(linear_response)
    assert np.allclose(min_response, linear_response, atol=1e-4)


def test_minimum_phase_energy_is_front_loaded() -> None:
    """The minimum-phase filter concentrates more energy in its first half than the linear-phase one."""
    target, _ = _band_limited_target()
    delta_frequency = SAMPLING_FREQUENCY / DESIGN_SIZE
    minimum_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=64, minimum_phase=True
    )
    linear_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=64, minimum_phase=False
    )
    min_front = float(np.sum(minimum_phase[:32] ** 2) / np.sum(minimum_phase**2))
    linear_front = float(np.sum(linear_phase[:32] ** 2) / np.sum(linear_phase**2))
    assert min_front > 0.5
    assert min_front > linear_front


def test_minimum_phase_filter_zeros_are_inside_unit_circle() -> None:
    """A minimum-phase filter has all its zeros inside the unit circle, unlike the linear-phase design."""
    target, _ = _band_limited_target()
    delta_frequency = SAMPLING_FREQUENCY / DESIGN_SIZE
    minimum_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=32, minimum_phase=True
    )
    linear_phase = design_colouring_filter(
        target, delta_frequency=delta_frequency, filter_length=32, minimum_phase=False
    )
    assert np.max(np.abs(np.roots(minimum_phase))) < 1.0
    assert np.max(np.abs(np.roots(linear_phase))) > 1.0


def test_minimum_phase_from_magnitude_validates_input() -> None:
    """The cepstral factorisation rejects malformed magnitude arrays."""
    with pytest.raises(ValueError, match="one-dimensional"):
        minimum_phase_from_magnitude(np.ones((2, 3)))
    with pytest.raises(ValueError, match="more than two"):
        minimum_phase_from_magnitude(np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        minimum_phase_from_magnitude(np.array([1.0, np.nan, 1.0]))


def test_design_colouring_filter_rejects_zero_variance_target() -> None:
    """A target that integrates to zero variance cannot define a filter."""
    target = np.zeros(DESIGN_SIZE // 2 + 1)
    with pytest.raises(ValueError, match="zero variance"):
        design_colouring_filter(target, delta_frequency=0.125, filter_length=64)


def test_design_colouring_filter_rejects_coarse_grid() -> None:
    """The design grid must be at least as long as the filter."""
    target = np.ones(9)
    with pytest.raises(ValueError, match="too coarse"):
        design_colouring_filter(target, delta_frequency=0.125, filter_length=64)


def test_overlap_save_matches_direct_convolution() -> None:
    """Block overlap-save reproduces a direct long-span linear convolution."""
    rng = np.random.default_rng(2026)
    taps = rng.standard_normal(37)
    convolver = OverlapSaveFilter(taps, block_size=64)
    samples = rng.standard_normal(64 * 9)
    blocked = np.concatenate([convolver.process(samples[index : index + 64]) for index in range(0, 64 * 9, 64)])
    np.testing.assert_allclose(blocked, np.convolve(samples, taps)[: samples.size], atol=1e-12)


def test_overlap_save_filter_memory_is_filter_length_minus_one() -> None:
    """The continuation state is exactly the filter memory, independent of span."""
    convolver = OverlapSaveFilter(np.ones(33), block_size=16)
    assert convolver.filter_memory.shape == (32,)
    assert convolver.state_nbytes == 32 * np.dtype(float).itemsize
    convolver.process(np.ones(16))
    assert convolver.filter_memory.shape == (32,)
    assert convolver.state_nbytes == 32 * np.dtype(float).itemsize


def test_overlap_save_filter_reset_clears_memory() -> None:
    """Reset restores the zero initial filter memory."""
    convolver = OverlapSaveFilter(np.ones(9), block_size=8)
    convolver.process(np.arange(8.0))
    assert np.any(convolver.filter_memory != 0.0)
    convolver.reset()
    assert np.array_equal(convolver.filter_memory, np.zeros(8))


def test_overlap_save_filter_validates_inputs() -> None:
    """Malformed taps and blocks are rejected."""
    with pytest.raises(ValueError, match="non-empty one-dimensional"):
        OverlapSaveFilter(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="block_size"):
        OverlapSaveFilter(np.ones(4), block_size=0)
    convolver = OverlapSaveFilter(np.ones(4), block_size=8)
    with pytest.raises(ValueError, match="must have shape"):
        convolver.process(np.zeros(7))


def test_simulator_generates_requested_shape_and_detectors() -> None:
    """The simulator returns the requested sample count per detector."""
    target, _ = _band_limited_target()
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        detectors=["H1", "L1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    strain = simulator.generate(1.0, SAMPLING_FREQUENCY, ["H1", "L1"], seed=5)
    assert set(strain) == {"H1", "L1"}
    assert strain["H1"].shape == (256,)
    assert strain["L1"].shape == (256,)


def test_simulator_chunked_generation_is_bit_identical_to_single_shot() -> None:
    """Splitting a run into calls does not change the samples."""
    target, _ = _band_limited_target()
    single = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    chunked = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    one_shot = single.generate(2.0, SAMPLING_FREQUENCY, ["H1"], seed=9)["H1"]
    pieces = [
        chunked.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=9 if index == 0 else None)["H1"] for index in range(4)
    ]
    np.testing.assert_array_equal(one_shot, np.concatenate(pieces))


def test_simulator_state_size_is_independent_of_generated_span() -> None:
    """The continuation state is the filter memory, not the generated strain."""
    target, _ = _band_limited_target()
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=128,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=1)
    short_bytes = simulator.state_nbytes
    for _ in range(64):
        simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"])
    assert simulator.state_nbytes <= short_bytes + 32
    assert simulator.state_nbytes < 4096


def test_simulator_state_size_grows_with_filter_length() -> None:
    """A longer filter costs more state bytes, the trade the truncation control exposes."""
    target, _ = _band_limited_target()
    sizes = []
    for filter_length in (16, 64, 256, 1024):
        simulator = OverlapSaveFirSimulator(
            target_psd=target,
            filter_length=filter_length,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            block_size=128,
            low_frequency_cutoff=LOW_FREQUENCY,
        )
        simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=1)
        sizes.append(simulator.state_nbytes)
    assert all(later > earlier for earlier, later in pairwise(sizes))


def test_simulator_resume_is_bit_identical() -> None:
    """A stream stopped and resumed from its exported state matches an uninterrupted run."""
    target, _ = _band_limited_target()

    def make() -> OverlapSaveFirSimulator:
        return OverlapSaveFirSimulator(
            target_psd=target,
            filter_length=64,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            block_size=128,
            low_frequency_cutoff=LOW_FREQUENCY,
        )

    uninterrupted = make()
    full = [
        uninterrupted.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=31 if index == 0 else None)["H1"]
        for index in range(5)
    ]

    stopped = make()
    head = [
        stopped.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=31 if index == 0 else None)["H1"] for index in range(3)
    ]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    tail = [resumed.generate(0.5, SAMPLING_FREQUENCY, ["H1"])["H1"] for _ in range(2)]

    for expected, actual in zip(full, head + tail, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_simulator_export_state_rejects_mismatched_settings() -> None:
    """A snapshot from a different configuration is refused."""
    target, _ = _band_limited_target()
    exporter = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    exporter.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=2)
    snapshot = exporter.export_state()

    other = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=128,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    with pytest.raises(ValueError, match="filter settings"):
        other.import_state(snapshot)


def test_simulator_accepts_array_target_without_frequencies() -> None:
    """An analytic target array is usable on the default one-sided grid."""
    target, _ = _band_limited_target(design_size=256)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    assert simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=4)["H1"].shape == (128,)


def test_simulator_accepts_array_target_with_frequencies() -> None:
    """An explicit frequency grid is honoured for an array target."""
    target, grid = _band_limited_target(design_size=512)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        target_frequencies=grid,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    assert simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=4)["H1"].shape == (128,)


def test_simulator_rejects_conflicting_or_missing_targets(tmp_path: Path) -> None:
    """Exactly one target source must be supplied."""
    target, _ = _band_limited_target(design_size=256)
    psd_path = _write_flat_psd(tmp_path / "flat_psd.txt")
    with pytest.raises(ValueError, match="Exactly one"):
        OverlapSaveFirSimulator(
            psd_file=psd_path,
            target_psd=target,
            filter_length=32,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
        )
    with pytest.raises(ValueError, match="Exactly one"):
        OverlapSaveFirSimulator(filter_length=32, detectors=["H1"], sampling_frequency=SAMPLING_FREQUENCY)


def test_simulator_validates_filter_length_at_construction() -> None:
    """The documented truncation control is enforced when the simulator is built."""
    target, _ = _band_limited_target(design_size=256)
    with pytest.raises(ValueError, match="power of two"):
        OverlapSaveFirSimulator(
            target_psd=target,
            filter_length=48,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
        )


def test_simulator_is_registered_as_a_configurable_component() -> None:
    """The new simulator joins the registry of config-driven components."""
    assert "overlap_save_fir" in available_simulator_names()


def test_simulator_metadata_describes_the_design() -> None:
    """Metadata exposes the truncation control and the measured state size."""
    target, _ = _band_limited_target(design_size=256)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    metadata = simulator.metadata
    assert metadata["implementation"] == "overlap_save_fir"
    assert metadata["overlap_save_fir"]["filter_length"] == 32
    assert metadata["overlap_save_fir"]["minimum_phase"] is True
    assert metadata["overlap_save_fir"]["state_bytes"] == simulator.state_nbytes


def test_simulator_exposes_target_and_filter_for_analysis() -> None:
    """The target array and the designed taps are available to a caller."""
    target, grid = _band_limited_target()
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        target_frequencies=grid,
        filter_length=64,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
        high_frequency_cutoff=HIGH_FREQUENCY,
    )
    assert simulator.filter_taps.shape == (64,)
    assert simulator.target_psd.shape == (simulator._design_size // 2 + 1,)
    returned = simulator.filter_taps
    returned[:] = 0.0
    assert np.any(simulator.filter_taps != 0.0)


def test_simulator_streams_through_the_public_helper() -> None:
    """The simulator drives the package's streaming protocol."""
    target, _ = _band_limited_target(design_size=256)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    stream = open_stream(simulator, chunk_duration=0.5, sampling_frequency=SAMPLING_FREQUENCY, detectors=["H1"], seed=6)
    chunks = [next(stream)["H1"] for _ in range(3)]
    assert all(chunk.shape == (128,) for chunk in chunks)


def test_simulator_output_variance_matches_target() -> None:
    """A long realization carries the target band variance."""
    target, grid = _band_limited_target()
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        target_frequencies=grid,
        filter_length=256,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=512,
        low_frequency_cutoff=LOW_FREQUENCY,
        high_frequency_cutoff=HIGH_FREQUENCY,
    )
    strain = simulator.generate(120.0, SAMPLING_FREQUENCY, ["H1"], seed=13)["H1"]
    expected = (SAMPLING_FREQUENCY / DESIGN_SIZE) * float(np.sum(target))
    assert np.var(strain) == pytest.approx(expected, rel=0.15)


def test_minimum_phase_and_linear_phase_give_different_realizations() -> None:
    """The phase factorisation choice changes the time series."""
    target, _ = _band_limited_target()
    minimum_phase = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        minimum_phase=True,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    linear_phase = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        minimum_phase=False,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=LOW_FREQUENCY,
    )
    first = minimum_phase.generate(1.0, SAMPLING_FREQUENCY, ["H1"], seed=17)["H1"]
    second = linear_phase.generate(1.0, SAMPLING_FREQUENCY, ["H1"], seed=17)["H1"]
    assert not np.array_equal(first, second)


def test_min_filter_length_and_max_filter_length_constants() -> None:
    """The documented sweep bounds are the fourth and sixteenth powers of two."""
    assert MIN_FILTER_LENGTH == 16
    assert MAX_FILTER_LENGTH == 65536


def test_design_colouring_filter_validates_target_and_delta_frequency() -> None:
    """The free function rejects a malformed target and a non-positive spacing."""
    target, _ = _band_limited_target()
    with pytest.raises(ValueError, match="one-dimensional"):
        design_colouring_filter(np.ones((2, 3)), delta_frequency=0.125, filter_length=64)
    with pytest.raises(ValueError, match="more than two"):
        design_colouring_filter(np.ones(2), delta_frequency=0.125, filter_length=64)
    with pytest.raises(ValueError, match="delta_frequency"):
        design_colouring_filter(target, delta_frequency=0.0, filter_length=64)


def test_overlap_save_filter_handles_a_single_tap() -> None:
    """A one-tap filter scales the input and keeps an empty memory."""
    convolver = OverlapSaveFilter(np.array([2.0]), block_size=4)
    assert convolver.filter_memory.shape == (0,)
    np.testing.assert_allclose(convolver.process(np.arange(4.0)), 2.0 * np.arange(4.0))


def test_simulator_accepts_a_tabulated_psd_file(tmp_path: Path) -> None:
    """A two-column PSD table is loaded, masked and used to design the filter."""
    psd_path = _write_flat_psd(tmp_path / "flat_psd.txt")
    simulator = OverlapSaveFirSimulator(
        psd_file=psd_path,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    strain = simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=2)
    assert strain["H1"].shape == (128,)
    assert simulator.metadata["overlap_save_fir"]["psd_file"] == str(psd_path)


def test_simulator_rejects_a_non_positive_block_size() -> None:
    """The overlap-save block length must be positive."""
    target, _ = _band_limited_target(design_size=256)
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        OverlapSaveFirSimulator(
            target_psd=target,
            filter_length=32,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            block_size=0,
        )


def test_simulator_from_component_builds_from_psd_options(tmp_path: Path) -> None:
    """A config-driven component constructs a working simulator from a PSD table."""
    psd_path = _write_flat_psd(tmp_path / "component_psd.txt")
    component = NoiseComponentConfig(
        simulator="overlap_save_fir",
        options={
            "psd_file": str(psd_path),
            "filter_length": 32,
            "block_size": 128,
            "low_frequency_cutoff": 1.0,
        },
    )
    config = NoiseConfig(
        detectors=["H1"],
        duration=1.0,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=5,
        components=[component],
    )
    simulator = OverlapSaveFirSimulator.from_component(component, config)
    assert isinstance(simulator, OverlapSaveFirSimulator)
    assert simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"])["H1"].shape == (128,)


def test_simulator_from_component_requires_a_target() -> None:
    """A component without a PSD file or array target is rejected."""
    component = NoiseComponentConfig(simulator="overlap_save_fir", options={})
    config = NoiseConfig(detectors=["H1"], duration=1.0, sampling_frequency=SAMPLING_FREQUENCY, components=[component])
    with pytest.raises(ValueError, match="requires 'psd_file' or 'target_psd'"):
        OverlapSaveFirSimulator.from_component(component, config)


def test_simulator_validates_array_target_shape_and_frequencies() -> None:
    """An array target needs at least three samples and a matching, increasing grid."""
    target, grid = _band_limited_target(design_size=512)
    with pytest.raises(ValueError, match="at least three"):
        OverlapSaveFirSimulator(
            target_psd=np.ones(2),
            filter_length=32,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
        )
    with pytest.raises(ValueError, match="same length"):
        OverlapSaveFirSimulator(
            target_psd=target,
            target_frequencies=grid[:-1],
            filter_length=32,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        OverlapSaveFirSimulator(
            target_psd=target,
            target_frequencies=grid[::-1].copy(),
            filter_length=32,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
        )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"duration": 0.0}, "duration must be greater than zero"),
        ({"sampling_frequency": 0.0}, "sampling_frequency must be greater than zero"),
        ({"detectors": []}, "at least one detector"),
        ({"detectors": ["H1", "H1"]}, "duplicate"),
        ({"low_frequency_cutoff": -1.0}, "non-negative"),
        ({"low_frequency_cutoff": 10.0, "high_frequency_cutoff": 5.0}, "greater than low_frequency_cutoff"),
        ({"high_frequency_cutoff": 1000.0}, "must not exceed the Nyquist"),
    ],
)
def test_simulator_runtime_validation(overrides: dict[str, object], match: str) -> None:
    """Runtime and band arguments are validated at construction."""
    target, grid = _band_limited_target(design_size=512)
    arguments: dict[str, object] = {
        "target_psd": target,
        "target_frequencies": grid,
        "filter_length": 32,
        "detectors": ["H1"],
        "sampling_frequency": SAMPLING_FREQUENCY,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=match):
        OverlapSaveFirSimulator(**arguments)


def test_simulator_rejects_a_band_with_no_design_bins() -> None:
    """A band that falls between design-grid points is rejected."""
    target, grid = _band_limited_target(design_size=256)
    with pytest.raises(ValueError, match="no design bins"):
        OverlapSaveFirSimulator(
            target_psd=target,
            target_frequencies=grid,
            filter_length=16,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            low_frequency_cutoff=127.2,
            high_frequency_cutoff=127.8,
        )


def test_simulator_reconfigures_when_sampling_frequency_changes() -> None:
    """A changed sampling frequency rebuilds the design grid and clears the stream."""
    target, grid = _band_limited_target(design_size=512)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        target_frequencies=grid,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    assert simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=3)["H1"].shape == (128,)
    resampled = simulator.generate(0.5, 2 * SAMPLING_FREQUENCY, ["H1"])["H1"]
    assert resampled.shape == (256,)
    assert simulator.sampling_frequency == 2 * SAMPLING_FREQUENCY


def test_simulator_rejects_a_sub_sample_duration() -> None:
    """A duration that rounds to zero samples is rejected."""
    target, _ = _band_limited_target(design_size=256)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    with pytest.raises(ValueError, match="at least one sample"):
        simulator.generate(0.001, SAMPLING_FREQUENCY, ["H1"], seed=1)


def test_simulator_import_state_rejects_detector_and_frequency_mismatch() -> None:
    """A snapshot from a different detector set or sampling frequency is refused."""
    target, _ = _band_limited_target(design_size=512)
    exporter = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    exporter.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=4)
    snapshot = exporter.export_state()

    other_detectors = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["L1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    with pytest.raises(ValueError, match="detectors do not match"):
        other_detectors.import_state(snapshot)

    other_frequency = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=32,
        detectors=["H1"],
        sampling_frequency=2 * SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    with pytest.raises(ValueError, match="sampling frequency does not match"):
        other_frequency.import_state(snapshot)


def test_simulator_state_bytes_include_bounded_pending_output() -> None:
    """A non-block-multiple request leaves pending output that is counted in the state.

    The pending samples are phase-of-block bookkeeping: bounded by one block and
    present only between the block boundary and the request boundary, so they add
    to the serialized state without making it grow with the generated span.
    """
    target, _ = _band_limited_target(design_size=512)
    simulator = OverlapSaveFirSimulator(
        target_psd=target,
        filter_length=64,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        block_size=128,
        low_frequency_cutoff=1.0,
    )
    simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=6)
    aligned_bytes = simulator.state_nbytes
    assert simulator.continuation_state()["pending_samples"]["H1"].shape == (0,)

    simulator.generate(0.25, SAMPLING_FREQUENCY, ["H1"])
    pending = simulator.continuation_state()["pending_samples"]["H1"]
    assert 0 < pending.shape[0] <= simulator.block_size
    assert simulator.state_nbytes > aligned_bytes

    for _ in range(64):
        simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"])
    assert abs(simulator.state_nbytes - aligned_bytes) <= simulator.block_size * np.dtype(float).itemsize + 32
