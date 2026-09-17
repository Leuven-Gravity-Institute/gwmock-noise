"""Tests for RNG-state resumability of the overlap-add stitcher."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import ColoredNoiseSimulator
from gwmock_noise.simulators._stitching import RESUME_STATE_FORMAT_VERSION, OverlapAddStitcher

WINDOW_SIZE = 8
OVERLAP_SIZE = 4
STRIDE = WINDOW_SIZE - OVERLAP_SIZE
SAMPLING_FREQUENCY = 256.0


def _write_flat_psd(path: Path, *, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, 129)
    np.savetxt(path, np.column_stack((frequencies, np.full_like(frequencies, value))))
    return path


def _make_stitcher() -> OverlapAddStitcher:
    """Return a small stitcher over one detector."""
    return OverlapAddStitcher(["H1"], window_size=WINDOW_SIZE, overlap_size=OVERLAP_SIZE)


def _make_driven_stream(
    seed: int,
) -> tuple[np.random.Generator, list[np.ndarray], Callable[[], dict[str, np.ndarray]]]:
    """Return an RNG, its drawn-chunk log, and a chunk generator driven by it."""
    rng = np.random.default_rng(seed)
    drawn: list[np.ndarray] = []

    def generator() -> dict[str, np.ndarray]:
        chunk = rng.standard_normal(WINDOW_SIZE)
        drawn.append(chunk.copy())
        return {"H1": chunk}

    return rng, drawn, generator


def test_stitcher_export_state_is_bit_generator_state_and_counter_only() -> None:
    """The resume payload carries the counter and RNG state, never cached strain."""
    rng, drawn, generator = _make_driven_stream(101)
    stitcher = _make_stitcher()
    stitcher.bind_rngs({"H1": rng})
    for _ in range(3):
        stitcher.stitch(n_samples=STRIDE, chunk_generator=generator)

    state = stitcher.export_state()
    assert state["format_version"] == RESUME_STATE_FORMAT_VERSION
    assert state["chunk_counter"] == len(drawn)
    assert state["detectors"] == ["H1"]
    assert "previous_strain" not in state
    assert set(state["rng_state"]) == {"H1"}
    assert not any(isinstance(value, np.ndarray) for value in state.values())
    assert not any(isinstance(value, np.ndarray) for value in state["rng_state"].values())


def test_stitcher_resume_regenerates_the_previous_chunk() -> None:
    """Importing state reproduces the last raw chunk from the bit-generator state."""
    rng_a, _, generator_a = _make_driven_stream(202)
    uninterrupted = _make_stitcher()
    uninterrupted.bind_rngs({"H1": rng_a})
    outputs = [uninterrupted.stitch(n_samples=STRIDE, chunk_generator=generator_a)["H1"] for _ in range(5)]

    rng_b, drawn_b, generator_b = _make_driven_stream(202)
    stopped = _make_stitcher()
    stopped.bind_rngs({"H1": rng_b})
    for _ in range(3):
        stopped.stitch(n_samples=STRIDE, chunk_generator=generator_b)
    last_chunk = drawn_b[-1].copy()
    chunks_before_resume = len(drawn_b)
    state = stopped.export_state()

    fresh = _make_stitcher()
    fresh.bind_rngs({"H1": rng_b})
    fresh.import_state(state, generator_b)

    np.testing.assert_array_equal(fresh.previous_strain["H1"], last_chunk)
    assert len(drawn_b) == chunks_before_resume + 1
    np.testing.assert_array_equal(drawn_b[-1], last_chunk)

    continuation = [fresh.stitch(n_samples=STRIDE, chunk_generator=generator_b)["H1"] for _ in range(2)]
    for expected, actual in zip(outputs[3:], continuation, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_stitcher_import_state_requires_bound_generators() -> None:
    """A stitcher without bound RNGs cannot import resume state."""
    state = {
        "format_version": RESUME_STATE_FORMAT_VERSION,
        "detectors": ["H1"],
        "chunk_counter": 2,
        "rng_state": {"H1": np.random.default_rng(1).bit_generator.state},
    }
    stitcher = _make_stitcher()
    with pytest.raises(ValueError, match="RNGs must be bound"):
        stitcher.import_state(state, lambda: {"H1": np.zeros(WINDOW_SIZE)})


def test_stitcher_import_state_rejects_mismatched_detectors() -> None:
    """Resume state from a different detector set is refused."""
    state = {
        "format_version": RESUME_STATE_FORMAT_VERSION,
        "detectors": ["L1"],
        "chunk_counter": 1,
        "rng_state": {"H1": np.random.default_rng(1).bit_generator.state},
    }
    holder = {"rng": np.random.default_rng(4)}
    stitcher = _make_stitcher()
    stitcher.bind_rngs({"H1": holder["rng"]})
    with pytest.raises(ValueError, match="detectors do not match"):
        stitcher.import_state(state, lambda: {"H1": np.zeros(WINDOW_SIZE)})


def test_stitcher_import_state_rejects_unknown_format_version() -> None:
    """An incompatible payload layout is rejected instead of misread."""
    state = {"format_version": RESUME_STATE_FORMAT_VERSION + 1, "detectors": ["H1"], "chunk_counter": 0}
    holder = {"rng": np.random.default_rng(4)}
    stitcher = _make_stitcher()
    stitcher.bind_rngs({"H1": holder["rng"]})
    with pytest.raises(ValueError, match="format version"):
        stitcher.import_state(state, lambda: {"H1": np.zeros(WINDOW_SIZE)})


def test_stitcher_bind_rngs_validates_detectors() -> None:
    """Binding generators with the wrong detector keys is refused."""
    stitcher = _make_stitcher()
    with pytest.raises(ValueError, match="exactly match"):
        stitcher.bind_rngs({"L1": np.random.default_rng(1)})


def test_stitcher_import_state_of_an_empty_stream_resets() -> None:
    """A zero-counter payload with no generator state resets the stitcher."""
    rng, _, generator = _make_driven_stream(11)
    stitcher = _make_stitcher()
    stitcher.bind_rngs({"H1": rng})
    stitcher.stitch(n_samples=STRIDE, chunk_generator=generator)

    stitcher.import_state(
        {
            "format_version": RESUME_STATE_FORMAT_VERSION,
            "detectors": ["H1"],
            "chunk_counter": 0,
            "rng_state": None,
        },
        generator,
    )

    assert stitcher.chunk_counter == 0
    assert stitcher.previous_strain == {}


def test_stitcher_import_state_requires_a_state_for_every_detector() -> None:
    """A non-empty payload without a generator state for each detector is refused."""
    rng, _, generator = _make_driven_stream(12)
    stitcher = _make_stitcher()
    stitcher.bind_rngs({"H1": rng})
    state = {
        "format_version": RESUME_STATE_FORMAT_VERSION,
        "detectors": ["H1"],
        "chunk_counter": 2,
        "rng_state": {"L1": np.random.default_rng(1).bit_generator.state},
    }
    with pytest.raises(ValueError, match="per detector"):
        stitcher.import_state(state, generator)


def test_colored_stream_resume_is_bit_identical(tmp_path: Path) -> None:
    """A colored stream stopped mid-way and resumed matches an uninterrupted run.

    The resume boundary is placed after three of five chunks, so the stitcher
    must regenerate a real previous window rather than start a fresh stream.
    """
    psd_path = _write_flat_psd(tmp_path / "resume_psd.txt")

    def make() -> ColoredNoiseSimulator:
        return ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            window_duration=1.0,
            seed=77,
        )

    uninterrupted = make()
    full = [
        uninterrupted.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=77 if index == 0 else None)["H1"]
        for index in range(5)
    ]

    stopped = make()
    head = [
        stopped.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=77 if index == 0 else None)["H1"] for index in range(3)
    ]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    assert resumed.previous_strain["H1"].shape == (resumed._stitcher.window_size,)
    tail = [resumed.generate(0.5, SAMPLING_FREQUENCY, ["H1"])["H1"] for _ in range(2)]

    for expected, actual in zip(full, head + tail, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_colored_scheduled_stream_resume_is_bit_identical(tmp_path: Path) -> None:
    """Resuming across a schedule change regenerates the previous window with its own spectrum."""
    first_anchor = _write_flat_psd(tmp_path / "schedule_a.txt", value=2.0e-3)
    second_anchor = _write_flat_psd(tmp_path / "schedule_b.txt", value=8.0e-3)
    schedule = [(0.0, first_anchor), (1.0, second_anchor)]

    def make() -> ColoredNoiseSimulator:
        return ColoredNoiseSimulator(
            psd_schedule=schedule,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            window_duration=1.0,
            seed=101,
        )

    uninterrupted = make()
    full = [
        uninterrupted.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=101 if index == 0 else None)["H1"]
        for index in range(6)
    ]

    stopped = make()
    head = [
        stopped.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=101 if index == 0 else None)["H1"] for index in range(4)
    ]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    assert resumed.previous_strain["H1"].shape == (resumed._stitcher.window_size,)
    tail = [resumed.generate(0.5, SAMPLING_FREQUENCY, ["H1"])["H1"] for _ in range(2)]

    for expected, actual in zip(full, head + tail, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_colored_import_state_requires_a_configured_simulator(tmp_path: Path) -> None:
    """Importing state into a simulator with no configured PSD anchors is refused."""
    psd_path = _write_flat_psd(tmp_path / "guard_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        window_duration=1.0,
        seed=3,
    )
    simulator._psd_anchors = []
    with pytest.raises(ValueError, match="must be configured"):
        simulator.import_state({})


def test_colored_export_state_contains_no_cached_strain(tmp_path: Path) -> None:
    """The colored snapshot is metadata-sized, not a cached window."""
    psd_path = _write_flat_psd(tmp_path / "metadata_psd.txt")
    simulator = ColoredNoiseSimulator(
        psd_file=psd_path,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        window_duration=1.0,
        seed=11,
    )
    simulator.generate(0.5, SAMPLING_FREQUENCY, ["H1"], seed=11)
    snapshot = simulator.export_state()

    assert "previous_strain" not in snapshot["stitcher"]
    window_bytes = simulator._stitcher.window_size * np.dtype(float).itemsize
    assert len(pickle.dumps(snapshot)) < window_bytes
