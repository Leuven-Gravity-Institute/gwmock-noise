"""Tests for the streaming simulator interface."""

from __future__ import annotations

import gc
import inspect
import tracemalloc
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise import take as top_level_take
from gwmock_noise.simulators import ARNoiseSimulator, ColoredNoiseSimulator, take


def _write_psd_file(path: Path, *, sampling_frequency: float = 256.0, value: float = 2.0e-3) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _peak_bytes_for_stream(stream, n_chunks: int) -> int:
    """Return traced peak memory while consuming ``n_chunks`` streamed chunks."""
    gc.collect()
    tracemalloc.start()
    try:
        for _ in range(n_chunks):
            next(stream)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak


def test_take_is_importable_from_top_level_package() -> None:
    """take() is re-exported from the top-level package."""
    assert top_level_take is take


def test_colored_generate_stream_is_a_generator_function() -> None:
    """ColoredNoiseSimulator.generate_stream exposes a real generator method."""
    assert inspect.isgeneratorfunction(ColoredNoiseSimulator.generate_stream)


def test_ar_generate_stream_is_a_generator_function() -> None:
    """ARNoiseSimulator.generate_stream exposes a real generator method."""
    assert inspect.isgeneratorfunction(ARNoiseSimulator.generate_stream)


def test_colored_stream_is_lazy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a stream does not precompute a chunk before iteration starts."""
    psd_path = _write_psd_file(tmp_path / "lazy_colored_psd.txt")
    simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, seed=7)
    calls = 0
    original_generate = simulator.generate

    def counted_generate(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_generate(*args, **kwargs)

    monkeypatch.setattr(simulator, "generate", counted_generate)

    stream = simulator.generate_stream(4.0, 256.0, ["H1"], seed=11)
    assert calls == 0

    chunk = next(stream)
    assert calls == 1
    assert chunk["H1"].shape == (1024,)


def test_colored_stream_matches_single_generate_call(tmp_path: Path) -> None:
    """Collecting streamed colored chunks reproduces one long realization."""
    psd_path = _write_psd_file(tmp_path / "colored_stream_psd.txt")
    long_simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0)
    stream_simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0)

    expected = long_simulator.generate(12.0, 256.0, ["H1"], seed=123)["H1"]
    actual = take(
        stream_simulator.generate_stream(4.0, 256.0, ["H1"], seed=123),
        total_duration=12.0,
        chunk_duration=4.0,
        sampling_frequency=256.0,
    )["H1"]

    np.testing.assert_allclose(actual, expected)


def test_ar_stream_matches_single_generate_call(tmp_path: Path) -> None:
    """Collecting streamed AR chunks reproduces one long realization."""
    psd_path = _write_psd_file(tmp_path / "ar_stream_psd.txt")
    long_simulator = ARNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, order=32)
    stream_simulator = ARNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, order=32)

    expected = long_simulator.generate(12.0, 256.0, ["H1"], seed=123)["H1"]
    actual = take(
        stream_simulator.generate_stream(4.0, 256.0, ["H1"], seed=123),
        total_duration=12.0,
        chunk_duration=4.0,
        sampling_frequency=256.0,
    )["H1"]

    np.testing.assert_allclose(actual, expected)


def test_take_trims_the_final_stream_chunk(tmp_path: Path) -> None:
    """take() truncates the last chunk back to the requested sample count."""
    psd_path = _write_psd_file(tmp_path / "trimmed_stream_psd.txt")
    simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0)

    result = take(
        simulator.generate_stream(4.0, 256.0, ["H1"], seed=5),
        total_duration=10.0,
        chunk_duration=4.0,
        sampling_frequency=256.0,
    )

    assert result["H1"].shape == (2560,)


def test_colored_stream_peak_memory_is_stable_over_longer_runs(tmp_path: Path) -> None:
    """Streaming chunk generation keeps peak memory roughly independent of run length."""
    psd_path = _write_psd_file(tmp_path / "memory_stream_psd.txt")

    short_simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, seed=17)
    short_peak = _peak_bytes_for_stream(short_simulator.generate_stream(4.0, 256.0, ["H1"]), n_chunks=4)

    long_simulator = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=256.0, seed=17)
    long_peak = _peak_bytes_for_stream(long_simulator.generate_stream(4.0, 256.0, ["H1"]), n_chunks=32)

    assert long_peak <= max(short_peak * 2, short_peak + 200_000)
