"""Tests for the parallel detector adapter."""

from __future__ import annotations

import functools
import inspect
import os
import time
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.parallel import ParallelAdapter, _derive_seeds
from gwmock_noise.simulators import ARNoiseSimulator, ColoredNoiseSimulator, CorrelatedNoiseSimulator, NoiseSimulator

FLAT_CSD = 8.0e-4
FLAT_PSD = 2.0e-3


def _write_psd_file(path: Path, *, sampling_frequency: float = 256.0, value: float = FLAT_PSD) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _write_csd_file(path: Path, *, sampling_frequency: float = 256.0, value: float = FLAT_CSD) -> Path:
    """Write a flat real CSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
    values = np.full(frequencies.shape, value, dtype=np.complex128)
    np.save(path, np.column_stack((frequencies, values)))
    return path.with_suffix(".npy")


class _MutableStateSimulator:
    """Simple simulator that reveals whether state is shared across workers."""

    def __init__(self) -> None:
        self.duration = 1.0
        self.sampling_frequency = 8.0
        self.detectors = ["H1", "L1"]
        self.seed = None
        self._calls = 0

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        self._calls += 1
        n_samples = round(duration * sampling_frequency)
        return {detector: np.full(n_samples, self._calls, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, str]:
        return {"implementation": "mutable_state_test"}


@pytest.mark.parametrize(
    ("factory_builder", "detectors"),
    [
        (
            lambda tmp_path: functools.partial(
                ColoredNoiseSimulator,
                psd_file=_write_psd_file(tmp_path / "colored_parallel_psd.txt"),
                detectors=["H1", "L1"],
                sampling_frequency=256.0,
            ),
            ["H1", "L1"],
        ),
        (
            lambda tmp_path: functools.partial(
                ARNoiseSimulator,
                psd_file=_write_psd_file(tmp_path / "ar_parallel_psd.txt"),
                detectors=["H1", "L1"],
                sampling_frequency=256.0,
                order=16,
            ),
            ["H1", "L1"],
        ),
    ],
)
def test_parallel_output_matches_single_worker_output(
    tmp_path: Path,
    factory_builder,
    detectors: list[str],
) -> None:
    """ParallelAdapter output is deterministic across worker counts."""
    factory = factory_builder(tmp_path)
    sequential = ParallelAdapter(factory, max_workers=1)
    parallel = ParallelAdapter(factory, max_workers=2, backend="process")

    expected = sequential.generate(4.0, 256.0, detectors, seed=1234)
    actual = parallel.generate(4.0, 256.0, detectors, seed=1234)

    for detector in detectors:
        np.testing.assert_allclose(actual[detector], expected[detector])


def test_parallel_adapter_satisfies_noise_protocol(tmp_path: Path) -> None:
    """ParallelAdapter satisfies the runtime-checkable protocol."""
    factory = functools.partial(
        ColoredNoiseSimulator,
        psd_file=_write_psd_file(tmp_path / "protocol_parallel_psd.txt"),
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
    )
    adapter = ParallelAdapter(factory, max_workers=2)

    assert isinstance(adapter, NoiseSimulator)
    assert inspect.isgeneratorfunction(ParallelAdapter.generate_stream)


def test_parallel_adapter_falls_back_to_threads_for_unpicklable_factory(tmp_path: Path) -> None:
    """Auto backend uses threads when the factory cannot be pickled."""
    psd_path = _write_psd_file(tmp_path / "thread_fallback_psd.txt")

    def local_factory() -> ColoredNoiseSimulator:
        return ColoredNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1", "L1"],
            sampling_frequency=256.0,
        )

    adapter = ParallelAdapter(local_factory, max_workers=2, backend="auto")
    result = adapter.generate(4.0, 256.0, ["H1", "L1"], seed=7)

    assert adapter.metadata["parallel"]["backend"] == "thread"
    assert result["H1"].shape == (1024,)
    assert result["L1"].shape == (1024,)


def test_parallel_adapter_uses_isolated_worker_state() -> None:
    """Each worker receives a fresh simulator instance rather than shared mutable state."""
    adapter = ParallelAdapter(_MutableStateSimulator, max_workers=2, backend="thread")
    result = adapter.generate(1.0, 8.0, ["H1", "L1"], seed=3)

    np.testing.assert_allclose(result["H1"], np.ones(8))
    np.testing.assert_allclose(result["L1"], np.ones(8))


def test_parallel_adapter_rejects_correlated_simulators(tmp_path: Path) -> None:
    """Correlated simulators are out of scope because detectors are coupled."""
    psd_h1 = _write_psd_file(tmp_path / "h1_psd.txt")
    psd_l1 = _write_psd_file(tmp_path / "l1_psd.txt")
    csd_h1_l1 = _write_csd_file(tmp_path / "h1_l1_csd.npy")
    factory = functools.partial(
        CorrelatedNoiseSimulator,
        psd_files={"H1": psd_h1, "L1": psd_l1},
        csd_files={("H1", "L1"): csd_h1_l1},
        detectors=["H1", "L1"],
        sampling_frequency=256.0,
    )
    adapter = ParallelAdapter(factory, max_workers=2)

    with pytest.raises(ValueError, match="independent-detector simulators"):
        adapter.generate(4.0, 256.0, ["H1", "L1"], seed=2)


@pytest.mark.skipif((os.cpu_count() or 1) < 4, reason="Benchmark is only interesting on 4-core machines.")
def test_parallel_adapter_benchmark_smoke(tmp_path: Path) -> None:
    """Collect a non-strict timing ratio for a four-detector colored-noise run."""
    detectors = ["H1", "L1", "V1", "K1"]
    factory = functools.partial(
        ColoredNoiseSimulator,
        psd_file=_write_psd_file(tmp_path / "benchmark_parallel_psd.txt"),
        detectors=detectors,
        sampling_frequency=256.0,
    )
    single = ParallelAdapter(factory, max_workers=1)
    parallel = ParallelAdapter(factory, max_workers=4, backend="process")

    single_start = time.perf_counter()
    single.generate(4.0, 256.0, detectors, seed=99)
    single_elapsed = time.perf_counter() - single_start

    parallel_start = time.perf_counter()
    parallel.generate(4.0, 256.0, detectors, seed=99)
    parallel_elapsed = time.perf_counter() - parallel_start

    assert single_elapsed > 0.0
    assert parallel_elapsed > 0.0


def test_derive_seeds_validates_detectors_and_none_seed() -> None:
    """Seed derivation handles duplicate names and seed=None passthrough."""
    with pytest.raises(ValueError, match="must not contain duplicate names"):
        _derive_seeds(1, ["H1", "H1"])

    assert _derive_seeds(None, ["H1", "L1"]) == {"H1": None, "L1": None}


def test_parallel_adapter_input_validation_and_metadata(tmp_path: Path) -> None:
    """Adapter validates init/generate arguments and reports resolved metadata."""
    with pytest.raises(ValueError, match="max_workers must be greater than zero"):
        ParallelAdapter(_MutableStateSimulator, max_workers=0)
    with pytest.raises(ValueError, match="backend must be one of"):
        ParallelAdapter(_MutableStateSimulator, backend="bogus")  # type: ignore[arg-type]

    adapter = ParallelAdapter(_MutableStateSimulator, max_workers=1, backend="thread")
    with pytest.raises(ValueError, match="at least one detector"):
        adapter.generate(1.0, 8.0, [], seed=1)
    with pytest.raises(ValueError, match="must not contain duplicate names"):
        adapter.generate(1.0, 8.0, ["H1", "H1"], seed=1)

    result = adapter.generate(1.0, 8.0, ["H1"], seed=1)
    assert result["H1"].shape == (8,)
    meta = adapter.metadata
    assert meta["implementation"] == "parallel"
    assert meta["parallel"]["backend"] == "thread"
    assert meta["parallel"]["resolved_max_workers"] == 1
