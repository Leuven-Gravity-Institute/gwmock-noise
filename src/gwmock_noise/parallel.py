"""Parallel adapter for independent-detector noise simulators."""

from __future__ import annotations

import os
import pickle  # nosec: B403
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Literal

import numpy as np

from gwmock_noise.simulators.protocol import NoiseSimulator

ExecutorBackend = Literal["auto", "process", "thread"]
_CPU_BOUND_IMPLEMENTATIONS = {"autoregressive", "colored"}
_UNSUPPORTED_IMPLEMENTATIONS = {"correlated", "correlated_autoregressive"}


def _derive_seeds(seed: int | None, detectors: list[str]) -> dict[str, int | None]:
    """Return deterministic per-detector seeds."""
    if len(set(detectors)) != len(detectors):
        raise ValueError("detectors must not contain duplicate names.")
    if seed is None:
        return dict.fromkeys(detectors)

    rng = np.random.default_rng(seed)
    upper_bound = np.iinfo(np.int32).max
    return {detector: int(rng.integers(0, upper_bound)) for detector in detectors}


def _worker_generate(
    base_factory: Callable[[], NoiseSimulator],
    duration: float,
    sampling_frequency: float,
    detectors: list[str],
    seeds: dict[str, int | None],
) -> dict[str, np.ndarray]:
    """Generate independent detector batches inside one worker."""
    simulator = base_factory()
    results: dict[str, np.ndarray] = {}
    for detector in detectors:
        result = simulator.generate(
            duration,
            sampling_frequency,
            [detector],
            seed=seeds[detector],
        )
        results[detector] = np.asarray(result[detector])
    return results


def _worker_generate_with_instances(
    simulators: dict[str, NoiseSimulator],
    duration: float,
    sampling_frequency: float,
    detectors: list[str],
    seeds: dict[str, int | None],
) -> dict[str, np.ndarray]:
    """Generate detector batches using persistent simulator instances."""
    results: dict[str, np.ndarray] = {}
    for detector in detectors:
        simulator = simulators[detector]
        result = simulator.generate(
            duration,
            sampling_frequency,
            [detector],
            seed=seeds[detector],
        )
        results[detector] = np.asarray(result[detector])
    return results


class ParallelAdapter:
    """Wrap an independent-detector simulator factory with parallel execution."""

    def __init__(
        self,
        base_factory: Callable[[], NoiseSimulator],
        *,
        max_workers: int | None = None,
        backend: ExecutorBackend = "auto",
    ) -> None:
        """Store the simulator factory and executor preferences."""
        if max_workers is not None and max_workers < 1:
            raise ValueError("max_workers must be greater than zero.")
        if backend not in {"auto", "process", "thread"}:
            raise ValueError("backend must be one of {'auto', 'process', 'thread'}.")

        self.base_factory = base_factory
        self.max_workers = max_workers
        self.backend = backend

        self.duration = 4.0
        self.sampling_frequency = 4096.0
        self.detectors = ["H1", "L1"]
        self.seed: int | None = None

        self._preview_metadata: dict[str, Any] | None = None
        self._resolved_backend: ExecutorBackend | None = None
        self._resolved_max_workers: int | None = None
        self._worker_simulators: dict[str, NoiseSimulator] = {}

    def _preview_simulator(self) -> NoiseSimulator:
        """Instantiate one simulator for validation and metadata inspection."""
        simulator = self.base_factory()
        if not isinstance(simulator, NoiseSimulator):
            raise TypeError("base_factory must construct a NoiseSimulator-compatible object.")
        self._preview_metadata = dict(simulator.metadata)
        self.duration = simulator.duration
        self.sampling_frequency = simulator.sampling_frequency
        self.detectors = list(simulator.detectors)
        self.seed = simulator.seed
        return simulator

    def _metadata(self) -> dict[str, Any]:
        """Return cached base metadata, instantiating a preview simulator if needed."""
        if self._preview_metadata is None:
            self._preview_simulator()
        if self._preview_metadata is None:
            raise RuntimeError("Preview metadata could not be initialized.")
        return dict(self._preview_metadata)

    def _validate_supported(self) -> None:
        """Reject correlated simulators that cannot be split per detector."""
        metadata = self._metadata()
        implementations = {
            metadata.get("implementation"),
            metadata.get("base_implementation"),
        }
        if any(implementation in _UNSUPPORTED_IMPLEMENTATIONS for implementation in implementations):
            raise ValueError(
                "ParallelAdapter only supports independent-detector simulators; correlated simulators are unsupported."
            )

    def _resolve_max_workers(self, detectors: list[str]) -> int:
        """Resolve the worker count for one generate call."""
        available_cpus = os.cpu_count() or 1
        requested = self.max_workers if self.max_workers is not None else available_cpus
        return max(1, min(len(detectors), requested))

    def _factory_is_picklable(self) -> bool:
        """Return whether the base factory can be shipped to worker processes."""
        try:
            pickle.dumps(self.base_factory)
        except (AttributeError, pickle.PickleError, TypeError):
            return False
        return True

    def _resolve_backend(self) -> ExecutorBackend:
        """Pick a backend using the configured preference and runtime constraints."""
        if self.backend in {"process", "thread"}:
            return self.backend

        implementation = self._metadata().get("implementation")
        if not self._factory_is_picklable():
            return "thread"
        if implementation in _CPU_BOUND_IMPLEMENTATIONS:
            return "process"
        return "thread"

    def _build_assignments(self, detectors: list[str], n_workers: int) -> list[list[str]]:
        """Distribute detectors across workers in stable round-robin order."""
        assignments = [[] for _ in range(n_workers)]
        for index, detector in enumerate(detectors):
            assignments[index % n_workers].append(detector)
        return [assignment for assignment in assignments if assignment]

    def _executor_type(self, backend: ExecutorBackend) -> type[Executor]:
        """Map backend labels to executor classes."""
        if backend == "process":
            return ProcessPoolExecutor
        return ThreadPoolExecutor

    def _get_worker_simulator(self, detector: str) -> NoiseSimulator:
        """Create and cache a persistent simulator for one detector."""
        simulator = self._worker_simulators.get(detector)
        if simulator is not None:
            return simulator

        simulator = self.base_factory()
        if not isinstance(simulator, NoiseSimulator):
            raise TypeError("base_factory must construct a NoiseSimulator-compatible object.")
        self._worker_simulators[detector] = simulator
        return simulator

    def _run_assignments_persistent(
        self,
        *,
        executor_type: type[Executor] | None,
        duration: float,
        sampling_frequency: float,
        seeds: dict[str, int | None],
        assignments: list[list[str]],
    ) -> dict[str, np.ndarray]:
        """Execute detector assignments using cached simulator instances."""
        simulators = {detector: self._get_worker_simulator(detector) for detector in seeds}
        if executor_type is None:
            return _worker_generate_with_instances(simulators, duration, sampling_frequency, assignments[0], seeds)

        futures = {}
        with executor_type(max_workers=len(assignments)) as executor:
            for assignment in assignments:
                futures[tuple(assignment)] = executor.submit(
                    _worker_generate_with_instances,
                    simulators,
                    duration,
                    sampling_frequency,
                    assignment,
                    seeds,
                )

        combined: dict[str, np.ndarray] = {}
        for _assignment, future in futures.items():
            combined.update(future.result())
        return combined

    def _run_assignments(
        self,
        *,
        executor_type: type[Executor] | None,
        duration: float,
        sampling_frequency: float,
        seeds: dict[str, int | None],
        assignments: list[list[str]],
    ) -> dict[str, np.ndarray]:
        """Execute detector assignments either directly or through an executor."""
        if executor_type is None:
            results = _worker_generate(
                self.base_factory,
                duration,
                sampling_frequency,
                assignments[0],
                seeds,
            )
            return results

        futures = {}
        with executor_type(max_workers=len(assignments)) as executor:
            for assignment in assignments:
                futures[tuple(assignment)] = executor.submit(
                    _worker_generate,
                    self.base_factory,
                    duration,
                    sampling_frequency,
                    assignment,
                    seeds,
                )

        combined: dict[str, np.ndarray] = {}
        for _assignment, future in futures.items():
            combined.update(future.result())
        return combined

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate per-detector strain arrays in parallel."""
        runtime_detectors = list(detectors)
        if not runtime_detectors:
            raise ValueError("detectors must contain at least one detector.")
        if len(set(runtime_detectors)) != len(runtime_detectors):
            raise ValueError("detectors must not contain duplicate names.")

        self._validate_supported()
        resolved_workers = self._resolve_max_workers(runtime_detectors)
        resolved_backend = self._resolve_backend()
        seeds = _derive_seeds(seed, runtime_detectors)
        assignments = self._build_assignments(runtime_detectors, resolved_workers)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        self.seed = seed
        self._resolved_backend = resolved_backend
        self._resolved_max_workers = resolved_workers

        # Persistent simulator state cannot be preserved with one-off process pools.
        if resolved_backend == "process":
            resolved_backend = "thread"
            self._resolved_backend = "thread"

        executor_type: type[Executor] | None = None if resolved_workers == 1 else self._executor_type(resolved_backend)
        results = self._run_assignments_persistent(
            executor_type=executor_type,
            duration=duration,
            sampling_frequency=sampling_frequency,
            seeds=seeds,
            assignments=assignments,
        )
        return {detector: results[detector] for detector in runtime_detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield parallel-generated chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    def reset(self) -> None:
        """Clear cached worker simulators, resetting continuity across workers."""
        self._worker_simulators.clear()

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata describing the wrapped simulator and executor."""
        base_metadata = self._metadata()
        return base_metadata | {
            "implementation": "parallel",
            "base_implementation": base_metadata.get("implementation"),
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
            "parallel": {
                "backend": self._resolved_backend or self.backend,
                "max_workers": self.max_workers,
                "resolved_max_workers": self._resolved_max_workers,
            },
        }
