"""Streaming helpers for protocol-compatible simulators."""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import numpy as np

from gwmock_noise.simulators.protocol import NoiseSimulator


def open_stream(
    simulator: NoiseSimulator,
    *,
    chunk_duration: float,
    sampling_frequency: float,
    detectors: Sequence[str],
    seed: int | None = None,
) -> Iterator[dict[str, np.ndarray]]:
    """Open a stateful chunk stream on a protocol-compatible simulator."""
    if not isinstance(simulator, NoiseSimulator):
        raise TypeError("simulator must satisfy the NoiseSimulator protocol.")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be greater than zero.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be greater than zero.")
    if isinstance(detectors, str):
        raise TypeError("detectors must be a sequence of detector names, not a single string.")
    if round(chunk_duration * sampling_frequency) < 1:
        raise ValueError("chunk_duration and sampling_frequency must produce at least one sample.")

    runtime_detectors = list(detectors)
    if not runtime_detectors:
        raise ValueError("detectors must contain at least one detector.")

    return simulator.generate_stream(
        chunk_duration=chunk_duration,
        sampling_frequency=sampling_frequency,
        detectors=runtime_detectors,
        seed=seed,
    )


def take(
    stream: Iterator[dict[str, np.ndarray]],
    total_duration: float,
    chunk_duration: float,
    sampling_frequency: float,
) -> dict[str, np.ndarray]:
    """Collect stream chunks up to ``total_duration`` seconds."""
    if total_duration <= 0:
        raise ValueError("total_duration must be greater than zero.")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be greater than zero.")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be greater than zero.")

    n_samples = round(total_duration * sampling_frequency)
    if n_samples < 1:
        raise ValueError("total_duration and sampling_frequency must produce at least one sample.")
    chunk_samples = round(chunk_duration * sampling_frequency)
    if chunk_samples < 1:
        raise ValueError("chunk_duration and sampling_frequency must produce at least one sample.")

    n_chunks = math.ceil(total_duration / chunk_duration)
    collected: dict[str, list[np.ndarray]] | None = None

    for _ in range(n_chunks):
        try:
            chunk = next(stream)
        except StopIteration as error:
            raise ValueError("stream ended before total_duration could be collected.") from error

        if collected is None:
            collected = {detector: [np.asarray(strain)] for detector, strain in chunk.items()}
            continue

        if set(chunk) != set(collected):
            raise ValueError("stream chunks must contain a consistent detector set.")

        for detector, strain in chunk.items():
            collected[detector].append(np.asarray(strain))

    if collected is None:
        return {}

    concatenated = {detector: np.concatenate(chunks) for detector, chunks in collected.items()}
    if any(strain.shape[0] < n_samples for strain in concatenated.values()):
        raise ValueError("stream ended before total_duration could be collected.")
    return {detector: strain[:n_samples] for detector, strain in concatenated.items()}
