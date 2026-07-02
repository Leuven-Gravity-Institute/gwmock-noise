"""Acceptance tests for the public streaming continuation contract."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from gwmock_noise import ColoredNoiseSimulator, CorrelatedNoiseSimulator, NoiseSimulator, open_stream

FLAT_PSD = 2.0e-3
FLAT_CSD = 8.0e-4


def _write_psd_file(path: Path, *, sampling_frequency: float = 256.0, value: float = FLAT_PSD) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
    values = np.full_like(frequencies, value)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def _write_csd_file(path: Path, *, sampling_frequency: float = 256.0, value: complex = FLAT_CSD) -> Path:
    """Write a flat complex CSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, 1025)
    values = np.full(frequencies.shape, value, dtype=np.complex128)
    np.save(path, np.column_stack((frequencies, values)))
    return path


def _collect_chunks(
    stream: Iterator[dict[str, np.ndarray]],
    *,
    detectors: list[str],
    n_chunks: int,
) -> dict[str, np.ndarray]:
    """Collect and concatenate a fixed number of chunks from a stream."""
    collected = {detector: [] for detector in detectors}
    for _ in range(n_chunks):
        chunk = next(stream)
        for detector in detectors:
            collected[detector].append(chunk[detector])
    return {detector: np.concatenate(chunks) for detector, chunks in collected.items()}


def test_open_stream_chunked_colored_output_matches_single_shot_generate(tmp_path: Path) -> None:
    """Chunked colored output is byte-identical to a single seeded realization."""
    detectors = ["H1"]
    psd_path = _write_psd_file(tmp_path / "colored_psd.txt")
    # Explicit: this test compares a single 12 s generate() call against three
    # 4 s streamed chunks byte-for-byte. The 64 s default window makes the
    # window far larger than either request size, which changes how many
    # internal chunks each path draws and breaks exact equality. Pin a
    # smaller window here, where the two paths were validated to match.
    long_simulator = ColoredNoiseSimulator(
        psd_file=psd_path, detectors=detectors, sampling_frequency=256.0, window_duration=4.0
    )
    stream_simulator = ColoredNoiseSimulator(
        psd_file=psd_path, detectors=detectors, sampling_frequency=256.0, window_duration=4.0
    )

    expected = long_simulator.generate(12.0, 256.0, detectors, seed=123)
    actual = _collect_chunks(
        open_stream(
            stream_simulator,
            chunk_duration=4.0,
            sampling_frequency=256.0,
            detectors=detectors,
            seed=123,
        ),
        detectors=detectors,
        n_chunks=3,
    )

    for detector in detectors:
        np.testing.assert_array_equal(actual[detector], expected[detector])


def test_open_stream_chunked_correlated_output_matches_single_shot_generate(tmp_path: Path) -> None:
    """Chunked correlated output is byte-identical to a single seeded realization."""
    detectors = ["H1", "L1"]
    psd_h1 = _write_psd_file(tmp_path / "h1_psd.txt")
    psd_l1 = _write_psd_file(tmp_path / "l1_psd.txt")
    csd_h1_l1 = _write_csd_file(tmp_path / "h1_l1_csd.npy")
    psd_files = {"H1": psd_h1, "L1": psd_l1}
    csd_files = {("H1", "L1"): csd_h1_l1}

    # Explicit: see the colored-noise version of this test above for why the
    # 64 s default window breaks exact single-shot/streamed equality.
    long_simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        window_duration=4.0,
    )
    stream_simulator = CorrelatedNoiseSimulator(
        psd_files=psd_files,
        csd_files=csd_files,
        detectors=detectors,
        sampling_frequency=256.0,
        window_duration=4.0,
    )

    expected = long_simulator.generate(12.0, 256.0, detectors, seed=123)
    actual = _collect_chunks(
        open_stream(
            stream_simulator,
            chunk_duration=4.0,
            sampling_frequency=256.0,
            detectors=detectors,
            seed=123,
        ),
        detectors=detectors,
        n_chunks=3,
    )

    for detector in detectors:
        np.testing.assert_array_equal(actual[detector], expected[detector])


def test_open_stream_accepts_custom_protocol_simulators() -> None:
    """Protocol-conformant custom simulators can be consumed without internals."""

    class CustomNoiseSimulator:
        """Tiny duck-typed simulator used to exercise the public helper."""

        def __init__(self) -> None:
            self.duration = 1.0
            self.sampling_frequency = 8.0
            self.detectors = ["H1"]
            self.seed = None
            self.stream_calls: list[dict[str, Any]] = []
            self._offset = 0

        def generate(
            self,
            duration: float,
            sampling_frequency: float,
            detectors: list[str],
            seed: int | None = None,
        ) -> dict[str, np.ndarray]:
            n_samples = round(duration * sampling_frequency)
            self.duration = duration
            self.sampling_frequency = sampling_frequency
            self.detectors = list(detectors)
            self.seed = seed
            return {detector: np.arange(n_samples, dtype=float) for detector in detectors}

        def generate_stream(
            self,
            chunk_duration: float,
            sampling_frequency: float,
            detectors: list[str],
            seed: int | None = None,
        ) -> Iterator[dict[str, np.ndarray]]:
            self.stream_calls.append(
                {
                    "chunk_duration": chunk_duration,
                    "sampling_frequency": sampling_frequency,
                    "detectors": list(detectors),
                    "seed": seed,
                }
            )
            n_samples = round(chunk_duration * sampling_frequency)
            while True:
                start = self._offset
                stop = start + n_samples
                self._offset = stop
                yield {detector: np.arange(start, stop, dtype=float) for detector in detectors}

        @property
        def metadata(self) -> dict[str, Any]:
            return {"implementation": "custom"}

    simulator: NoiseSimulator = CustomNoiseSimulator()
    stream = open_stream(
        simulator,
        chunk_duration=0.5,
        sampling_frequency=8.0,
        detectors=("H1", "L1"),
        seed=17,
    )

    first = next(stream)
    second = next(stream)

    assert simulator.stream_calls == [
        {
            "chunk_duration": 0.5,
            "sampling_frequency": 8.0,
            "detectors": ["H1", "L1"],
            "seed": 17,
        }
    ]
    np.testing.assert_array_equal(first["H1"], np.array([0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(second["L1"], np.array([4.0, 5.0, 6.0, 7.0]))
