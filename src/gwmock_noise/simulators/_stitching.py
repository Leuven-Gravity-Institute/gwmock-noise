"""Shared overlap-add stitching for detector noise simulators."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

WINDOW_SIZE = 2048
OVERLAP_SIZE = WINDOW_SIZE // 2


class OverlapAddStitcher:
    """Blend adjacent noise chunks into a continuous time series."""

    def __init__(
        self,
        detectors: list[str],
        *,
        window_size: int = WINDOW_SIZE,
        overlap_size: int = OVERLAP_SIZE,
    ) -> None:
        self.window_size = window_size
        self.overlap_size = overlap_size
        if self.window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if self.overlap_size <= 0:
            raise ValueError("overlap_size must be a positive integer.")
        if self.overlap_size >= self.window_size:
            raise ValueError("overlap_size must be smaller than window_size.")
        self.previous_strain: dict[str, np.ndarray] = {}
        self._window_out = np.cos(np.linspace(0.0, np.pi / 2.0, overlap_size))
        self._window_in = np.sin(np.linspace(0.0, np.pi / 2.0, overlap_size))
        self._blend_norm = np.sqrt(self._window_out**2 + self._window_in**2)
        self.configure_detectors(detectors)

    def configure_detectors(self, detectors: list[str]) -> None:
        """Update detector ordering and clear continuity state."""
        self.detectors = list(detectors)
        self.reset()

    def reset(self) -> None:
        """Clear any cached overlap history."""
        self.previous_strain.clear()

    def _validate_chunk_map(self, chunks: dict[str, np.ndarray]) -> None:
        expected = set(self.detectors)
        actual = set(chunks)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise ValueError(f"chunk generator must return exactly the configured detectors ({', '.join(details)}).")

        for detector in self.detectors:
            if chunks[detector].shape != (self.window_size,):
                raise ValueError(f"chunk for {detector} must have shape ({self.window_size},).")

    def stitch(
        self,
        *,
        n_samples: int,
        chunk_generator: Callable[[], dict[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        """Generate a continuous realization with overlap-add stitching."""
        if n_samples <= 0:
            raise ValueError("n_samples must be positive.")

        history = self.previous_strain
        if history:
            self._validate_chunk_map(history)
        else:
            history = chunk_generator()
            self._validate_chunk_map(history)

        raw_buffers = {detector: history[detector].copy() for detector in self.detectors}
        strain_buffers = {detector: raw_buffers[detector].copy() for detector in self.detectors}

        for detector in self.detectors:
            strain_buffers[detector][-self.overlap_size :] *= self._window_out

        new_strain_segments = {detector: [] for detector in self.detectors}
        new_raw_segments = {detector: [] for detector in self.detectors}
        blended_overlaps = {detector: [] for detector in self.detectors}
        overlap_sources = {
            detector: strain_buffers[detector][-self.overlap_size :].copy() for detector in self.detectors
        }
        extension_step = self.window_size - self.overlap_size
        current_size = self.window_size

        while current_size - self.window_size < n_samples:
            raw_new_chunks = chunk_generator()
            self._validate_chunk_map(raw_new_chunks)

            for detector in self.detectors:
                raw_new = raw_new_chunks[detector]
                new_chunk = raw_new.copy()
                new_chunk[: self.overlap_size] *= self._window_in
                new_chunk[-self.overlap_size :] *= self._window_out
                blended_overlaps[detector].append(
                    (overlap_sources[detector] + new_chunk[: self.overlap_size]) / self._blend_norm
                )
                new_strain_segments[detector].append(new_chunk[self.overlap_size :])
                new_raw_segments[detector].append(raw_new[self.overlap_size :])
                overlap_sources[detector] = new_chunk[-self.overlap_size :].copy()

            current_size += extension_step

        for detector in self.detectors:
            if new_strain_segments[detector]:
                strain_buffers[detector] = np.concatenate(
                    [strain_buffers[detector], np.concatenate(new_strain_segments[detector])]
                )
                raw_buffers[detector] = np.concatenate(
                    [raw_buffers[detector], np.concatenate(new_raw_segments[detector])]
                )
                for index, blended_overlap in enumerate(blended_overlaps[detector]):
                    overlap_start = (self.window_size - self.overlap_size) + (index * extension_step)
                    overlap_stop = overlap_start + self.overlap_size
                    strain_buffers[detector][overlap_start:overlap_stop] = blended_overlap

        realization = {
            detector: strain_buffers[detector][self.window_size : self.window_size + n_samples].copy()
            for detector in self.detectors
        }
        self.previous_strain.clear()
        self.previous_strain.update(
            {
                detector: raw_buffers[detector][n_samples : n_samples + self.window_size].copy()
                for detector in self.detectors
            }
        )
        return realization
