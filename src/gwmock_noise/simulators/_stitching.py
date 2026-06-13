"""Shared overlap-add stitching for detector noise simulators."""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np

# Default synthesis window expressed as a fixed duration (seconds). A
# duration-based window keeps the frequency resolution ``df = 1 / window_duration``
# invariant to the sampling frequency. The default of 4 s (df = 0.25 Hz) is fine
# enough to resolve the narrow, fast-varying resonance modes in typical LIGO PSDs;
# the historical 0.5 s implied by a 2048-sample window at 4096 Hz was too coarse.
DEFAULT_WINDOW_DURATION = 4.0

# Smallest window (samples) that still yields a positive overlap (>= 1).
MIN_WINDOW_SIZE = 2

WINDOW_SIZE = 2048
OVERLAP_SIZE = WINDOW_SIZE // 2


def resolve_window_sizes(window_duration: float, sampling_frequency: float) -> tuple[int, int]:
    """Resolve a seconds-based window into ``(window_size, overlap_size)`` samples.

    Args:
        window_duration: Synthesis window length in seconds.
        sampling_frequency: Sampling frequency in Hz.

    Returns:
        A tuple of ``(window_size, overlap_size)`` in samples, with a half-window
        overlap.

    Raises:
        ValueError: If ``window_duration`` is not positive or is too short to span
            at least ``MIN_WINDOW_SIZE`` samples at the given sampling frequency.
    """
    if window_duration <= 0:
        raise ValueError("window_duration must be greater than zero.")
    window_size = round(window_duration * sampling_frequency)
    if window_size < MIN_WINDOW_SIZE:
        raise ValueError(
            "window_duration is too short for the sampling frequency: it must span "
            f"at least {MIN_WINDOW_SIZE} samples (got {window_size})."
        )
    overlap_size = window_size // 2
    return window_size, overlap_size


def warn_if_underresolved(
    *,
    delta_frequency: float,
    low_frequency_cutoff: float,
    reference_spacing: float | None = None,
    logger: logging.Logger,
    context: str = "",
) -> None:
    """Warn when the synthesis grid is too coarse to capture the target spectrum.

    Two conditions trigger a warning (see #139):

    * ``delta_frequency`` is coarser than ``low_frequency_cutoff`` — the band of
      interest is spanned by too few bins to shape it reliably.
    * ``delta_frequency`` is coarser than ``reference_spacing`` — the input
      spectrum is sampled (or varies) more finely than the synthesis grid, so its
      structure is lost.

    The warning is informational; generation proceeds regardless.
    """
    prefix = f"{context}: " if context else ""
    if low_frequency_cutoff > 0 and delta_frequency > low_frequency_cutoff:
        logger.warning(
            "%sfrequency resolution df=%.4g Hz is coarser than the low-frequency "
            "cutoff %.4g Hz; consider increasing window_duration.",
            prefix,
            delta_frequency,
            low_frequency_cutoff,
        )
    if reference_spacing is not None and reference_spacing > 0 and delta_frequency > reference_spacing:
        logger.warning(
            "%sfrequency resolution df=%.4g Hz is coarser than the input spectrum "
            "scale %.4g Hz; fine structure will be lost. Consider increasing window_duration.",
            prefix,
            delta_frequency,
            reference_spacing,
        )


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
