"""Tests for overlap-add stitching helper."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise.simulators._stitching import DEFAULT_WINDOW_DURATION, OverlapAddStitcher


def test_default_window_duration_is_64_seconds() -> None:
    """The default synthesis window is 64 s, tuned for ET PSD accuracy."""
    assert DEFAULT_WINDOW_DURATION == 64


@pytest.mark.parametrize("window_size", [0, -1])
def test_stitcher_validates_positive_window_size(window_size: int) -> None:
    """OverlapAddStitcher rejects non-positive window sizes."""
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        OverlapAddStitcher(detectors=["H1"], window_size=window_size, overlap_size=1)


@pytest.mark.parametrize("overlap_size", [0, -1])
def test_stitcher_validates_positive_overlap_size(overlap_size: int) -> None:
    """OverlapAddStitcher rejects non-positive overlap sizes."""
    with pytest.raises(ValueError, match="overlap_size must be a positive integer"):
        OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=overlap_size)


@pytest.mark.parametrize("overlap_size", [8, 9])
def test_stitcher_validates_overlap_smaller_than_window(overlap_size: int) -> None:
    """OverlapAddStitcher requires overlap_size < window_size."""
    with pytest.raises(ValueError, match="overlap_size must be smaller than window_size"):
        OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=overlap_size)


def test_stitcher_accepts_valid_sizes() -> None:
    """Valid window/overlap sizes initialize stitching state."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=4)
    assert stitcher.window_size == 8
    assert stitcher.overlap_size == 4
    assert np.all(stitcher._blend_norm > 0)


def test_stitcher_rejects_chunk_map_with_missing_detector() -> None:
    """Stitch validates missing detectors in generated chunks."""
    stitcher = OverlapAddStitcher(detectors=["H1", "L1"], window_size=8, overlap_size=4)

    def bad_generator() -> dict[str, np.ndarray]:
        return {"H1": np.zeros(8)}

    with pytest.raises(ValueError, match="missing"):
        stitcher.stitch(n_samples=4, chunk_generator=bad_generator)


def test_stitcher_rejects_chunk_map_with_extra_detector() -> None:
    """Stitch validates extra detectors in generated chunks."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=4)

    def bad_generator() -> dict[str, np.ndarray]:
        return {"H1": np.zeros(8), "L1": np.zeros(8)}

    with pytest.raises(ValueError, match="extra"):
        stitcher.stitch(n_samples=4, chunk_generator=bad_generator)


def test_stitcher_rejects_chunk_map_with_wrong_shape() -> None:
    """Stitch validates per-detector chunk shapes."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=4)

    def bad_generator() -> dict[str, np.ndarray]:
        return {"H1": np.zeros(7)}

    with pytest.raises(ValueError, match="must have shape"):
        stitcher.stitch(n_samples=4, chunk_generator=bad_generator)


def test_stitcher_validates_cached_history_before_generation() -> None:
    """Stitch validates cached continuity buffers before using them."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=4)
    stitcher.previous_strain["H1"] = np.zeros(7)

    with pytest.raises(ValueError, match="must have shape"):
        stitcher.stitch(n_samples=4, chunk_generator=lambda: {"H1": np.zeros(8)})


def test_stitcher_rejects_non_positive_sample_request() -> None:
    """Stitch rejects non-positive sample counts."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=8, overlap_size=4)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        stitcher.stitch(n_samples=0, chunk_generator=lambda: {"H1": np.zeros(8)})
