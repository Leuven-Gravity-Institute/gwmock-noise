"""Tests for overlap-add stitching helper."""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise.simulators._stitching import OverlapAddStitcher


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
