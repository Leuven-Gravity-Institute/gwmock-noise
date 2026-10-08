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


@pytest.mark.parametrize(("window_size", "overlap_size"), [(8, 5), (8, 7), (8, 8), (8, 9), (9, 5), (64, 48)])
def test_stitcher_validates_overlap_at_most_half_window(window_size: int, overlap_size: int) -> None:
    """OverlapAddStitcher rejects overlaps longer than half the window.

    Beyond half a window the overlap regions of consecutive chunks cover each
    sample three times, and the pairwise blend normalisation then damps the
    whole output instead of preserving the process variance.
    """
    with pytest.raises(ValueError, match="overlap_size must be at most half of window_size"):
        OverlapAddStitcher(detectors=["H1"], window_size=window_size, overlap_size=overlap_size)


@pytest.mark.parametrize(("window_size", "overlap_size"), [(2, 1), (8, 4), (9, 4), (64, 32)])
def test_stitcher_accepts_overlap_up_to_half_window(window_size: int, overlap_size: int) -> None:
    """An overlap of up to half the window (rounded down) is accepted."""
    stitcher = OverlapAddStitcher(detectors=["H1"], window_size=window_size, overlap_size=overlap_size)
    assert stitcher.overlap_size == overlap_size


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


@pytest.mark.parametrize(("overlap_size", "n_samples"), [(32, 320), (32, 300), (32, 32), (16, 280)])
def test_stitch_tail_keeps_process_variance(overlap_size: int, n_samples: int) -> None:
    """The last ``overlap_size`` output samples are blended, not faded to zero.

    Unit-variance white chunks give unit variance everywhere in a correctly
    blended output, so the tail must match the interior instead of decaying.
    """
    window_size = 64
    trials = 2000
    rng = np.random.default_rng(2024)

    def generator() -> dict[str, np.ndarray]:
        return {"H1": rng.standard_normal(window_size)}

    realizations = np.array(
        [
            OverlapAddStitcher(["H1"], window_size=window_size, overlap_size=overlap_size).stitch(
                n_samples=n_samples, chunk_generator=generator
            )["H1"]
            for _ in range(trials)
        ]
    )
    variance = np.var(realizations, axis=0)

    assert realizations.shape == (trials, n_samples)
    np.testing.assert_allclose(variance[-overlap_size:].mean(), 1.0, rtol=0.05, atol=0.0)
    assert variance[-overlap_size:].min() > 0.85
    assert variance[-1] > 0.85
