"""Tests for spectral-loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.simulators._spectral import load_spectral_series


def test_load_spectral_series_rejects_complex_frequency_column(tmp_path: Path) -> None:
    """Frequency column must be real-valued."""
    path = tmp_path / "bad_freq.npy"
    data = np.column_stack(
        (
            np.array([1.0 + 0.1j, 2.0 + 0.0j], dtype=np.complex128),
            np.array([1.0, 1.0], dtype=np.complex128),
        )
    )
    np.save(path, data)

    with pytest.raises(ValueError, match="frequency column must be real"):
        load_spectral_series(path, kind="PSD")
