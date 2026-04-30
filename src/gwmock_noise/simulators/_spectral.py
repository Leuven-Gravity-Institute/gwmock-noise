"""Helpers for loading PSD and CSD inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

SPECTRAL_COLUMNS = 2


def load_spectral_series(
    file_path: str | Path,
    *,
    kind: str,
    complex_values: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column spectral series from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"{kind} file not found: {path}")

    if path.suffix == ".npy":
        data = np.load(path)
    elif path.suffix == ".txt":
        data = np.loadtxt(path, dtype=np.complex128 if complex_values else float)
    elif path.suffix == ".csv":
        data = np.loadtxt(path, delimiter=",", dtype=np.complex128 if complex_values else float)
    else:
        raise ValueError(f"Unsupported {kind} file format: {path.suffix}. Use .npy, .txt, or .csv.")

    if data.ndim != SPECTRAL_COLUMNS or data.shape[1] != SPECTRAL_COLUMNS:
        raise ValueError(f"{kind} file must have shape (N, 2).")

    raw_frequencies = np.asarray(data[:, 0])
    if np.iscomplexobj(raw_frequencies) and not np.allclose(raw_frequencies.imag, 0.0):
        raise ValueError(f"{kind} frequency column must be real.")

    frequencies = np.asarray(raw_frequencies.real, dtype=float)
    values = np.asarray(data[:, 1], dtype=np.complex128 if complex_values else float)
    order = np.argsort(frequencies)
    return frequencies[order], values[order]
