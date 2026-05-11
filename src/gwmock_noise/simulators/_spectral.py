"""Helpers for loading PSD and CSD inputs."""

from __future__ import annotations

import importlib.resources
import logging
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np

logger = logging.getLogger(__name__)

SPECTRAL_COLUMNS = 2
REMOTE_SPECTRAL_SCHEMES = {"http", "https"}
_BUNDLED_PSD_SUFFIXES = (".txt", ".csv", ".npy")


def _resolve_bundled_psd(name: str) -> Path | None:
    """Return the path to a bundled PSD preset, or None if not found."""
    stem = Path(name).stem
    package = importlib.resources.files("gwmock_noise.data.psd")
    for suffix in _BUNDLED_PSD_SUFFIXES:
        candidate = package.joinpath(f"{stem}{suffix}")
        try:
            # is_file() works for both real paths and importlib traversable objects
            if candidate.is_file():
                return Path(str(candidate))
        except (TypeError, AttributeError):
            logger.debug(f"Candidate {candidate} is not a file in the bundled PSDs.")
    return None


def _is_remote_spectral_reference(file_path: str | Path) -> bool:
    """Return whether the reference points to an HTTP(S) resource."""
    if isinstance(file_path, Path):
        return False

    parsed = urlparse(file_path)
    return parsed.scheme in REMOTE_SPECTRAL_SCHEMES and bool(parsed.netloc)


def normalize_spectral_reference(file_path: str | Path) -> str | Path:
    """Normalize spectral references while preserving remote URLs."""
    if isinstance(file_path, Path):
        return file_path

    return file_path if _is_remote_spectral_reference(file_path) else Path(file_path)


def load_spectral_series(  # noqa: PLR0912
    file_path: str | Path,
    *,
    kind: str,
    complex_values: bool = False,
    timeout: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column spectral series from disk or a remote URL.

    Args:
        file_path: Path to the spectral file.
        kind: The type of spectral series to load.
        complex_values: Whether the spectral values are complex-valued.
        timeout: The timeout for remote requests.

    Returns:
        A tuple of frequency and value arrays.

    Raises:
        FileNotFoundError: If the file is not found.
        ValueError: If the file format is unsupported.
    """
    source = normalize_spectral_reference(file_path)
    is_remote = isinstance(source, str)
    suffix_source = urlparse(source).path if is_remote else str(source)
    suffix = Path(suffix_source).suffix.lower()

    if not is_remote and not source.exists():
        resolved = _resolve_bundled_psd(str(source))
        if resolved is not None:
            source = resolved
            suffix = source.suffix.lower()
        else:
            raise FileNotFoundError(f"{kind} file not found: {source}")

    if suffix == ".npy":
        if is_remote:
            raise ValueError(f"Unsupported remote {kind} file format: {suffix}. Use .txt or .csv for URL sources.")
        data = np.load(source)
    elif suffix == ".txt":
        if is_remote:
            with urlopen(source, timeout=timeout) as response:  # noqa: S310
                data = np.loadtxt(response, dtype=np.complex128 if complex_values else float)
        else:
            data = np.loadtxt(source, dtype=np.complex128 if complex_values else float)
    elif suffix == ".csv":
        if is_remote:
            with urlopen(source, timeout=timeout) as response:  # noqa: S310
                data = np.loadtxt(response, delimiter=",", dtype=np.complex128 if complex_values else float)
        else:
            data = np.loadtxt(source, delimiter=",", dtype=np.complex128 if complex_values else float)
    else:
        raise ValueError(f"Unsupported {kind} file format: {suffix}. Use .npy, .txt, or .csv.")

    if data.ndim != SPECTRAL_COLUMNS or data.shape[1] != SPECTRAL_COLUMNS:
        raise ValueError(f"{kind} file must have shape (N, 2).")

    raw_frequencies = np.asarray(data[:, 0])
    if np.iscomplexobj(raw_frequencies) and not np.allclose(raw_frequencies.imag, 0.0):
        raise ValueError(f"{kind} frequency column must be real.")

    frequencies = np.asarray(raw_frequencies.real, dtype=float)
    values = np.asarray(data[:, 1], dtype=np.complex128 if complex_values else float)
    order = np.argsort(frequencies)
    return frequencies[order], values[order]
