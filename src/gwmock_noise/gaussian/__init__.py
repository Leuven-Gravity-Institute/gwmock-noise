"""Gaussian-noise domain models and helpers."""

from __future__ import annotations

from gwmock_noise.gaussian.psd import is_remote_psd_reference, resolve_bundled_psd_preset
from gwmock_noise.gaussian.spectral_lines import SpectralLine, normalize_spectral_lines

__all__ = [
    "SpectralLine",
    "is_remote_psd_reference",
    "normalize_spectral_lines",
    "resolve_bundled_psd_preset",
]
