"""PSD reference helpers used by Gaussian-noise configuration and simulators."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from urllib.parse import urlparse

PSD_PRESET_PACKAGE = "gwmock_noise.data.psd"
PSD_PRESET_SUFFIX = ".txt"
REMOTE_PSD_SCHEMES = {"http", "https"}


def is_remote_psd_reference(value: str) -> bool:
    """Return whether a PSD reference is an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in REMOTE_PSD_SCHEMES and bool(parsed.netloc)


def resolve_bundled_psd_preset(value: str) -> Path | None:
    """Resolve a bare preset name to a bundled PSD asset."""
    if any(separator in value for separator in ("/", "\\")) or Path(value).suffix:
        return None

    resource = resources.files(PSD_PRESET_PACKAGE).joinpath(f"{value}{PSD_PRESET_SUFFIX}")
    if not resource.is_file():
        return None

    return Path(str(resource))
