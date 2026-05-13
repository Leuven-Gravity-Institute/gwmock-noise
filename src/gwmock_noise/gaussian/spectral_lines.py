"""Runtime spectral-line model definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SpectralLine:
    """Configuration for a narrow-band spectral line."""

    frequency: float
    amplitude: float
    phase: float | None = None
    drift_rate: float = 0.0

    def __post_init__(self) -> None:
        """Validate scalar line parameters."""
        if self.frequency < 0:
            raise ValueError("spectral line frequency must be non-negative.")
        if self.amplitude < 0:
            raise ValueError("spectral line amplitude must be non-negative.")


def normalize_spectral_lines(value: Any) -> list[SpectralLine]:
    """Normalize heterogeneous spectral-line inputs to dataclass instances."""
    if not isinstance(value, list):
        raise ValueError("spectral line component options must provide a list of lines.")

    normalized: list[SpectralLine] = []
    for entry in value:
        if isinstance(entry, SpectralLine):
            normalized.append(entry)
        elif isinstance(entry, dict):
            normalized.append(SpectralLine(**entry))
        else:
            raise ValueError("spectral line entries must be mappings or SpectralLine instances.")
    return normalized
