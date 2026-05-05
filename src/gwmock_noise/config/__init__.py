"""Configuration models for gwmock_noise.

This module provides Pydantic models for noise simulation configuration.
Upstream packages (e.g., gwmock) can import these models to build and validate
their full configuration.
"""

from __future__ import annotations

from gwmock_noise.config.loader import load_config
from gwmock_noise.config.models import (
    BlipGlitch,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
    ScatteredLightGlitch,
    SpectralLine,
)
from gwmock_noise.glitches import GengliBlipGlitch

__all__ = [
    "BlipGlitch",
    "GengliBlipGlitch",
    "GlitchModel",
    "LogNormalAmplitudeDistribution",
    "NoiseConfig",
    "OutputConfig",
    "ScatteredLightGlitch",
    "SpectralLine",
    "load_config",
]
