"""Configuration models for gwsim_noise.

This module provides Pydantic models for noise simulation configuration.
Upstream packages (e.g., gwsim) can import these models to build and validate
their full configuration.
"""

from __future__ import annotations

from gwsim_noise.config.loader import load_config
from gwsim_noise.config.models import NoiseConfig, OutputConfig

__all__ = ["NoiseConfig", "OutputConfig", "load_config"]
