"""Pydantic models for noise simulation configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")


class NoiseConfig(BaseModel):
    """Configuration for gravitational wave detector noise simulation.

    This model is designed to be imported and composed into larger configuration
    structures by upstream packages (e.g., gwsim).
    """

    detectors: list[str] = Field(
        default=["H1", "L1"],
        description="List of detector names to simulate.",
        min_length=1,
    )
    duration: float = Field(
        default=4.0,
        gt=0,
        description="Duration of the noise realization in seconds.",
    )
    sampling_frequency: float = Field(
        default=4096.0,
        gt=0,
        description="Sampling frequency in Hz.",
    )
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Output configuration.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducibility. If None, use system entropy.",
    )

    model_config = {"frozen": False, "extra": "ignore"}
