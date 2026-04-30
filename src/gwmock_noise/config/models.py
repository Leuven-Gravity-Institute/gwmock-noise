"""Pydantic models for noise simulation configuration."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")


class NoiseConfig(BaseModel):
    """Configuration for gravitational wave detector noise simulation.

    This model is designed to be imported and composed into larger configuration
    structures by upstream packages (e.g., gwmock).
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
    psd_file: Path | None = Field(
        default=None,
        description="Optional PSD file for FFT-based colored-noise simulation.",
    )
    low_frequency_cutoff: float = Field(
        default=2.0,
        ge=0,
        description="Lower frequency cutoff applied during colored-noise generation.",
    )
    high_frequency_cutoff: float | None = Field(
        default=None,
        gt=0,
        description="Upper frequency cutoff applied during colored-noise generation.",
    )

    @model_validator(mode="after")
    def validate_frequency_cutoffs(self) -> NoiseConfig:
        """Validate cutoff ordering and Nyquist limits."""
        nyquist = self.sampling_frequency / 2
        low = self.low_frequency_cutoff
        high = self.high_frequency_cutoff

        if low < 0:
            raise ValueError("low_frequency_cutoff must be >= 0.")
        if low > nyquist:
            raise ValueError(
                f"low_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
            )

        if high is not None:
            if high < 0:
                raise ValueError("high_frequency_cutoff must be >= 0.")
            if high > nyquist:
                raise ValueError(
                    f"high_frequency_cutoff must be <= Nyquist ({nyquist} Hz) for sampling_frequency={self.sampling_frequency} Hz."
                )
            if high <= low:
                raise ValueError("high_frequency_cutoff must be greater than low_frequency_cutoff.")

        return self

    model_config = {"frozen": False, "extra": "ignore"}
