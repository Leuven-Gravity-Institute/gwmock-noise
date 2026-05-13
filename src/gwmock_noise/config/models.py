"""Pydantic configuration schema for noise simulation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, Field, model_validator


class NoiseComponentConfig(BaseModel):
    """One configurable noise component in a composed simulation."""

    simulator: str = Field(description="Registered simulator/component name.")
    options: dict[str, Any] = Field(
        default_factory=dict,
        description="Simulator-specific options passed to the component builder.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_component_definition(cls, value: Any) -> Any:
        """Accept string, flat mapping, or explicit ``{simulator, options}`` input."""
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return {"simulator": value, "options": {}}
        if not isinstance(value, dict):
            raise ValueError("components entries must be strings, mappings, or NoiseComponentConfig instances.")

        if "simulator" not in value:
            raise ValueError("components entries must define a simulator name.")

        normalized = dict(value)
        simulator = normalized.pop("simulator")
        declared_options = normalized.pop("options", None)
        if declared_options is None:
            options = normalized
        else:
            if not isinstance(declared_options, dict):
                raise ValueError("component options must be a mapping when provided explicitly.")
            overlap = set(normalized) & set(declared_options)
            if overlap:
                duplicated = ", ".join(sorted(overlap))
                raise ValueError(f"component options duplicate explicit fields: {duplicated}.")
            options = dict(declared_options)
            options.update(normalized)
        return {"simulator": simulator, "options": options}

    @model_validator(mode="after")
    def validate_component(self) -> Self:
        """Validate the normalized component entry."""
        if not self.simulator.strip():
            raise ValueError("component simulator names must be non-empty.")
        return self


class OutputConfig(BaseModel):
    """Configuration for simulation output."""

    directory: Path = Field(default=Path("."), description="Output directory for generated data.")
    prefix: str = Field(default="noise", description="Prefix for output filenames.")
    format: str = Field(
        default="npy",
        description="Artifact format written by BaseNoiseSimulator.run().",
        pattern="^(npy|gwf)$",
    )
    gps_start: float = Field(
        default=0.0,
        description="GPS start time used for timestamped output formats such as GWF.",
    )
    channel_prefix: str = Field(
        default="MOCK",
        description="Channel-name prefix used for GWF frame output.",
    )


def _default_components() -> list[NoiseComponentConfig]:
    """Return the legacy default of one white-noise component."""
    return [NoiseComponentConfig(simulator="white")]


class NoiseConfig(BaseModel):
    """Generic configuration for composed detector-noise simulations."""

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
    components: list[NoiseComponentConfig] = Field(
        default_factory=_default_components,
        description="Ordered list of noise components to generate and add together.",
    )

    model_config = {"frozen": False, "extra": "ignore"}
