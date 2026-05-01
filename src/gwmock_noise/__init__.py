"""Top-level package for gwmock-noise."""

from __future__ import annotations

from gwmock_noise.config import NoiseConfig, OutputConfig, SpectralLine, load_config
from gwmock_noise.simulators import (
    AddLines,
    ARNoiseSimulator,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CorrelatedARNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    NoiseSimulator,
    SimulationResult,
    SpectralLineSimulator,
    TimeVaryingColoredNoiseSimulator,
)
from gwmock_noise.version import __version__

__all__ = [
    "ARNoiseSimulator",
    "AddLines",
    "BaseNoiseSimulator",
    "ColoredNoiseSimulator",
    "CorrelatedARNoiseSimulator",
    "CorrelatedNoiseSimulator",
    "DefaultNoiseSimulator",
    "NoiseConfig",
    "NoiseSimulator",
    "OutputConfig",
    "SimulationResult",
    "SpectralLine",
    "SpectralLineSimulator",
    "TimeVaryingColoredNoiseSimulator",
    "__version__",
    "load_config",
]
