"""Top-level package for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gwmock_noise.config import (
    BlipGlitch,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
    ScatteredLightGlitch,
    SpectralLine,
    load_config,
)
from gwmock_noise.simulators import (
    AddLines,
    ARNoiseSimulator,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CorrelatedARNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    InjectGlitches,
    NoiseSimulator,
    SimulationResult,
    SpectralLineSimulator,
    TimeVaryingColoredNoiseSimulator,
)
from gwmock_noise.version import __version__

_OPTIONAL_EXPORTS = {
    "FrameWriter": "gwmock_noise.output",
    "GWpyAdapter": "gwmock_noise.output",
}

__all__ = [
    "ARNoiseSimulator",
    "AddLines",
    "BaseNoiseSimulator",
    "BlipGlitch",
    "ColoredNoiseSimulator",
    "CorrelatedARNoiseSimulator",
    "CorrelatedNoiseSimulator",
    "DefaultNoiseSimulator",
    "FrameWriter",
    "GWpyAdapter",
    "GlitchModel",
    "InjectGlitches",
    "LogNormalAmplitudeDistribution",
    "NoiseConfig",
    "NoiseSimulator",
    "OutputConfig",
    "ScatteredLightGlitch",
    "SimulationResult",
    "SpectralLine",
    "SpectralLineSimulator",
    "TimeVaryingColoredNoiseSimulator",
    "__version__",
    "load_config",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional top-level exports."""
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is not None:
        export = getattr(import_module(module_name), name)
        globals()[name] = export
        return export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
