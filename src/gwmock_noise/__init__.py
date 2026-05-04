"""Top-level package for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gwmock_noise.config import (
    BlipGlitch,
    GengliBlipGlitch,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
    ScatteredLightGlitch,
    SpectralLine,
    load_config,
)
from gwmock_noise.parallel import ParallelAdapter
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
    SchumannNoiseSimulator,
    SchumannParams,
    SimulationResult,
    SpectralLineSimulator,
    TimeVaryingColoredNoiseSimulator,
    open_stream,
    take,
)
from gwmock_noise.version import __version__

_OPTIONAL_EXPORTS = {
    "DiagnosticResult": "gwmock_noise.diagnostics",
    "compare_psd": "gwmock_noise.diagnostics",
    "estimate_psd": "gwmock_noise.diagnostics",
    "FrameWriter": "gwmock_noise.output",
    "GWpyAdapter": "gwmock_noise.output",
    "run_diagnostics": "gwmock_noise.diagnostics",
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
    "DiagnosticResult",
    "FrameWriter",
    "GWpyAdapter",
    "GengliBlipGlitch",
    "GlitchModel",
    "InjectGlitches",
    "LogNormalAmplitudeDistribution",
    "NoiseConfig",
    "NoiseSimulator",
    "OutputConfig",
    "ParallelAdapter",
    "ScatteredLightGlitch",
    "SchumannNoiseSimulator",
    "SchumannParams",
    "SimulationResult",
    "SpectralLine",
    "SpectralLineSimulator",
    "TimeVaryingColoredNoiseSimulator",
    "__version__",
    "compare_psd",
    "estimate_psd",
    "load_config",
    "open_stream",
    "run_diagnostics",
    "take",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional top-level exports."""
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is not None:
        export = getattr(import_module(module_name), name)
        globals()[name] = export
        return export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
