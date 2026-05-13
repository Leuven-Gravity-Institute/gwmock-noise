"""Top-level package for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gwmock_noise.config import NoiseComponentConfig, NoiseConfig, OutputConfig, load_config
from gwmock_noise.gaussian import SpectralLine
from gwmock_noise.glitches import (
    BlipGlitch,
    GengliBlipGlitch,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    ScatteredLightGlitch,
)
from gwmock_noise.parallel import ParallelAdapter
from gwmock_noise.simulators import (
    AddLines,
    ARNoiseSimulator,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CompositeNoiseSimulator,
    ConfigurableNoiseSimulator,
    CorrelatedARNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    GlitchNoiseSimulator,
    InjectGlitches,
    NoiseSimulator,
    SchumannNoiseSimulator,
    SchumannParams,
    SimulationResult,
    SpectralLineSimulator,
    TimeVaryingColoredNoiseSimulator,
    WhiteNoiseSimulator,
    open_stream,
    take,
)
from gwmock_noise.version import __version__

_OPTIONAL_EXPORTS = {
    "DiagnosticResult": "gwmock_noise.diagnostics",
    "FilterType": "gwmock_noise.gwosc",
    "GwoscFilterConfig": "gwmock_noise.gwosc",
    "GwoscNoiseConfig": "gwmock_noise.gwosc",
    "GwoscNoiseFetcher": "gwmock_noise.gwosc",
    "GwoscNoiseSimulator": "gwmock_noise.simulators",
    "GwoscSegmentFilter": "gwmock_noise.gwosc",
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
    "CompositeNoiseSimulator",
    "ConfigurableNoiseSimulator",
    "CorrelatedARNoiseSimulator",
    "CorrelatedNoiseSimulator",
    "DefaultNoiseSimulator",
    "DiagnosticResult",
    "FilterType",
    "FrameWriter",
    "GWpyAdapter",
    "GengliBlipGlitch",
    "GlitchModel",
    "GlitchNoiseSimulator",
    "GwoscFilterConfig",
    "GwoscNoiseConfig",
    "GwoscNoiseFetcher",
    "GwoscNoiseSimulator",
    "GwoscSegmentFilter",
    "InjectGlitches",
    "LogNormalAmplitudeDistribution",
    "NoiseComponentConfig",
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
    "WhiteNoiseSimulator",
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
