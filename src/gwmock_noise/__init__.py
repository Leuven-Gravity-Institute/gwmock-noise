"""Top-level package for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from gwmock_noise.config import NoiseComponentConfig, NoiseConfig, OutputConfig, load_config
from gwmock_noise.gaussian import SpectralLine
from gwmock_noise.glitches import (
    BlipGlitch,
    DeepExtractorGlitch,
    EmpiricalSNRDistribution,
    GengliBlipGlitch,
    GlitchDraw,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    NetworkCoherence,
    PowerLawSNRDistribution,
    ScatteredLightGlitch,
    SNRDistribution,
)
from gwmock_noise.parallel import ParallelAdapter
from gwmock_noise.populations import (
    GLITCH_POPULATION_SCHEMA_VERSION,
    ExpectedCount,
    GlitchClass,
    GlitchPopulation,
    GlitchRealization,
    available_glitch_populations,
    et_o3_anchored_population,
    get_glitch_population,
)
from gwmock_noise.simulators import (
    GLITCH_CATALOGUE_COLUMNS,
    GLITCH_CATALOGUE_SCHEMA_VERSION,
    GLITCH_CATALOGUE_TIME_CONVENTION,
    AddLines,
    ARMANoiseSimulator,
    ARNoiseSimulator,
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CompositeNoiseSimulator,
    ConfigurableNoiseSimulator,
    CorrelatedARNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    FitError,
    GlitchNoiseSimulator,
    InjectGlitches,
    MultichannelNoiseSimulator,
    NoiseSimulator,
    OverlapSaveFirSimulator,
    SchumannNoiseSimulator,
    SchumannParams,
    SimulationResult,
    SpectralLineSimulator,
    TimeVaryingColoredNoiseSimulator,
    WhiteNoiseSimulator,
    apply_segment_gps_start,
    open_stream,
    take,
)
from gwmock_noise.spectral import (
    SpectralCovariance,
    assemble_hermitian_spectral_matrices,
    build_spectral_covariance_from_files,
    cholesky_factors_from_spectral_matrices,
    interpolate_complex_spectral_series,
    interpolate_real_spectral_series,
    load_and_interpolate_csd,
    load_and_interpolate_psd,
    normalize_csd_mapping,
    normalize_detector_pair,
    regularized_cholesky,
    sample_complex_frequency_coefficients,
    simulate_spectral_covariance_chunk,
    time_series_from_frequency_coefficients,
)
from gwmock_noise.utils.log import setup_logger
from gwmock_noise.version import __version__

# Configure the shared package logger on import so warnings (e.g. coarse
# frequency resolution) are emitted with a clear severity label instead of the
# bare-message fallback of ``logging.lastResort``. Applications and the CLI may
# call setup_logger() again to adjust the level or add a log file.
setup_logger()

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
    "GLITCH_CATALOGUE_COLUMNS",
    "GLITCH_CATALOGUE_SCHEMA_VERSION",
    "GLITCH_CATALOGUE_TIME_CONVENTION",
    "GLITCH_POPULATION_SCHEMA_VERSION",
    "ARMANoiseSimulator",
    "ARNoiseSimulator",
    "AddLines",
    "BaseNoiseSimulator",
    "BlipGlitch",
    "ColoredNoiseSimulator",
    "CompositeNoiseSimulator",
    "ConfigurableNoiseSimulator",
    "CorrelatedARNoiseSimulator",
    "CorrelatedNoiseSimulator",
    "DeepExtractorGlitch",
    "DefaultNoiseSimulator",
    "DiagnosticResult",
    "EmpiricalSNRDistribution",
    "ExpectedCount",
    "FilterType",
    "FitError",
    "FrameWriter",
    "GWpyAdapter",
    "GengliBlipGlitch",
    "GlitchClass",
    "GlitchDraw",
    "GlitchModel",
    "GlitchNoiseSimulator",
    "GlitchPopulation",
    "GlitchRealization",
    "GwoscFilterConfig",
    "GwoscNoiseConfig",
    "GwoscNoiseFetcher",
    "GwoscNoiseSimulator",
    "GwoscSegmentFilter",
    "InjectGlitches",
    "LogNormalAmplitudeDistribution",
    "MultichannelNoiseSimulator",
    "NetworkCoherence",
    "NoiseComponentConfig",
    "NoiseConfig",
    "NoiseSimulator",
    "OutputConfig",
    "OverlapSaveFirSimulator",
    "ParallelAdapter",
    "PowerLawSNRDistribution",
    "SNRDistribution",
    "ScatteredLightGlitch",
    "SchumannNoiseSimulator",
    "SchumannParams",
    "SimulationResult",
    "SpectralCovariance",
    "SpectralLine",
    "SpectralLineSimulator",
    "TimeVaryingColoredNoiseSimulator",
    "WhiteNoiseSimulator",
    "__version__",
    "apply_segment_gps_start",
    "assemble_hermitian_spectral_matrices",
    "available_glitch_populations",
    "build_spectral_covariance_from_files",
    "cholesky_factors_from_spectral_matrices",
    "compare_psd",
    "estimate_psd",
    "et_o3_anchored_population",
    "get_glitch_population",
    "interpolate_complex_spectral_series",
    "interpolate_real_spectral_series",
    "load_and_interpolate_csd",
    "load_and_interpolate_psd",
    "load_config",
    "normalize_csd_mapping",
    "normalize_detector_pair",
    "open_stream",
    "regularized_cholesky",
    "run_diagnostics",
    "sample_complex_frequency_coefficients",
    "simulate_spectral_covariance_chunk",
    "take",
    "time_series_from_frequency_coefficients",
]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional top-level exports."""
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is not None:
        export = getattr(import_module(module_name), name)
        globals()[name] = export
        return export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
