"""Noise simulators for gravitational wave detectors."""

from __future__ import annotations

from gwmock_noise.glitches.models import (
    BlipGlitch,
    GlitchDraw,
    GlitchModel,
    LogNormalAmplitudeDistribution,
    ScatteredLightGlitch,
)
from gwmock_noise.simulators._fit import FitError
from gwmock_noise.simulators.arma import ARMANoiseSimulator
from gwmock_noise.simulators.autoregressive import ARNoiseSimulator
from gwmock_noise.simulators.base import BaseNoiseSimulator, ConfigurableNoiseSimulator, SimulationResult
from gwmock_noise.simulators.colored import ColoredNoiseSimulator, TimeVaryingColoredNoiseSimulator
from gwmock_noise.simulators.composite import CompositeNoiseSimulator
from gwmock_noise.simulators.correlated import CorrelatedNoiseSimulator
from gwmock_noise.simulators.correlated_ar import CorrelatedARNoiseSimulator
from gwmock_noise.simulators.default import DefaultNoiseSimulator
from gwmock_noise.simulators.glitch_component import GlitchNoiseSimulator
from gwmock_noise.simulators.glitches import (
    GLITCH_CATALOGUE_COLUMNS,
    GLITCH_CATALOGUE_SCHEMA_VERSION,
    GLITCH_CATALOGUE_TIME_CONVENTION,
    InjectGlitches,
    apply_segment_gps_start,
)
from gwmock_noise.simulators.joint_dummy import JointDummyCorrelatedSimulator
from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    ChannelMetadata,
    JointCovariance,
    JointRealization,
    JointStrainWitnessSimulator,
)
from gwmock_noise.simulators.joint_registry import (
    JOINT_BACKEND_ENTRY_POINT_GROUP,
    available_joint_backend_names,
    discover_joint_backends,
    load_joint_backend,
)
from gwmock_noise.simulators.matrix_factorization import whittle_levinson_factorization
from gwmock_noise.simulators.multichannel import MultichannelNoiseSimulator
from gwmock_noise.simulators.overlap_save import OverlapSaveFirSimulator
from gwmock_noise.simulators.protocol import NoiseSimulator
from gwmock_noise.simulators.real_noise import GwoscNoiseSimulator
from gwmock_noise.simulators.schumann import SchumannNoiseSimulator, SchumannParams
from gwmock_noise.simulators.spectral_lines import AddLines, SpectralLineSimulator
from gwmock_noise.simulators.streaming import open_stream, take
from gwmock_noise.simulators.white import WhiteNoiseSimulator

__all__ = [
    "GLITCH_CATALOGUE_COLUMNS",
    "GLITCH_CATALOGUE_SCHEMA_VERSION",
    "GLITCH_CATALOGUE_TIME_CONVENTION",
    "JOINT_BACKEND_ENTRY_POINT_GROUP",
    "ARMANoiseSimulator",
    "ARNoiseSimulator",
    "AddLines",
    "BaseNoiseSimulator",
    "BlipGlitch",
    "ChannelDomain",
    "ChannelKind",
    "ChannelMetadata",
    "ColoredNoiseSimulator",
    "CompositeNoiseSimulator",
    "ConfigurableNoiseSimulator",
    "CorrelatedARNoiseSimulator",
    "CorrelatedNoiseSimulator",
    "DefaultNoiseSimulator",
    "FitError",
    "GlitchDraw",
    "GlitchModel",
    "GlitchNoiseSimulator",
    "GwoscNoiseSimulator",
    "InjectGlitches",
    "JointCovariance",
    "JointDummyCorrelatedSimulator",
    "JointRealization",
    "JointStrainWitnessSimulator",
    "LogNormalAmplitudeDistribution",
    "MultichannelNoiseSimulator",
    "NoiseSimulator",
    "OverlapSaveFirSimulator",
    "ScatteredLightGlitch",
    "SchumannNoiseSimulator",
    "SchumannParams",
    "SimulationResult",
    "SpectralLineSimulator",
    "TimeVaryingColoredNoiseSimulator",
    "WhiteNoiseSimulator",
    "apply_segment_gps_start",
    "available_joint_backend_names",
    "discover_joint_backends",
    "load_joint_backend",
    "open_stream",
    "take",
    "whittle_levinson_factorization",
]
