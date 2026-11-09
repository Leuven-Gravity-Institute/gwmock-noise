"""Registered glitch populations and the machinery that pins them."""

from __future__ import annotations

from gwmock_noise.populations.deepextractor_o3 import deepextractor_o3_population
from gwmock_noise.populations.et_o3_anchored import (
    POPULATION_NAME,
    REFERENCES,
    UNANCHORED,
    et_o3_anchored_population,
)
from gwmock_noise.populations.population import (
    GLITCH_POPULATION_SCHEMA_VERSION,
    ExpectedCount,
    GlitchClass,
    GlitchPopulation,
    GlitchRealization,
)
from gwmock_noise.populations.registry import (
    UnknownPopulationArgumentError,
    available_glitch_populations,
    get_glitch_population,
)

__all__ = [
    "GLITCH_POPULATION_SCHEMA_VERSION",
    "POPULATION_NAME",
    "REFERENCES",
    "UNANCHORED",
    "ExpectedCount",
    "GlitchClass",
    "GlitchPopulation",
    "GlitchRealization",
    "UnknownPopulationArgumentError",
    "available_glitch_populations",
    "deepextractor_o3_population",
    "et_o3_anchored_population",
    "get_glitch_population",
]
