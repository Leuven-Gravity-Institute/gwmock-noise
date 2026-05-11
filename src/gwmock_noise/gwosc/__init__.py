"""GWOSC real-noise fetching subpackage.

Provides configuration models, segment filtering, and data fetching
for retrieving real gravitational-wave detector strain data from the
Gravitational-Wave Open Science Centre (GWOSC).
"""

from __future__ import annotations

from gwmock_noise.gwosc.fetcher import GwoscNoiseFetcher
from gwmock_noise.gwosc.filters import GwoscSegmentFilter
from gwmock_noise.gwosc.models import FilterType, GwoscFilterConfig, GwoscNoiseConfig

__all__ = [
    "FilterType",
    "GwoscFilterConfig",
    "GwoscNoiseConfig",
    "GwoscNoiseFetcher",
    "GwoscSegmentFilter",
]
