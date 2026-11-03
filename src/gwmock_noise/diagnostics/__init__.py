"""Diagnostics helpers for gwmock-noise."""

from __future__ import annotations

from gwmock_noise.diagnostics.crossfade_covariance import (
    CovarianceError,
    CrossfadeCovarianceError,
    crossfade_covariance_error,
    crossfade_target,
    simulator_settings,
)
from gwmock_noise.diagnostics.psd import compare_psd, estimate_psd, plot_psd
from gwmock_noise.diagnostics.statistics import (
    DiagnosticResult,
    run_diagnostics,
    test_gaussianity,
    test_stationarity,
)

__all__ = [
    "CovarianceError",
    "CrossfadeCovarianceError",
    "DiagnosticResult",
    "compare_psd",
    "crossfade_covariance_error",
    "crossfade_target",
    "estimate_psd",
    "plot_psd",
    "run_diagnostics",
    "simulator_settings",
    "test_gaussianity",
    "test_stationarity",
]
