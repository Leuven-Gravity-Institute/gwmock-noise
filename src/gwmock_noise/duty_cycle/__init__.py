"""When each interferometer of a network observes: unplanned unlocks and planned maintenance."""

from __future__ import annotations

from gwmock_noise.duty_cycle.model import (
    DAY,
    DEFAULT_CORRELATED_DOWNTIME_FRACTION,
    DEFAULT_MAINTENANCE_FRACTION,
    DEFAULT_MAINTENANCE_OFFSET,
    DEFAULT_MEAN_DOWNTIME,
    DEFAULT_MEAN_UPTIME,
    HOUR,
    WEEK,
    DutyCycleModel,
    MaintenanceSchedule,
    correlated_unlock_share,
    rate_matrix,
    stationary_distribution,
)
from gwmock_noise.duty_cycle.simulate import (
    MAINTENANCE_BIT,
    OBSERVING_MASK,
    DutyCycleRealization,
    simulate_duty_cycle,
)

__all__ = [
    "DAY",
    "DEFAULT_CORRELATED_DOWNTIME_FRACTION",
    "DEFAULT_MAINTENANCE_FRACTION",
    "DEFAULT_MAINTENANCE_OFFSET",
    "DEFAULT_MEAN_DOWNTIME",
    "DEFAULT_MEAN_UPTIME",
    "HOUR",
    "MAINTENANCE_BIT",
    "OBSERVING_MASK",
    "WEEK",
    "DutyCycleModel",
    "DutyCycleRealization",
    "MaintenanceSchedule",
    "correlated_unlock_share",
    "rate_matrix",
    "simulate_duty_cycle",
    "stationary_distribution",
]
