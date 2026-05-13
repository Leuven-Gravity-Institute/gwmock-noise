"""Base simulator interface for gravitational wave detector noise."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig
    from gwmock_noise.simulators.protocol import NoiseSimulator


@dataclass
class SimulationResult:
    """Result of a noise simulation run.

    Attributes:
        output_paths: Paths to generated output files, keyed by detector name.
        config: The configuration used for the simulation.
    """

    output_paths: dict[str, Path]
    config: NoiseConfig


class BaseNoiseSimulator(ABC):
    """Abstract base class for noise simulators.

    This interface is the stable API through which the upstream gwmock package
    interacts with gwmock_noise. Implementations must override :meth:`run`.
    """

    @abstractmethod
    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """


class ConfigurableNoiseSimulator(ABC):
    """Abstract mixin for built-in simulators usable as composed components."""

    simulator_name: ClassVar[str]
    auto_register: ClassVar[bool] = True

    @classmethod
    @abstractmethod
    def from_component(cls, component: NoiseComponentConfig, config: NoiseConfig) -> NoiseSimulator:
        """Construct one simulator instance from a generic component config."""
