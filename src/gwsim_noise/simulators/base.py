"""Base simulator interface for gravitational wave detector noise."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from gwsim_noise.config import NoiseConfig


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

    This interface is the stable API through which the upstream gwsim package
    interacts with gwsim_noise. Implementations must override :meth:`run`.
    """

    @abstractmethod
    def run(self, config: NoiseConfig) -> SimulationResult:
        """Run the noise simulation with the given configuration.

        Args:
            config: Validated noise simulation configuration.

        Returns:
            Result containing paths to generated outputs and the config used.
        """
        ...
