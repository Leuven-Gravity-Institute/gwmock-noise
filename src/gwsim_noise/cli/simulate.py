"""Command for running noise simulations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer


def simulate(
    config_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the configuration file (YAML or JSON).",
        ),
    ],
) -> None:
    """Run noise simulation from a configuration file.

    Loads the configuration, validates it, and runs the noise simulator.
    Output files are written to the directory specified in the config.
    """
    from gwsim_noise.config import load_config  # noqa: PLC0415
    from gwsim_noise.simulators import DefaultNoiseSimulator  # noqa: PLC0415

    logger = logging.getLogger("gwsim_noise")
    logger.info("Loading configuration from %s", config_path)
    config = load_config(config_path)
    logger.info("Running noise simulation for detectors: %s", config.detectors)
    simulator = DefaultNoiseSimulator()
    result = simulator.run(config)
    for detector, path in result.output_paths.items():
        logger.info("Wrote output for %s to %s", detector, path)
