"""Automatic discovery for built-in config-driven noise components."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from gwmock_noise.simulators.base import ConfigurableNoiseSimulator

if TYPE_CHECKING:
    from gwmock_noise.config.models import NoiseComponentConfig, NoiseConfig
    from gwmock_noise.simulators.protocol import NoiseSimulator

DISCOVERY_EXCLUDED_MODULES = {"base", "composite", "default", "protocol", "registry", "streaming"}


def _iter_concrete_subclasses(
    base_class: type[ConfigurableNoiseSimulator],
) -> set[type[ConfigurableNoiseSimulator]]:
    subclasses: set[type[ConfigurableNoiseSimulator]] = set()
    for subclass in base_class.__subclasses__():
        subclasses.update(_iter_concrete_subclasses(subclass))
        if inspect.isabstract(subclass):
            continue
        if not getattr(subclass, "auto_register", False):
            continue
        if not subclass.__module__.startswith("gwmock_noise.simulators."):
            continue
        subclasses.add(subclass)
    return subclasses


def _import_configurable_simulator_modules() -> None:
    package_dir = Path(__file__).resolve().parent
    for module_info in pkgutil.iter_modules([str(package_dir)]):
        module_name = module_info.name
        if module_name.startswith("_") or module_name in DISCOVERY_EXCLUDED_MODULES:
            continue
        importlib.import_module(f"gwmock_noise.simulators.{module_name}")


@lru_cache(maxsize=1)
def discover_configurable_simulators() -> tuple[type[ConfigurableNoiseSimulator], ...]:
    """Return all built-in component simulator classes."""
    _import_configurable_simulator_modules()
    simulators = sorted(
        _iter_concrete_subclasses(ConfigurableNoiseSimulator),
        key=lambda simulator_class: simulator_class.simulator_name,
    )

    seen_names: dict[str, type[ConfigurableNoiseSimulator]] = {}
    for simulator_class in simulators:
        existing = seen_names.get(simulator_class.simulator_name)
        if existing is not None and existing is not simulator_class:
            raise RuntimeError(
                f"Duplicate configurable simulator name {simulator_class.simulator_name!r}: "
                f"{existing.__module__}.{existing.__name__} and "
                f"{simulator_class.__module__}.{simulator_class.__name__}."
            )
        seen_names[simulator_class.simulator_name] = simulator_class

    return tuple(simulators)


def available_simulator_names() -> tuple[str, ...]:
    """Return the names accepted by ``NoiseComponentConfig.simulator``."""
    return tuple(sorted(simulator_class.simulator_name for simulator_class in discover_configurable_simulators()))


def build_component_simulator(component: NoiseComponentConfig, config: NoiseConfig) -> NoiseSimulator:
    """Build one runtime simulator from a generic component configuration."""
    named_simulators = {
        simulator_class.simulator_name: simulator_class for simulator_class in discover_configurable_simulators()
    }
    simulator_class = named_simulators.get(component.simulator)
    if simulator_class is None:
        available = ", ".join(available_simulator_names())
        raise ValueError(f"Unknown simulator component {component.simulator!r}. Available simulators: {available}.")
    return simulator_class.from_component(component, config)
