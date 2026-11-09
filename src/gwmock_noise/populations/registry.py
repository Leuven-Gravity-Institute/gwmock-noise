"""Registry of named glitch populations.

A campaign pins a population by *name* in its manifest and by *digest* in its run stamp.
The name is what a reader recognises and the digest is what proves the name still means
what it meant, so the registry exists to make the first of those resolvable without the
campaign carrying a copy of the definition.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from gwmock_noise.populations import deepextractor_o3, et_o3_anchored
from gwmock_noise.populations.population import GlitchPopulation

_REGISTERED: dict[str, Callable[..., GlitchPopulation]] = {
    et_o3_anchored.POPULATION_NAME: et_o3_anchored.et_o3_anchored_population,
    deepextractor_o3.POPULATION_NAME: deepextractor_o3.deepextractor_o3_population,
}


class UnknownPopulationArgumentError(TypeError):
    """A population builder was given a setting it does not take."""


def available_glitch_populations() -> list[str]:
    """Return the registered population names, sorted."""
    return sorted(_REGISTERED)


def get_glitch_population(name: str, **arguments: Any) -> GlitchPopulation:
    """Return a registered population by name.

    Each call builds a fresh population, so a caller cannot mutate the registry's copy out
    from under the next caller. Two calls compare equal and digest identically.

    Args:
        name: The registered name, e.g. ``"et-o3-anchored-v1"``.
        **arguments: A run's own settings, passed to the population's builder (for example
            ``psd_file``). Without them the call returns the registered population.

    Returns:
        The population.

    Raises:
        KeyError: If no population is registered under that name. The message lists the
            names that are, because a typo is the likeliest cause and a bare ``KeyError``
            does not say what to type instead.
        UnknownPopulationArgumentError: If the population's builder does not take one of
            ``arguments``. A setting it takes but cannot use raises whatever the builder raises.
    """
    builder = _REGISTERED.get(name)
    if builder is None:
        raise KeyError(
            f"unknown glitch population {name!r}; registered populations are {available_glitch_populations()}."
        )
    try:
        inspect.signature(builder).bind(**arguments)
    except TypeError as error:
        raise UnknownPopulationArgumentError(
            f"glitch population {name!r} takes the settings {sorted(inspect.signature(builder).parameters)}, "
            f"not {sorted(set(arguments) - set(inspect.signature(builder).parameters))}."
        ) from error
    return builder(**arguments)
