"""Registry of named glitch populations.

A campaign pins a population by *name* in its manifest and by *digest* in its run stamp.
The name is what a reader recognises and the digest is what proves the name still means
what it meant, so the registry exists to make the first of those resolvable without the
campaign carrying a copy of the definition.
"""

from __future__ import annotations

from collections.abc import Callable

from gwmock_noise.populations.et_o3_anchored import POPULATION_NAME, et_o3_anchored_population
from gwmock_noise.populations.population import GlitchPopulation

_REGISTERED: dict[str, Callable[[], GlitchPopulation]] = {
    POPULATION_NAME: et_o3_anchored_population,
}


def available_glitch_populations() -> list[str]:
    """Return the registered population names, sorted."""
    return sorted(_REGISTERED)


def get_glitch_population(name: str) -> GlitchPopulation:
    """Return a registered population by name.

    Each call builds a fresh population, so a caller cannot mutate the registry's copy out
    from under the next caller. Two calls compare equal and digest identically.

    Args:
        name: The registered name, e.g. ``"et-o3-anchored-v1"``.

    Returns:
        The population.

    Raises:
        KeyError: If no population is registered under that name. The message lists the
            names that are, because a typo is the likeliest cause and a bare ``KeyError``
            does not say what to type instead.
    """
    builder = _REGISTERED.get(name)
    if builder is None:
        raise KeyError(
            f"unknown glitch population {name!r}; registered populations are {available_glitch_populations()}."
        )
    return builder()
