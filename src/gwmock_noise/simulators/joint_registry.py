"""External entry-point discovery for joint strain+witness backends.

A separate, optional discovery surface from
:mod:`gwmock_noise.simulators.registry`, which only discovers this package's
own built-in :class:`~gwmock_noise.simulators.base.ConfigurableNoiseSimulator`
subclasses by scanning ``gwmock_noise.simulators``' own modules. This module
instead discovers joint strain+witness backend classes registered by ANY
installed distribution -- including a private package this package never
imports -- through the standard :func:`importlib.metadata.entry_points`
mechanism.

Nothing here imports, names, or depends on any specific third-party or
private package. A backend becomes discoverable purely by declaring itself
under the :data:`JOINT_BACKEND_ENTRY_POINT_GROUP` entry-point group in its own
package metadata; this module never needs to know the backend's package
exists ahead of time, and loading a backend only imports the module the
*discovered package itself* named.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.metadata import EntryPoint, entry_points
from typing import Any

#: The entry-point group name a package registers a joint strain+witness
#: backend class under, e.g. in ``pyproject.toml``::
#:
#:     [project.entry-points."gwmock_noise.joint_backends"]
#:     my_backend = "my_package.module:MyBackendClass"
JOINT_BACKEND_ENTRY_POINT_GROUP = "gwmock_noise.joint_backends"


@lru_cache(maxsize=1)
def discover_joint_backends() -> tuple[EntryPoint, ...]:
    """Return the entry points registered under :data:`JOINT_BACKEND_ENTRY_POINT_GROUP`.

    This performs discovery only -- it does not import any backend's module.
    Call :meth:`importlib.metadata.EntryPoint.load` on a returned entry point
    (or use :func:`load_joint_backend`) to import and resolve the backend
    class it names.
    """
    return tuple(entry_points(group=JOINT_BACKEND_ENTRY_POINT_GROUP))


def available_joint_backend_names() -> tuple[str, ...]:
    """Return the registered joint backend names, sorted."""
    return tuple(sorted(entry_point.name for entry_point in discover_joint_backends()))


def load_joint_backend(name: str) -> Any:
    """Import and return the backend class registered under ``name``.

    Args:
        name: The entry-point name a package registered its backend under.

    Returns:
        The backend class (or factory callable) the entry point names.

    Raises:
        KeyError: If no backend is registered under ``name``.
    """
    for entry_point in discover_joint_backends():
        if entry_point.name == name:
            return entry_point.load()
    available = ", ".join(available_joint_backend_names())
    raise KeyError(f"Unknown joint backend {name!r}. Available: {available}.")


__all__ = [
    "JOINT_BACKEND_ENTRY_POINT_GROUP",
    "available_joint_backend_names",
    "discover_joint_backends",
    "load_joint_backend",
]
