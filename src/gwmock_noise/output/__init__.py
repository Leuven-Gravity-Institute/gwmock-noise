"""Output adapters for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["GWpyAdapter"]


def __getattr__(name: str) -> Any:
    """Lazily resolve optional adapter exports."""
    if name == "GWpyAdapter":
        adapter = getattr(import_module("gwmock_noise.output.gwpy"), name)
        globals()[name] = adapter
        return adapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
