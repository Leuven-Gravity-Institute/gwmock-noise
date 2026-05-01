"""Output adapters for gwmock-noise."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_OPTIONAL_EXPORTS = {
    "FrameWriter": "gwmock_noise.output.frame",
    "GWpyAdapter": "gwmock_noise.output.gwpy",
}

__all__ = list(_OPTIONAL_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve optional adapter exports."""
    module_name = _OPTIONAL_EXPORTS.get(name)
    if module_name is not None:
        export = getattr(import_module(module_name), name)
        globals()[name] = export
        return export
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
