"""Configuration and fixtures for pytest."""

from __future__ import annotations

import pytest


def _h5py_installed() -> bool:
    """Whether h5py imports; a broken installation raises rather than reading as absent."""
    try:
        import h5py  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "h5py":
            raise
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the tests marked ``requires_h5py`` where h5py is not installed (Python 3.15 has no wheel yet)."""
    if _h5py_installed():
        return
    skip = pytest.mark.skip(reason="h5py is not installed (the hdf5 extra)")
    for item in items:
        if item.get_closest_marker("requires_h5py") is not None:
            item.add_marker(skip)


@pytest.fixture
def some_name() -> str:
    """Provide a string name.

    Returns:
        A string name.

    """
    return "developer"
