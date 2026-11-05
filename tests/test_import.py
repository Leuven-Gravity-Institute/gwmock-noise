"""Test importing the gwmock-noise package and its modules."""

from __future__ import annotations

import pkgutil

import pytest

import gwmock_noise


def get_all_submodules(package):
    """Discover all submodules in the package.

    Args:
        package: The package to inspect.

    """
    submodules = []
    for _, mod_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        submodules.append(mod_name)
    return submodules


def test_import_main_package():
    """Test that the main gwmock_noise package can be imported."""
    assert hasattr(gwmock_noise, "__version__")
    assert gwmock_noise.__version__ is not None
    assert hasattr(gwmock_noise, "open_stream")


@pytest.mark.parametrize("module_name", get_all_submodules(gwmock_noise))
def test_import_submodule(module_name):
    """Test that all submodules can be imported."""
    __import__(module_name)


def _import_first(module_name: str) -> str | None:
    """Import ``module_name`` in a fresh interpreter; return the error's last line, or ``None``."""
    import subprocess
    import sys

    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {module_name}"], capture_output=True, text=True, check=False
    )
    return result.stderr.strip().splitlines()[-1] if result.returncode else None


def test_every_submodule_imports_first():
    """Each submodule imports when it is the first of the package to be loaded.

    ``test_import_submodule`` imports them after the package, which has loaded most of them already, so
    a circular import between two submodules only fails when one of them is imported first: as
    ``gwmock_noise.spectral`` is by a caller that needs nothing else. Each import runs in its own
    interpreter, so no module loaded for one can satisfy another; the interpreters run in parallel.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    names = get_all_submodules(gwmock_noise)
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        errors = dict(zip(names, pool.map(_import_first, names), strict=True))
    failing = sorted(name for name, error in errors.items() if error)
    assert failing == [], {name: errors[name] for name in failing}
