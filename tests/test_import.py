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


_FIRST_IMPORT = """
import importlib, json, sys

failures = {}
for name in json.loads(sys.argv[1]):
    for loaded in [key for key in sys.modules if key == "gwmock_noise" or key.startswith("gwmock_noise.")]:
        del sys.modules[loaded]
    try:
        importlib.import_module(name)
    except Exception as exc:
        failures[name] = f"{type(exc).__name__}: {exc}"
print(json.dumps(failures))
"""


def test_every_submodule_imports_first():
    """Each submodule imports when it is the first of the package to be loaded.

    ``test_import_submodule`` imports them after the package, which has loaded most of them already, so
    a circular import between two submodules only fails when one of them is imported first: as
    ``gwmock_noise.spectral`` is by a caller that needs nothing else. One interpreter, with the
    package's modules cleared before each import, keeps third-party imports loaded and the test fast.
    """
    import json
    import subprocess
    import sys

    names = get_all_submodules(gwmock_noise)
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", _FIRST_IMPORT, json.dumps(names)], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout) == {}
