# Copyright (C) 2026 Leuven Gravity Institute
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

"""``require``: an optional package that is missing is reported with the extra that installs it."""

from __future__ import annotations

import importlib
import types

import pytest

from gwmock_noise._optional import require

pytestmark = pytest.mark.unit


def _failing_import(missing: str):
    """An ``import_module`` stand-in that fails as if ``missing`` were not installed."""

    def fake(name: str) -> types.ModuleType:
        raise ModuleNotFoundError(f"No module named {missing!r}", name=missing)

    return fake


def test_an_installed_module_is_returned():
    """An importable module comes back as the module itself."""
    assert require("json", "unused", "Anything") is importlib.import_module("json")


def test_a_missing_package_names_the_extra(monkeypatch):
    """The message names the feature, the package and the install command; ``name`` is kept."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("h5py"))

    with pytest.raises(ModuleNotFoundError, match=r"HDF5 output needs h5py") as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert "pip install 'gwmock-noise[hdf5]'" in str(caught.value)
    assert caught.value.name == "h5py"


def test_a_missing_enclosing_package_names_the_extra(monkeypatch):
    """Asking for ``gwpy.timeseries`` when ``gwpy`` itself is absent."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("gwpy"))

    with pytest.raises(ModuleNotFoundError, match=r"pip install 'gwmock-noise\[gwpy\]'"):
        require("gwpy.timeseries", "gwpy", "GWpy output")


def test_a_broken_installation_is_not_reported_as_missing(monkeypatch):
    """A module inside an installed package failing to import is re-raised unchanged."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("h5py._internal"))

    with pytest.raises(ModuleNotFoundError) as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert caught.value.name == "h5py._internal"
    assert "pip install" not in str(caught.value)


def test_a_missing_dependency_of_the_package_is_not_reported_as_the_package(monkeypatch):
    """A module the package needs, rather than the package, is re-raised unchanged."""
    monkeypatch.setattr(importlib, "import_module", _failing_import("numpy"))

    with pytest.raises(ModuleNotFoundError) as caught:
        require("h5py", "hdf5", "HDF5 output")

    assert caught.value.name == "numpy"
    assert "pip install" not in str(caught.value)
