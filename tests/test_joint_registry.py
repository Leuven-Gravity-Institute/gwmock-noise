"""Conformance tests for external entry-point discovery of joint backends.

This is the acceptance-critical path: a downstream package (including a
private one this repository never imports) registers a
:class:`~gwmock_noise.simulators.joint_protocol.JointStrainWitnessSimulator`
backend under the ``gwmock_noise.joint_backends`` entry-point group, and
gwmock-noise discovers it purely through package metadata. These tests
exercise that mechanism using this repository's own public dummy backend,
registered the same way any other package -- private or public -- would
register one.
"""

from __future__ import annotations

import inspect

import pytest

import gwmock_noise.simulators.joint_registry as joint_registry_module
from gwmock_noise import (
    JOINT_BACKEND_ENTRY_POINT_GROUP,
    JointDummyCorrelatedSimulator,
    available_joint_backend_names,
    discover_joint_backends,
    load_joint_backend,
)


def test_entry_point_group_name_is_stable() -> None:
    """The published group name is the literal string other packages register against."""
    assert JOINT_BACKEND_ENTRY_POINT_GROUP == "gwmock_noise.joint_backends"


def test_discovery_module_never_imports_a_specific_backend_package() -> None:
    """The discovery module's source names no concrete backend package.

    This is the executable form of "discovery works without importing a
    private module": the discovery *mechanism* is generic over
    :func:`importlib.metadata.entry_points` and contains no hardcoded
    reference to this repository's own dummy backend, let alone to any
    private package. Anything it can discover, it discovers purely from
    installed package metadata.
    """
    source = inspect.getsource(joint_registry_module)
    assert "joint_dummy" not in source
    assert "JointDummyCorrelatedSimulator" not in source


def test_discover_joint_backends_finds_the_public_dummy() -> None:
    """The public dummy backend is discoverable via the standard entry-point call."""
    entry_points = discover_joint_backends()
    names = {entry_point.name for entry_point in entry_points}
    assert "dummy_correlated" in names


def test_available_joint_backend_names_is_sorted_and_deduplicated() -> None:
    """available_joint_backend_names reports each registered name once, sorted."""
    names = available_joint_backend_names()
    assert names == tuple(sorted(set(names)))
    assert "dummy_correlated" in names


def test_load_joint_backend_resolves_the_registered_class() -> None:
    """Loading the entry point by name imports and returns the backend class it names.

    The entry point's *value* string
    (``gwmock_noise.simulators.joint_dummy:JointDummyCorrelatedSimulator``)
    is what performs the import here -- this test module never imports
    ``joint_dummy`` directly, so a passing assertion demonstrates the
    discovery-then-load path end to end.
    """
    backend_class = load_joint_backend("dummy_correlated")
    assert backend_class is JointDummyCorrelatedSimulator


def test_load_joint_backend_raises_for_unknown_name() -> None:
    """An unregistered name raises with the available names listed."""
    with pytest.raises(KeyError, match="Unknown joint backend"):
        load_joint_backend("does_not_exist")
