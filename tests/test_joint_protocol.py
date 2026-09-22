"""Conformance tests for the joint strain+witness protocol contract.

These tests pin the *shape* of the contract itself -- independent of any one
backend -- and confirm it is additive: the legacy
:class:`~gwmock_noise.simulators.protocol.NoiseSimulator` protocol is
untouched, so nothing here can change what it accepts.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import numpy as np
import pytest

from gwmock_noise.simulators.joint_dummy import JointDummyCorrelatedSimulator
from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    ChannelMetadata,
    JointCovariance,
    JointRealization,
    JointStrainWitnessSimulator,
)
from gwmock_noise.simulators.protocol import NoiseSimulator


def test_channel_kind_values() -> None:
    """ChannelKind carries the two documented roles."""
    assert {member.value for member in ChannelKind} == {"strain", "witness"}


def test_channel_domain_values() -> None:
    """ChannelDomain carries the two documented sample domains."""
    assert {member.value for member in ChannelDomain} == {"time", "frequency"}


def test_channel_metadata_is_frozen() -> None:
    """ChannelMetadata instances are immutable value objects."""
    metadata = ChannelMetadata(
        name="H1",
        kind=ChannelKind.STRAIN,
        unit="strain",
        domain=ChannelDomain.TIME,
        sampling_frequency=256.0,
        dtype="float64",
    )
    with pytest.raises(AttributeError):
        metadata.name = "L1"  # type: ignore[misc]


def test_joint_realization_is_frozen() -> None:
    """JointRealization instances are immutable value objects."""
    realization = JointRealization(strain={}, witness={}, channel_metadata={}, provenance={})
    with pytest.raises(AttributeError):
        realization.strain = {}  # type: ignore[misc]


def test_joint_covariance_is_frozen() -> None:
    """JointCovariance instances are immutable value objects."""
    covariance = JointCovariance(frequencies=np.array([0.0]), matrix=np.zeros((1, 1, 1)), channel_order=["H1"])
    with pytest.raises(AttributeError):
        covariance.channel_order = ["L1"]  # type: ignore[misc]


def test_dummy_satisfies_joint_protocol_by_structure() -> None:
    """The dummy backend structurally satisfies the runtime-checkable protocol."""
    simulator = JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=["SEIS1"], sampling_frequency=64.0)
    assert isinstance(simulator, JointStrainWitnessSimulator)


def test_joint_protocol_is_independent_of_legacy_protocol() -> None:
    """A JointStrainWitnessSimulator need not satisfy the legacy NoiseSimulator protocol.

    The joint dummy backend has no ``generate``/``generate_stream`` pair
    (it has ``generate_joint``/``generate_joint_stream`` instead), which
    pins that the two protocols are independent, additive contracts rather
    than one extending the other structurally.
    """
    simulator = JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=["SEIS1"], sampling_frequency=64.0)
    assert not isinstance(simulator, NoiseSimulator)


def test_generate_joint_stream_is_a_generator_function() -> None:
    """generate_joint_stream exposes a real generator method, like the legacy protocol."""
    assert inspect.isgeneratorfunction(JointDummyCorrelatedSimulator.generate_joint_stream)


def test_dummy_generate_joint_stream_returns_iterator() -> None:
    """generate_joint_stream's return value satisfies Iterator[JointRealization]."""
    simulator = JointDummyCorrelatedSimulator(detectors=["H1"], witnesses=["SEIS1"], sampling_frequency=64.0)
    stream = simulator.generate_joint_stream(1.0, 64.0, ["H1"], ["SEIS1"], seed=1)
    assert isinstance(stream, Iterator)
    first = next(stream)
    assert isinstance(first, JointRealization)
