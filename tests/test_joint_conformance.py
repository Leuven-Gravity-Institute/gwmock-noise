"""Run the shared joint-backend conformance suite against the public dummy backend.

The backend is obtained by name through entry-point discovery, the same way a
downstream package's backend is, and is held to exactly the checks in
:mod:`gwmock_noise.testing.joint_conformance` that downstream packages run
against their own backends.
"""

from __future__ import annotations

import numpy as np
import pytest

from gwmock_noise import load_joint_backend
from gwmock_noise.simulators.joint_protocol import JointStrainWitnessSimulator
from gwmock_noise.testing.joint_conformance import JointBackendCase, JointBackendConformance, array_leaks


def _dummy(seed: int | None) -> JointStrainWitnessSimulator:
    return load_joint_backend("dummy_correlated")(
        detectors=["H1", "L1"], witnesses=["SEIS1", "SEIS2"], sampling_frequency=64.0, seed=seed
    )


@pytest.fixture(
    params=[JointBackendCase("dummy_correlated", _dummy, 64.0, "joint_dummy_correlated")],
    ids=lambda case: case.name,
)
def joint_backend(request: pytest.FixtureRequest) -> JointBackendCase:
    """The public dummy backend, loaded through entry-point discovery."""
    return request.param


class TestDummyConformance(JointBackendConformance):
    """The public dummy backend meets the shared joint-backend contract."""


def test_the_leak_detector_flags_arrays_and_numeric_lists() -> None:
    """The metadata check is not vacuous: it flags an ndarray and any short numeric series."""
    series = np.zeros(4)
    assert array_leaks({"witness": {"SEIS1": series}}) == ["/witness/SEIS1: ndarray(4,)"]
    assert array_leaks({"SEIS1": series.tolist()}) == ["/SEIS1: 4 numbers"]
    assert array_leaks({"nested": [{"samples": (1, 2.0, 3, 4)}]}) == ["/nested[0]/samples: 4 numbers"]
    legitimate = {"channel_order": ["H1", "L1", "SEIS1", "SEIS2"], "position": [0.0, 1.0, 2.0], "c": [1.0, 0.0]}
    assert array_leaks(legitimate) == []
