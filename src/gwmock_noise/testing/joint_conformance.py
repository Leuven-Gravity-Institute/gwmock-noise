"""A shared conformance suite for :class:`JointStrainWitnessSimulator` backends.

Every package that registers a joint strain+witness backend can run exactly
the same contract checks against it, under its own test runner. gwmock-noise
runs this suite against its public dummy backend; a downstream package runs it
against its own backends.

Usage, in a consumer's test module::

    import pytest
    from gwmock_noise import load_joint_backend
    from gwmock_noise.testing.joint_conformance import JointBackendCase, JointBackendConformance


    def _build(seed):
        return load_joint_backend("my_backend")(sampling_frequency=64.0, seed=seed)


    @pytest.fixture(params=[JointBackendCase("my_backend", _build, 64.0, "my_backend")], ids=lambda case: case.name)
    def joint_backend(request):
        return request.param


    class TestMyBackendConformance(JointBackendConformance):
        pass

The suite checks the realization and metadata shapes, seeded determinism,
stream continuity, a Hermitian positive-semidefinite covariance over exactly
the realized channels, JSON-native provenance and metadata, refusal of an
unknown channel, and that no witness is disguised as a strain channel or
carried as an array in metadata.

This module requires ``pytest``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    JointCovariance,
    JointRealization,
    JointStrainWitnessSimulator,
)

#: Relative floor on eigenvalues of the normalized covariance slices.
PSD_FLOOR = 1e-10

#: Longest numeric sequence a metadata record may legitimately hold: a 3-vector
#: (position, orientation) or a ``[re, im]`` pair.
MAX_METADATA_VECTOR = 3


@dataclass(frozen=True)
class JointBackendCase:
    """One joint backend under test, and how to build it.

    Attributes:
        name: Label for the case, used in test ids.
        build: Builds a fresh simulator; called with the construction seed, or
            ``None`` when the seed is passed to the generation call instead.
        sampling_frequency: The sampling frequency the built simulator runs at.
        provenance_backend: The value ``provenance["backend"]`` must carry.
        chunk_samples: Samples per generated chunk.
    """

    name: str
    build: Callable[[int | None], JointStrainWitnessSimulator]
    sampling_frequency: float
    provenance_backend: str
    chunk_samples: int = 64

    @property
    def chunk_duration(self) -> float:
        """Duration of one chunk in seconds."""
        return self.chunk_samples / self.sampling_frequency

    def generate(
        self, simulator: JointStrainWitnessSimulator, seed: int | None = None, chunks: int = 1
    ) -> JointRealization:
        """Generate ``chunks`` chunks' worth of the simulator's own channels in one call."""
        return simulator.generate_joint(
            chunks * self.chunk_duration,
            self.sampling_frequency,
            list(simulator.detectors),
            list(simulator.witnesses),
            seed=seed,
        )


def array_leaks(value: Any, path: str = "") -> list[str]:
    """Return the paths in a metadata/provenance tree that hold channel-like data.

    Flags any :class:`numpy.ndarray`, and any list or tuple of more than
    :data:`MAX_METADATA_VECTOR` numbers. Sequences of names (channel orders)
    are not data and are walked, not flagged.

    Args:
        value: The tree to walk.
        path: The path of ``value`` within the enclosing tree.

    Returns:
        One description per leaking path; empty when nothing leaks.
    """
    if isinstance(value, np.ndarray):
        return [f"{path or '<root>'}: ndarray{value.shape}"]
    if isinstance(value, dict):
        return [leak for key, item in value.items() for leak in array_leaks(item, f"{path}/{key}")]
    if isinstance(value, (list, tuple)):
        numeric = len(value) > MAX_METADATA_VECTOR and all(
            isinstance(item, (int, float, complex)) and not isinstance(item, bool) for item in value
        )
        own = [f"{path}: {len(value)} numbers"] if numeric else []
        return own + [leak for index, item in enumerate(value) for leak in array_leaks(item, f"{path}[{index}]")]
    return []


class JointBackendConformance:
    """The contract every joint backend must meet.

    Subclass it under a ``Test``-prefixed name and provide a ``joint_backend``
    fixture yielding :class:`JointBackendCase` instances.

    Beyond the structural protocol, the suite holds a backend to these
    conventions: ``provenance`` records ``backend``, ``package_version``,
    ``seed`` and ``rng_bit_generator``; ``metadata`` lists the channel names
    under ``detectors`` and ``witnesses``; and a channel name the simulator was
    not built with raises a :class:`ValueError` whose message says the change
    is unsupported.
    """

    def test_satisfies_the_public_protocol(self, joint_backend: JointBackendCase) -> None:
        """The runtime-checkable public protocol recognizes the backend."""
        assert isinstance(joint_backend.build(1), JointStrainWitnessSimulator)

    def test_realization_shape_dtype_and_channel_sets(self, joint_backend: JointBackendCase) -> None:
        """Strain and witness dicts hold exactly the requested channels as finite 1-D float64 arrays."""
        simulator = joint_backend.build(1)
        realization = joint_backend.generate(simulator, seed=1)
        assert isinstance(realization, JointRealization)
        assert set(realization.strain) == set(simulator.detectors)
        assert set(realization.witness) == set(simulator.witnesses)
        for series in (*realization.strain.values(), *realization.witness.values()):
            assert series.dtype == np.float64
            assert series.shape == (joint_backend.chunk_samples,)
            assert np.all(np.isfinite(series))

    def test_channel_metadata_is_typed_and_complete(self, joint_backend: JointBackendCase) -> None:
        """Every realized channel has typed metadata with the right kind, domain, rate and dtype."""
        simulator = joint_backend.build(1)
        realization = joint_backend.generate(simulator, seed=1)
        metadata = realization.channel_metadata
        assert set(metadata) == set(realization.strain) | set(realization.witness)
        for name, record in metadata.items():
            expected_kind = ChannelKind.STRAIN if name in realization.strain else ChannelKind.WITNESS
            assert record.name == name
            assert record.kind is expected_kind
            assert record.domain is ChannelDomain.TIME
            assert record.sampling_frequency == joint_backend.sampling_frequency
            assert record.dtype == "float64"
            assert isinstance(record.unit, str)
            assert record.unit

    def test_seeded_generation_is_deterministic(self, joint_backend: JointBackendCase) -> None:
        """Equal seeds give bit-identical realizations; a different seed changes the strain."""
        first = joint_backend.generate(joint_backend.build(None), seed=7)
        second = joint_backend.generate(joint_backend.build(None), seed=7)
        other = joint_backend.generate(joint_backend.build(None), seed=8)
        for name in first.strain:
            np.testing.assert_array_equal(first.strain[name], second.strain[name])
            assert not np.array_equal(first.strain[name], other.strain[name])
        for name in first.witness:
            np.testing.assert_array_equal(first.witness[name], second.witness[name])

    def test_stream_chunks_continue_the_batch(self, joint_backend: JointBackendCase) -> None:
        """Four seeded stream chunks concatenate to the seeded batch over the same span."""
        batch = joint_backend.generate(joint_backend.build(None), seed=21, chunks=4)
        simulator = joint_backend.build(None)
        stream = simulator.generate_joint_stream(
            joint_backend.chunk_duration,
            joint_backend.sampling_frequency,
            list(simulator.detectors),
            list(simulator.witnesses),
            seed=21,
        )
        chunks = [next(stream) for _ in range(4)]
        for name in batch.strain:
            np.testing.assert_array_equal(np.concatenate([c.strain[name] for c in chunks]), batch.strain[name])
        for name in batch.witness:
            np.testing.assert_array_equal(np.concatenate([c.witness[name] for c in chunks]), batch.witness[name])

    def test_covariance_is_hermitian_psd_over_the_realized_channels(self, joint_backend: JointBackendCase) -> None:
        """The covariance indexes exactly the realized channels and is Hermitian PSD on an rfft grid."""
        simulator = joint_backend.build(1)
        covariance = simulator.covariance()
        assert isinstance(covariance, JointCovariance)
        assert set(covariance.channel_order) == set(simulator.detectors) | set(simulator.witnesses)
        assert len(covariance.channel_order) == len(set(covariance.channel_order))
        frequencies = covariance.frequencies
        assert frequencies[0] == 0.0
        assert np.all(np.diff(frequencies) > 0.0)
        assert frequencies[-1] == pytest.approx(joint_backend.sampling_frequency / 2.0)
        n = len(covariance.channel_order)
        assert covariance.matrix.shape == (frequencies.size, n, n)
        diagonal = np.real(np.diagonal(covariance.matrix, axis1=1, axis2=2))
        excited = np.all(diagonal > 0.0, axis=1)
        assert np.count_nonzero(excited) > frequencies.size // 2
        scale = 1.0 / np.sqrt(diagonal[excited])
        normalized = covariance.matrix[excited] * scale[:, :, None] * scale[:, None, :]
        hermitian_part = 0.5 * (normalized + np.conj(np.swapaxes(normalized, 1, 2)))
        assert np.max(np.abs(normalized - hermitian_part)) < 1e-12
        assert np.min(np.linalg.eigvalsh(hermitian_part)) > -PSD_FLOOR

    def test_provenance_and_metadata_are_json_native(self, joint_backend: JointBackendCase) -> None:
        """Provenance and metadata survive a strict JSON round trip and record backend, version, seed and RNG."""
        simulator = joint_backend.build(3)
        realization = joint_backend.generate(simulator, seed=3)
        provenance = realization.provenance
        assert json.loads(json.dumps(provenance)) == provenance
        assert json.loads(json.dumps(simulator.metadata)) == simulator.metadata
        assert provenance["backend"] == joint_backend.provenance_backend
        assert provenance["seed"] == 3
        assert provenance["package_version"]
        assert isinstance(provenance["rng_bit_generator"], str)
        assert provenance["rng_bit_generator"]

    def test_an_unknown_channel_is_refused(self, joint_backend: JointBackendCase) -> None:
        """An unknown detector or witness name is refused rather than silently generated."""
        simulator = joint_backend.build(1)
        fs = joint_backend.sampling_frequency
        duration = joint_backend.chunk_duration
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(duration, fs, ["NOT_A_DETECTOR"], list(simulator.witnesses), seed=1)
        with pytest.raises(ValueError, match="unsupported"):
            simulator.generate_joint(duration, fs, list(simulator.detectors), ["NOT_A_WITNESS"], seed=1)

    def test_witnesses_are_never_strain_channels(self, joint_backend: JointBackendCase) -> None:
        """Detector and witness names are disjoint, and every witness is typed and delivered as a witness."""
        simulator = joint_backend.build(1)
        realization = joint_backend.generate(simulator, seed=1)
        assert not set(simulator.detectors) & set(simulator.witnesses)
        assert not set(realization.strain) & set(simulator.witnesses)
        for name in simulator.witnesses:
            assert name in realization.witness
            assert realization.channel_metadata[name].kind is ChannelKind.WITNESS
        for name in simulator.detectors:
            assert realization.channel_metadata[name].kind is ChannelKind.STRAIN

    def test_no_channel_data_in_metadata_or_provenance(self, joint_backend: JointBackendCase) -> None:
        """Metadata, provenance and channel metadata hold no arrays and no channel-length sequences."""
        simulator = joint_backend.build(1)
        realization = joint_backend.generate(simulator, seed=1)
        for label, tree in (
            ("metadata", simulator.metadata),
            ("provenance", realization.provenance),
            ("channel_metadata", {name: vars(record) for name, record in realization.channel_metadata.items()}),
        ):
            assert array_leaks(tree) == [], label
        assert all(isinstance(name, str) for name in simulator.metadata["witnesses"])
        assert all(isinstance(name, str) for name in simulator.metadata["detectors"])


__all__ = ["MAX_METADATA_VECTOR", "PSD_FLOOR", "JointBackendCase", "JointBackendConformance", "array_leaks"]
