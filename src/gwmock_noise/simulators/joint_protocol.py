"""Structural protocol for joint strain-and-witness noise simulators.

This module defines an OPTIONAL extension of the legacy
:class:`~gwmock_noise.simulators.protocol.NoiseSimulator` contract. Nothing in
``NoiseSimulator`` changes: a caller that only knows about strain channels
keeps working exactly as before, and a simulator that never implements this
protocol is unaffected by its existence. A simulator MAY additionally satisfy
:class:`JointStrainWitnessSimulator` when it can generate one or more
auxiliary/witness channels alongside strain, jointly and with a documented
statistical relationship to the strain channels.

Fourier convention: wherever this protocol exposes a frequency-domain
quantity (:meth:`JointStrainWitnessSimulator.covariance`), the frequency grid
is the one-sided, non-negative grid produced by :func:`numpy.fft.rfftfreq`,
and a spectral density is the *one-sided* density -- i.e. the two-sided
density folded onto non-negative frequencies, consistent with the convention
already used by :class:`~gwmock_noise.simulators.multichannel.MultichannelNoiseSimulator`.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ChannelKind(StrEnum):
    """The physical role a channel plays in a joint realization."""

    STRAIN = "strain"
    WITNESS = "witness"


class ChannelDomain(StrEnum):
    """The sample domain a channel array is expressed in."""

    TIME = "time"
    FREQUENCY = "frequency"


@dataclass(frozen=True)
class ChannelMetadata:
    """Typed, machine-checkable description of one channel.

    Attributes:
        name: Channel name, unique within one joint realization.
        kind: Whether the channel is a gravitational-wave strain channel or
            an auxiliary/witness channel.
        unit: Physical unit of the channel's samples (e.g. ``"strain"`` for
            dimensionless strain, or a physical unit string such as
            ``"m/s^2"``).
        domain: Whether the channel's array is a time series or a
            frequency-domain series.
        sampling_frequency: Sampling frequency in hertz.
        dtype: The numpy dtype name (e.g. ``"float64"``) the channel array is
            generated in.
    """

    name: str
    kind: ChannelKind
    unit: str
    domain: ChannelDomain
    sampling_frequency: float
    dtype: str


@dataclass(frozen=True)
class JointRealization:
    """One simultaneous draw of strain and witness channels.

    Attributes:
        strain: Per-detector strain arrays, one entry per requested detector.
        witness: Per-channel witness arrays, one entry per requested witness
            channel.
        channel_metadata: Typed metadata for every channel in ``strain`` and
            ``witness``, keyed by the same channel names.
        provenance: Implementation-defined provenance describing how this
            realization was produced (backend identity, package version,
            seed, RNG algorithm, and any upstream configuration actually
            used).
    """

    strain: dict[str, np.ndarray]
    witness: dict[str, np.ndarray]
    channel_metadata: dict[str, ChannelMetadata]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class JointCovariance:
    """A one-sided cross-spectral density matrix over a set of channels.

    The convention matches :func:`numpy.fft.rfftfreq`: ``frequencies`` is the
    one-sided, non-negative frequency grid, and ``matrix[k]`` is the
    Hermitian ``(n_channels, n_channels)`` one-sided cross-spectral density at
    ``frequencies[k]``, in units of ``channel_unit**2 / Hz``. The diagonal
    entry ``matrix[k, i, i]`` is the one-sided PSD of channel
    ``channel_order[i]`` at that frequency.

    Attributes:
        frequencies: One-sided frequency grid in hertz, ``shape (n_freq,)``.
        matrix: Cross-spectral density matrices, ``shape (n_freq, n, n)``,
            complex-valued and Hermitian at every frequency.
        channel_order: Channel names, in the order ``matrix``'s last two axes
            index them.
    """

    frequencies: np.ndarray
    matrix: np.ndarray
    channel_order: list[str]


@runtime_checkable
class JointStrainWitnessSimulator(Protocol):
    """Structural interface for simulators producing joint strain+witness output.

    This is an optional extension of the legacy ``NoiseSimulator`` contract:
    implementing it changes nothing about ``NoiseSimulator`` itself, and a
    ``NoiseSimulator``-only consumer is unaffected by its existence. A class
    may satisfy both protocols at once.
    """

    duration: float
    sampling_frequency: float
    detectors: list[str]
    witnesses: list[str]
    seed: int | None

    def generate_joint(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> JointRealization:
        """Generate one simultaneous strain+witness realization."""

    def generate_joint_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> Iterator[JointRealization]:
        """Yield simultaneous strain+witness chunks lazily.

        Continuity contract: consecutive chunks from this iterator must equal
        the same realization a caller would obtain from one seeded
        :meth:`generate_joint` call spanning the combined duration -- the
        same contract ``NoiseSimulator.generate_stream`` makes for
        strain-only output.
        """

    def covariance(self) -> JointCovariance:
        """Return the joint one-sided cross-spectral density over all channels."""

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata (same role as ``NoiseSimulator.metadata``)."""


__all__ = [
    "ChannelDomain",
    "ChannelKind",
    "ChannelMetadata",
    "JointCovariance",
    "JointRealization",
    "JointStrainWitnessSimulator",
]
