"""Public dummy backend implementing :class:`JointStrainWitnessSimulator`.

Produces jointly Gaussian, temporally white strain and witness channels that
share a fixed equicorrelation structure at every sample, so every
strain/witness pair is genuinely correlated by construction. It exists to
exercise and conformance-test the
:mod:`gwmock_noise.simulators.joint_protocol` contract with a backend that
carries no physical modelling and no spectral shaping: every channel is flat
(white) in frequency, drawn i.i.d. per sample from a multivariate normal with
a known, closed-form covariance -- so its cross-spectral density is exactly
known rather than fitted or estimated, and ``covariance()`` can be tested
against an exact analytic value rather than a statistical one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.simulators.joint_protocol import (
    ChannelDomain,
    ChannelKind,
    ChannelMetadata,
    JointCovariance,
    JointRealization,
)
from gwmock_noise.version import __version__

#: Default sampling frequency, in hertz.
DEFAULT_SAMPLING_FREQUENCY = 256.0
#: Default nominal generation duration, in seconds.
DEFAULT_DURATION = 4.0
#: Default equicorrelation coefficient between every pair of channels.
DEFAULT_COUPLING = 0.3
#: Per-channel per-sample variance, in ``unit**2``.
DEFAULT_SAMPLE_VARIANCE = 1.0
#: Number of one-sided frequency points ``covariance()`` reports on.
COVARIANCE_GRID_SIZE = 129


def _equicorrelation_matrix(n_channels: int, coupling: float) -> np.ndarray:
    """Return an ``n_channels x n_channels`` per-sample covariance matrix.

    Every off-diagonal entry is ``coupling`` and every diagonal entry is
    :data:`DEFAULT_SAMPLE_VARIANCE`. This form is positive definite for any
    ``n_channels >= 1`` whenever ``0.0 <= coupling < 1.0`` (eigenvalues are
    ``1 + (n_channels - 1) * coupling`` with multiplicity 1 and
    ``1 - coupling`` with multiplicity ``n_channels - 1``), so validating
    ``coupling`` once at construction is sufficient regardless of channel
    count.
    """
    matrix = np.full((n_channels, n_channels), coupling, dtype=float)
    np.fill_diagonal(matrix, DEFAULT_SAMPLE_VARIANCE)
    return matrix


class JointDummyCorrelatedSimulator:
    """Public dummy backend producing simultaneous, correlated strain and witness channels.

    Every channel is temporally white with unit per-sample variance and
    ``coupling`` correlation with every other channel; there is no spectral
    shaping and no physical model. The one-sided cross-spectral density is
    therefore flat and known in closed form: for per-sample covariance
    ``Sigma`` at sampling frequency ``fs``, the one-sided PSD/CSD is the
    frequency-independent matrix ``2 * Sigma / fs`` (the same
    covariance-to-PSD scale convention already used by
    :class:`~gwmock_noise.simulators.multichannel.MultichannelNoiseSimulator`,
    inverted).
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        detectors: list[str],
        witnesses: list[str],
        sampling_frequency: float = DEFAULT_SAMPLING_FREQUENCY,
        duration: float = DEFAULT_DURATION,
        seed: int | None = None,
        coupling: float = DEFAULT_COUPLING,
        strain_unit: str = "strain",
        witness_unit: str = "counts",
    ) -> None:
        """Initialize the dummy backend.

        Args:
            detectors: Strain channel names. Fixed for the lifetime of this
                instance; later calls may reorder but not change this set.
            witnesses: Witness channel names, disjoint from ``detectors``.
                Fixed for the lifetime of this instance in the same way.
            sampling_frequency: Sampling frequency in hertz.
            duration: Nominal generation duration in seconds.
            seed: Seed for the shared innovation generator.
            coupling: Equicorrelation coefficient applied between every pair
                of channels (strain-strain, witness-witness and
                strain-witness alike). Must satisfy ``0.0 <= coupling < 1.0``.
            strain_unit: Unit string recorded for strain channels.
            witness_unit: Unit string recorded for witness channels.

        Raises:
            ValueError: If ``detectors``/``witnesses`` are empty, contain
                duplicates, overlap each other, or if ``coupling`` is out of
                range.
        """
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if not witnesses:
            raise ValueError("witnesses must contain at least one witness channel.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicates.")
        if len(set(witnesses)) != len(witnesses):
            raise ValueError("witnesses must not contain duplicates.")
        if set(detectors) & set(witnesses):
            raise ValueError("detectors and witnesses must not share channel names.")
        if not (0.0 <= coupling < 1.0):
            raise ValueError("coupling must satisfy 0.0 <= coupling < 1.0.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")

        self._detectors = list(detectors)
        self._witnesses = list(witnesses)
        self._channel_order = self._detectors + self._witnesses

        self.detectors = list(self._detectors)
        self.witnesses = list(self._witnesses)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.seed = seed
        self.coupling = coupling
        self.strain_unit = strain_unit
        self.witness_unit = witness_unit

        self._sample_covariance = _equicorrelation_matrix(len(self._channel_order), coupling)
        self._cholesky = np.linalg.cholesky(self._sample_covariance)
        self._rng: np.random.Generator | None = None

    def _validate_channels(self, detectors: list[str], witnesses: list[str]) -> None:
        """Validate one call's requested channel lists against this instance's fixed sets."""
        if not detectors:
            raise ValueError("detectors must contain at least one detector.")
        if not witnesses:
            raise ValueError("witnesses must contain at least one witness channel.")
        if len(set(detectors)) != len(detectors):
            raise ValueError("detectors must not contain duplicates.")
        if len(set(witnesses)) != len(witnesses):
            raise ValueError("witnesses must not contain duplicates.")
        if set(detectors) != set(self._detectors):
            raise ValueError(
                "Changing the detector set after initialization is unsupported; "
                "use the same detector names as at initialization (reordering is allowed)."
            )
        if set(witnesses) != set(self._witnesses):
            raise ValueError(
                "Changing the witness set after initialization is unsupported; "
                "use the same witness names as at initialization (reordering is allowed)."
            )

    def _initialize_generator(self, seed: int | None) -> None:
        """Initialize the shared innovation generator."""
        self._rng = np.random.default_rng(seed)

    def generate_joint(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> JointRealization:
        """Generate one simultaneous, correlated strain+witness realization."""
        self._validate_channels(detectors, witnesses)
        if duration <= 0:
            raise ValueError("duration must be greater than zero.")
        if sampling_frequency <= 0:
            raise ValueError("sampling_frequency must be greater than zero.")
        n_samples = round(duration * sampling_frequency)
        if n_samples < 1:
            raise ValueError("duration and sampling_frequency must produce at least one sample.")

        if seed is not None or self._rng is None:
            self.seed = seed if seed is not None else self.seed
            self._initialize_generator(self.seed)

        innovations = self._rng.standard_normal((n_samples, len(self._channel_order)))
        realization_samples = innovations @ self._cholesky.T

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.witnesses = list(witnesses)

        channel_index = {name: index for index, name in enumerate(self._channel_order)}
        strain = {name: realization_samples[:, channel_index[name]].copy() for name in self.detectors}
        witness = {name: realization_samples[:, channel_index[name]].copy() for name in self.witnesses}
        return JointRealization(
            strain=strain,
            witness=witness,
            channel_metadata=self.channel_metadata,
            provenance=self._provenance(),
        )

    def generate_joint_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        witnesses: list[str],
        seed: int | None = None,
    ) -> Iterator[JointRealization]:
        """Yield simultaneous strain+witness chunks lazily, preserving RNG state."""
        while True:
            yield self.generate_joint(chunk_duration, sampling_frequency, detectors, witnesses, seed)
            seed = None

    def covariance(self) -> JointCovariance:
        """Return this backend's exact, closed-form one-sided cross-spectral density.

        Every channel is temporally white, so the returned matrix is
        frequency-independent: ``2 * Sigma / sampling_frequency`` at every
        frequency, where ``Sigma`` is the per-sample covariance matrix drawn
        from at construction. This is an exact analytic value, not a
        statistical estimate off a realization.
        """
        frequencies = np.fft.rfftfreq(2 * (COVARIANCE_GRID_SIZE - 1), d=1.0 / self.sampling_frequency)
        psd_matrix = (2.0 / self.sampling_frequency) * self._sample_covariance.astype(complex)
        matrix = np.broadcast_to(psd_matrix, (frequencies.size, *psd_matrix.shape)).copy()
        return JointCovariance(
            frequencies=frequencies,
            matrix=matrix,
            channel_order=list(self._channel_order),
        )

    @property
    def channel_metadata(self) -> dict[str, ChannelMetadata]:
        """Return typed metadata for every strain and witness channel."""
        metadata: dict[str, ChannelMetadata] = {}
        for name in self._detectors:
            metadata[name] = ChannelMetadata(
                name=name,
                kind=ChannelKind.STRAIN,
                unit=self.strain_unit,
                domain=ChannelDomain.TIME,
                sampling_frequency=self.sampling_frequency,
                dtype="float64",
            )
        for name in self._witnesses:
            metadata[name] = ChannelMetadata(
                name=name,
                kind=ChannelKind.WITNESS,
                unit=self.witness_unit,
                domain=ChannelDomain.TIME,
                sampling_frequency=self.sampling_frequency,
                dtype="float64",
            )
        return metadata

    def _provenance(self) -> dict[str, Any]:
        """Return provenance for the realization just generated."""
        return {
            "backend": "joint_dummy_correlated",
            "package": "gwmock_noise",
            "package_version": __version__,
            "seed": self.seed,
            "rng_bit_generator": "PCG64",
            "coupling": self.coupling,
            "channel_order": list(self._channel_order),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """Return simulator metadata."""
        return {
            "implementation": "joint_dummy_correlated",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "witnesses": list(self.witnesses),
            "seed": self.seed,
            "joint_dummy_correlated": {
                "coupling": self.coupling,
                "channel_order": list(self._channel_order),
                "model": "equicorrelated_white_gaussian",
            },
        }


__all__ = ["JointDummyCorrelatedSimulator"]
