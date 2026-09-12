"""Target-SNR specifications for PSD-calibrated glitch models.

A glitch model's ``snr`` fixes how loud its events are. A single number makes
every event of a class equally loud, which no measured glitch population is:
Gravity Spy classes are heavy-tailed, and for the heaviest of them the tail is
where the class's effect on a search or a classifier lives. A lognormal
``amplitude_distribution`` on top of a fixed target cannot stand in for that --
a lognormal has every moment finite, and a power law with an index below one
does not even have a mean.

So ``snr`` accepts a *distribution* as well as a number, and the target is drawn
per event from the model's own random stream. Two shapes are supported:

- :class:`PowerLawSNRDistribution` -- a power law above a threshold, written so
  a measured tail index goes in as it was measured.
- :class:`EmpiricalSNRDistribution` -- draws with replacement from a table of
  observed SNRs, for a caller who has the measurements themselves rather than a
  fit to them.

Both serialize back to the mapping that configures them, so a run replays from
its own metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HDF5_SUFFIXES = (".h5", ".hdf5")
SNR_DATASET = "snr"
# Any double below 1.0 is at most ``1 - 2**-53`` -- the spacing below 1.0 is
# ``2**-53`` -- so ``1 - u`` is never smaller than ``2**-53`` for a uniform drawn on
# [0, 1). That is a property of IEEE-754 doubles rather than of one generator, which
# is what makes the largest untruncated draw a bound rather than an observation.
UNIFORM_MANTISSA_BITS = 53


def _check_positive_number(value: Any, parameter: str) -> float:
    """Validate one finite, strictly positive scalar parameter."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{parameter} must be a number.")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{parameter} must be finite and greater than zero.")
    return float(value)


def _untruncated_headroom_bits(minimum: float) -> float:
    """Bits of exponent range an untruncated power-law draw has to fit into.

    Both halves of ``minimum * (1 - u) ** (-1 / alpha)`` have to stay representable,
    and they fail differently: the power is evaluated first and raises
    ``OverflowError`` on its own, while the multiplication that follows overflows
    silently to infinity. The binding constraint is therefore whichever is larger --
    the product when ``minimum`` is above 1, the power alone when it is below -- so
    the headroom is measured against that one.
    """
    return float(np.log2(np.finfo(float).max) - max(np.log2(minimum), 0.0))


def _smallest_representable_alpha(minimum: float) -> float:
    """The smallest ``alpha`` whose untruncated draws are all representable."""
    return UNIFORM_MANTISSA_BITS / _untruncated_headroom_bits(minimum)


def _validate_snr_samples(values: np.ndarray, source: str) -> np.ndarray:
    """Validate a table of SNR samples drawn from by the empirical distribution."""
    if values.ndim != 1:
        raise ValueError(f"{source} must be one-dimensional.")
    if values.size == 0:
        raise ValueError(f"{source} must contain at least one SNR sample.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{source} must contain only finite SNR samples.")
    if np.any(values <= 0.0):
        raise ValueError(f"{source} SNR samples must be greater than zero.")
    return values


def load_snr_samples(path: str | Path) -> np.ndarray:
    """Load a one-dimensional table of observed SNRs from disk.

    ``.h5``/``.hdf5`` files are read from their ``snr`` dataset -- the schema
    ``gwmock-noise build-blip-glitch-table`` writes -- and anything else is read
    as a whitespace-separated text column.

    Args:
        path: Path to the SNR table.

    Returns:
        The validated SNR samples.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the table is empty, not one-dimensional, or holds a
            non-finite or non-positive SNR.
    """
    samples_path = Path(path)
    if not samples_path.exists():
        raise FileNotFoundError(f"SNR sample file not found: {samples_path}")
    if samples_path.suffix.lower() in HDF5_SUFFIXES:
        import h5py  # noqa: PLC0415

        with h5py.File(samples_path, "r") as handle:
            if SNR_DATASET not in handle:
                raise ValueError(f"SNR sample file must contain the '{SNR_DATASET}' dataset: {samples_path}")
            values = np.asarray(handle[SNR_DATASET][...], dtype=float)
    else:
        values = np.atleast_1d(np.loadtxt(samples_path, dtype=float))
    return _validate_snr_samples(values, f"SNR sample file '{samples_path}'")


@dataclass(slots=True)
class SNRDistribution:
    """Base class for a sampled target-SNR distribution."""

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one target SNR."""
        raise NotImplementedError

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this distribution."""
        raise NotImplementedError


@dataclass(slots=True)
class PowerLawSNRDistribution(SNRDistribution):
    """Power-law target SNR above a threshold.

    With ``maximum`` unset the survival function above ``minimum`` goes as
    ``S(s) = (s / minimum) ** -alpha``, which is the convention a Hill or
    maximum-likelihood tail index is quoted in: a class measured to have a tail
    index of 1.34 above SNR 10 is configured as
    ``{distribution = "power_law", minimum = 10.0, alpha = 1.34}`` with no
    conversion. Note that ``alpha`` is the *survival* exponent, one less than the
    density's. Setting ``maximum`` renormalizes that survival onto
    ``[minimum, maximum]`` as ``(S(s) - S(maximum)) / (1 - S(maximum))``, which
    reaches zero at the cap rather than continuing past it.

    ``minimum`` is a threshold, not a fit to the whole population: the measured
    index describes the tail above it and says nothing about the bulk below, so
    a model configured this way reproduces the tail and replaces the bulk with
    the same power law continued down to the threshold.

    ``maximum`` truncates the draw. It is optional and unset by default, but for
    a heavy tail it is worth setting: with ``alpha`` below 1 the untruncated
    distribution has no finite mean, and a long enough run will eventually draw
    an SNR no detector could produce. The largest SNR observed for the class is
    the natural choice.

    Below ``alpha`` of about 0.052 it stops being optional and is required: the
    untruncated draw then runs off the top of the float range (see
    :func:`_untruncated_headroom_bits`), which a configuration cannot express and
    a run cannot use, so such a configuration is refused at construction rather
    than left to fail partway through a run. Every index in the measured
    range -- 0.40 for the heaviest Gravity Spy class, 3.11 for the lightest --
    is far above that, and is unaffected.

    Attributes:
        minimum: Threshold SNR; every draw is at or above it.
        alpha: Tail index of the survival function. Must be greater than zero.
        maximum: Optional upper truncation, strictly above ``minimum``.
        distribution: Discriminator naming this shape in a configuration.
    """

    minimum: float
    alpha: float
    maximum: float | None = None
    distribution: str = "power_law"

    def __post_init__(self) -> None:
        """Validate the configured power-law parameters."""
        if self.distribution != "power_law":
            raise ValueError("PowerLawSNRDistribution requires distribution='power_law'.")
        self.minimum = _check_positive_number(self.minimum, "snr distribution minimum")
        self.alpha = _check_positive_number(self.alpha, "snr distribution alpha")
        if self.maximum is not None:
            self.maximum = _check_positive_number(self.maximum, "snr distribution maximum")
            if self.maximum <= self.minimum:
                raise ValueError("snr distribution maximum must be greater than minimum.")
            # A truncated draw is bounded above by ``maximum`` itself, which is finite
            # by the check just made, so only the untruncated branch can run off the
            # top of the float range.
            return
        if UNIFORM_MANTISSA_BITS / self.alpha > _untruncated_headroom_bits(self.minimum):
            raise ValueError(self._unrepresentable_draw_message("can draw"))

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one target SNR by inverting the survival function.

        ``rng.random()`` returns a uniform on ``[0, 1)``, so the inverted
        survival ``minimum * (1 - u) ** (-1 / alpha)`` covers ``[minimum, inf)``
        and never divides by zero. Truncation rescales the same inversion onto
        ``[minimum, maximum]``.

        The untruncated branch is guarded rather than trusted. ``__post_init__``
        already refuses a configuration whose draws can leave the float range, so
        nothing built through it reaches the guard; what the guard covers is a
        value that got past that check anyway -- an instance mutated after
        construction, or a configuration sitting on the boundary where the bound
        is computed in logarithms and can be a rounding hair out. An unusable
        target has to stop here either way, because past this point it is a scale
        factor on a waveform and a number in a truth catalogue, where infinity is
        indistinguishable from a very loud glitch.
        """
        uniform = rng.random()
        if self.maximum is None:
            try:
                draw = float(self.minimum * (1.0 - uniform) ** (-1.0 / self.alpha))
            except OverflowError as exc:
                raise ValueError(self._unrepresentable_draw_message("drew")) from exc
            if not np.isfinite(draw):
                raise ValueError(self._unrepresentable_draw_message("drew"))
            return draw
        survival_at_maximum = (self.maximum / self.minimum) ** (-self.alpha)
        return float(self.minimum * (1.0 - uniform * (1.0 - survival_at_maximum)) ** (-1.0 / self.alpha))

    def _unrepresentable_draw_message(self, verb: str) -> str:
        """Explain a draw that runs off the top of the float range.

        One message for both the configuration that can produce such a draw and
        the draw itself, so the remedy it recommends cannot drift between the two
        places that recommend it.

        Args:
            verb: How the overflow is being reported -- ``"can draw"`` when a
                configuration is refused, ``"drew"`` when a sample is.

        Returns:
            The error message.
        """
        return (
            f"An untruncated power-law snr distribution with minimum={self.minimum} and "
            f"alpha={self.alpha} {verb} an SNR too large to represent as a float: the largest "
            f"draw is minimum * 2 ** (53 / alpha). Set a maximum to truncate the tail, or raise "
            f"alpha above {_smallest_representable_alpha(self.minimum):.4g}."
        )

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this distribution."""
        return {
            "distribution": "power_law",
            "minimum": self.minimum,
            "alpha": self.alpha,
            "maximum": self.maximum,
        }


@dataclass(slots=True)
class EmpiricalSNRDistribution(SNRDistribution):
    """Target SNR drawn with replacement from a table of observed SNRs.

    For a caller holding the measurements rather than a fit to them: the drawn
    population is exactly the supplied one, tail included, with no assumption
    about its shape. Supply the values inline as ``samples`` or point ``file``
    at a table on disk -- exactly one of the two.

    Attributes:
        samples: Observed SNRs given inline, or ``None`` when ``file`` is used.
        file: Path to a table of observed SNRs, or ``None`` when ``samples`` is
            used. Read by :func:`load_snr_samples`.
        distribution: Discriminator naming this shape in a configuration.
    """

    samples: Sequence[float] | None = None
    file: str | Path | None = None
    distribution: str = "empirical"
    _values: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate the configuration and load the SNR table."""
        if self.distribution != "empirical":
            raise ValueError("EmpiricalSNRDistribution requires distribution='empirical'.")
        if (self.samples is None) == (self.file is None):
            raise ValueError("The empirical snr distribution requires exactly one of 'samples' or 'file'.")
        if self.file is not None:
            self._values = load_snr_samples(self.file)
            return
        if isinstance(self.samples, (str, bytes)) or not isinstance(self.samples, Sequence):
            raise TypeError("snr distribution samples must be a sequence of numbers.")
        for value in self.samples:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("snr distribution samples must be a sequence of numbers.")
        self._values = _validate_snr_samples(
            np.asarray(self.samples, dtype=float), "The empirical snr distribution 'samples'"
        )

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one target SNR uniformly from the table, with replacement."""
        return float(self._values[int(rng.integers(0, self._values.size))])

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this distribution.

        Echoes whichever of the two inputs was configured, so a file-backed
        table replays as a reference to the file rather than as a copy of its
        contents inlined into the metadata.
        """
        if self.file is not None:
            return {"distribution": "empirical", "file": str(self.file)}
        return {"distribution": "empirical", "samples": [float(value) for value in self._values]}


SNR_DISTRIBUTION_KINDS: dict[str, type[SNRDistribution]] = {
    "power_law": PowerLawSNRDistribution,
    "empirical": EmpiricalSNRDistribution,
}


def is_snr_distribution_mapping(value: Any) -> bool:
    """Whether a mapping configures a distribution rather than per-class targets.

    A glitch class is never named ``distribution``, so the key's presence is what
    separates ``{"distribution": "power_law", ...}`` from ``{"Blip": 12.0, ...}``
    without having to know the class names here.
    """
    return isinstance(value, Mapping) and "distribution" in value


def parse_snr_distribution(value: Any, parameter: str = "snr") -> SNRDistribution:
    """Build a distribution from a configuration mapping.

    Args:
        value: Mapping carrying a ``distribution`` discriminator and that
            shape's parameters.
        parameter: Name of the configuration key, used in error messages.

    Returns:
        The constructed distribution.

    Raises:
        TypeError: If ``value`` is not a mapping.
        ValueError: If the discriminator names no supported shape.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"{parameter} distribution must be a mapping.")
    kind = value.get("distribution")
    if kind not in SNR_DISTRIBUTION_KINDS:
        raise ValueError(
            f"{parameter} distribution must be one of {', '.join(sorted(SNR_DISTRIBUTION_KINDS))}; got {kind!r}."
        )
    return SNR_DISTRIBUTION_KINDS[kind](**dict(value))


def normalize_snr(value: Any, parameter: str = "snr") -> float | SNRDistribution:
    """Normalize one target-SNR specification into a number or a distribution.

    Accepts what a configuration can hold for a single target: a number, a
    already-constructed :class:`SNRDistribution`, or a mapping describing one.

    Args:
        value: The configured specification.
        parameter: Name of the configuration key, used in error messages.

    Returns:
        The target SNR as a float, or the distribution to draw it from.

    Raises:
        TypeError: If ``value`` is neither a number nor a distribution.
        ValueError: If a numeric target is not finite and positive.
    """
    if isinstance(value, SNRDistribution):
        return value
    if is_snr_distribution_mapping(value):
        return parse_snr_distribution(value, parameter)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{parameter} values must be numbers or a distribution mapping.")
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{parameter} values must be finite and greater than zero.")
    return float(value)


def draw_target_snr(specification: float | SNRDistribution, rng: np.random.Generator) -> float:
    """Resolve one specification to the target SNR of a single event.

    A numeric specification consumes nothing from ``rng``, so adding this call
    to a model leaves every fixed-SNR realization the model has ever produced
    bit-for-bit unchanged.
    """
    if isinstance(specification, SNRDistribution):
        return specification.sample(rng)
    return float(specification)


def serialize_snr(specification: float | SNRDistribution | None) -> Any:
    """Return the metadata-friendly form of a target-SNR specification."""
    if isinstance(specification, SNRDistribution):
        return specification.serialize()
    return specification
