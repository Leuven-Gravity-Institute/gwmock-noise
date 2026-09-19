"""Runtime glitch-model definitions."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np

from gwmock_noise.glitches.snr import SNRDistribution, draw_target_snr, normalize_snr, serialize_snr

TWO_PI = 2.0 * np.pi


@dataclass(slots=True)
class LogNormalAmplitudeDistribution:
    """Log-normal amplitude sampler parameterized by linear mean and std."""

    distribution: str = "lognormal"
    mean: float = 1.0
    std: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configured distribution parameters."""
        if self.distribution != "lognormal":
            raise ValueError("Only the 'lognormal' amplitude distribution is supported.")
        if self.mean <= 0.0:
            raise ValueError("amplitude distribution mean must be greater than zero.")
        if self.std < 0.0:
            raise ValueError("amplitude distribution std must be non-negative.")

    def sample(self, rng: np.random.Generator) -> float:
        """Draw one amplitude sample."""
        if self.std == 0.0:
            return self.mean

        sigma_squared = float(np.log1p((self.std**2) / (self.mean**2)))
        sigma = float(np.sqrt(sigma_squared))
        mu = float(np.log(self.mean) - (0.5 * sigma_squared))
        return float(rng.lognormal(mean=mu, sigma=sigma))


@dataclass(slots=True)
class NetworkCoherence:
    """Share one glitch model's events across the interferometers it applies to.

    Without it, every glitch model runs an independent Poisson process and an
    independent random stream per interferometer, so two interferometers never
    carry the same transient. That is the right default, and it is what the one
    measurement of the question says for widely separated sites: over O1 and O2 the
    LIGO blip population produced *no* coincidences inside the +/-15 ms window an
    astrophysical signal can occupy, and the coincidences found in wider windows
    matched the accidental expectation.

    It is not what a network of *co-located* interferometers is expected to do. Three
    interferometers sharing one site, one vacuum system and one seismic environment
    see a common environmental transient in all three at once, and a coincidence veto
    -- or a null stream -- assumes exactly the incoherence that such an event breaks.
    A model carrying this runs **one** Poisson process for the whole network, draws
    **one** waveform per event, and offers that waveform to each interferometer it
    applies to.

    **Participation is decided per interferometer, independently, and is never
    conditioned on how many others took the event.** Conditioning -- "keep only
    events that land in at least two interferometers" -- is the obvious way to write
    a coincident population, and it is the wrong one here: it makes what one
    interferometer's strain contains depend on which *other* interferometers the run
    happens to include, so the same channel stops being reproducible between a
    three-interferometer run and a two-interferometer one. Paired-geometry
    comparisons need that reproducibility, so an event that lands nowhere simply
    lands nowhere, and the multiplicity comes out Binomial rather than being imposed.

    Attributes:
        participation_probability: Probability that any one interferometer the model
            applies to receives a given network event. ``1.0`` -- the default -- puts
            every event in every one of them, which is the strongest coherence the
            model can express.
        amplitude_ratio_std: Standard deviation of a per-interferometer lognormal
            amplitude ratio with linear mean ``1.0``, applied to the shared waveform.
            ``0.0`` -- the default -- injects the identical strain into every
            participating interferometer.
    """

    participation_probability: float = 1.0
    amplitude_ratio_std: float = 0.0

    def __post_init__(self) -> None:
        """Validate the configured coherence parameters.

        Raises:
            ValueError: If the participation probability is outside ``[0, 1]`` or the
                amplitude-ratio spread is negative.
        """
        if not 0.0 <= self.participation_probability <= 1.0:
            raise ValueError("network participation_probability must lie between zero and one.")
        if self.amplitude_ratio_std < 0.0:
            raise ValueError("network amplitude_ratio_std must be non-negative.")

    def amplitude_ratio(self, rng: np.random.Generator) -> float:
        """Draw one interferometer's amplitude ratio for a shared waveform.

        Args:
            rng: The generator for this (event, interferometer) pair.

        Returns:
            The multiplier to apply to the shared waveform.
        """
        return LogNormalAmplitudeDistribution(mean=1.0, std=self.amplitude_ratio_std).sample(rng)

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this coherence specification."""
        return {
            "participation_probability": self.participation_probability,
            "amplitude_ratio_std": self.amplitude_ratio_std,
        }


def _normalize_network_coherence(value: Any) -> NetworkCoherence | None:
    """Normalize a glitch model's network-coherence specification.

    Args:
        value: ``None`` for an independent per-interferometer model, a
            :class:`NetworkCoherence`, or the mapping that configures one.

    Returns:
        The coherence specification, or ``None``.

    Raises:
        TypeError: If the value is neither ``None``, a mapping, nor a
            :class:`NetworkCoherence`.
    """
    if value is None or isinstance(value, NetworkCoherence):
        return value
    if not isinstance(value, dict):
        raise TypeError("glitch model network must be a mapping or a NetworkCoherence.")
    return NetworkCoherence(**value)


def _normalize_detector_selector(value: Any) -> tuple[str, ...] | None:
    """Normalize a glitch model's interferometer selector.

    ``None`` -- no selector -- means every interferometer in the run, which is what
    every model meant before selectors existed, so a configuration written without
    one keeps its meaning exactly.

    A bare string is one interferometer rather than a sequence of characters.
    ``detectors="H1"`` is the obvious way to write a single-interferometer model,
    and reading it as a sequence would silently scope the model to interferometers
    named ``H`` and ``1`` -- which then fails the coverage check with a message
    about names nobody wrote.

    Args:
        value: The configured selector: ``None``, one name, or a list of names.

    Returns:
        The names as a tuple, or ``None`` for every interferometer.

    Raises:
        TypeError: If the selector is not a string or a list of strings.
        ValueError: If it names nothing, names a blank, or repeats a name.
    """
    if value is None:
        return None
    names = [value] if isinstance(value, str) else list(value) if isinstance(value, (list, tuple)) else None
    if names is None:
        raise TypeError("glitch model detectors must be a string or a list of strings.")
    for name in names:
        if not isinstance(name, str):
            raise TypeError("glitch model detectors must be a string or a list of strings.")
        if not name.strip():
            raise ValueError("glitch model detectors must be non-empty interferometer names.")
    if not names:
        raise ValueError(
            "glitch model detectors must name at least one interferometer; omit it to apply to all of them."
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"glitch model detectors names duplicate interferometers: {', '.join(repeated)}.")
    return tuple(names)


class GlitchDraw(NamedTuple):
    """One drawn glitch waveform and the parameters that produced it.

    What :meth:`GlitchModel.draw` returns so an injector can record *what it
    injected*, not merely that it injected something. Every field but the
    waveform is optional because not every model has the quantity: a parametric
    blip with no PSD has no SNR to report, and only the dataset-backed models
    draw a Gravity Spy class.

    Attributes:
        waveform: The glitch strain, sampled at the requested rate. Its first
            sample lands at the injection time -- the waveform *starts* there,
            it does not peak there.
        amplitude: The amplitude multiplier drawn from the model's amplitude
            distribution, or ``None`` when the model does not draw one.
        glitch_class: The drawn morphological class (e.g. a Gravity Spy class
            name), or ``None`` for a model with a single morphology.
        target_snr: The optimal SNR the draw was calibrated to *before* the
            amplitude multiplier, or ``None`` when the model does not calibrate
            SNR. This is the configured target, not what the strain holds.
        realized_snr: The optimal SNR of the whole of ``waveform`` against the
            PSD it was colored with, or ``None`` for an uncolored model. Only the
            end of the generated data can leave the strain holding less than
            this; a waveform crossing a segment boundary has its remainder
            injected into the next segment.

            **How it relates to ``target_snr`` is model-specific, so do not
            assume one from the other.** A model that calibrates against its own
            coloring PSD -- ``BlipGlitch``, ``ScatteredLightGlitch``,
            ``DeepExtractorGlitch`` -- realizes ``amplitude * target_snr``.
            ``GengliBlipGlitch`` does not: its target is sampled from a
            population and imposed by gengli on the *whitened* waveform, while
            the realized figure is measured on the colored, amplitude-scaled
            result, so the two are independent numbers rather than one restated.
    """

    waveform: np.ndarray
    amplitude: float | None = None
    glitch_class: str | None = None
    target_snr: float | None = None
    realized_snr: float | None = None


@dataclass(slots=True)
class GlitchModel:
    """Base dataclass for transient glitch generators.

    ``detectors`` scopes the model to a subset of the run's interferometers. It
    accepts one name or a list of them, and defaults to ``None``, which is every
    interferometer -- what a model without a selector has always meant. Scoping is
    what lets one configuration carry a 10 km model for one set of interferometers
    and a 15 km model for another, and per-interferometer rates for a network whose
    instruments are not equally glitchy (in O3, Fast_Scattering fired about 29 times
    more often in L1 than in H1).

    A selector is only half of it. :func:`validate_glitch_detector_coverage` refuses
    a run whose models do not between them account for every interferometer, or
    whose models color one interferometer against two different PSDs -- see its
    docstring for why proceeding was the defect.
    """

    rate: float
    amplitude_distribution: LogNormalAmplitudeDistribution
    #: Interferometers this model injects into, or ``None`` for all of them. Given
    #: as a name or a list of names; normalized to a tuple when the model is built.
    detectors: tuple[str, ...] | None = field(default=None, kw_only=True)
    #: How this model's events are shared across the interferometers it applies to.
    #: ``None`` -- the default -- runs an independent Poisson process and an
    #: independent random stream in each of them. A :class:`NetworkCoherence` runs one
    #: process for the whole network and offers each drawn waveform to every one of
    #: them; see that class for why participation is never conditioned on multiplicity.
    network: NetworkCoherence | None = field(default=None, kw_only=True)
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate common glitch parameters."""
        if self.rate < 0.0:
            raise ValueError("glitch rate must be non-negative.")
        self.detectors = _normalize_detector_selector(self.detectors)
        self.network = _normalize_network_coherence(self.network)

    def applies_to(self, detector: str) -> bool:
        """Return whether this model injects glitches into ``detector``.

        Args:
            detector: The interferometer name.

        Returns:
            ``True`` when the model has no selector, or names this interferometer.
        """
        return self.detectors is None or detector in self.detectors

    def coloring_reference(self) -> str | None:
        """Return a stable identity for the PSD this model colors against.

        ``None`` for a model that does no coloring, which therefore imposes no noise
        floor on the interferometers it claims and cannot contradict another model
        about one. Read from ``psd_file`` where the model has one, so a third-party
        model gets the same treatment as the built-in ones without restating the
        attribute name.

        Returns:
            The resolved PSD identity, or ``None`` when the model is uncolored.
        """
        from gwmock_noise.glitches._coloring import resolve_psd_reference  # noqa: PLC0415

        return resolve_psd_reference(getattr(self, "psd_file", None))

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a single glitch waveform."""
        raise NotImplementedError

    def _draw(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> GlitchDraw:
        """Draw one waveform together with the parameters that produced it.

        Where each built-in model's generation logic lives, so the waveform and
        its catalogue row come from one draw rather than two. ``draw`` dispatches
        here and ``generate_waveform`` returns just the waveform, so there is one
        implementation per model rather than one per entry point.
        """
        raise NotImplementedError

    def draw(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> GlitchDraw:
        """Generate one waveform together with the parameters that produced it.

        What the injector calls, so every event can be recorded rather than
        merely tallied.

        ``generate_waveform`` remains the extension point it has always been: a
        model that overrides it -- including a subclass of a built-in model --
        governs what is injected, and its draw is reported with the parameters
        blank rather than filled in from the superclass's draw, which produced a
        different waveform. Reversing that precedence would let an override
        change the strain while the catalogue described the code it replaced.
        """
        if self._waveform_override_supersedes_draw():
            return GlitchDraw(waveform=self.generate_waveform(sampling_frequency, rng=rng))
        return self._draw(sampling_frequency, rng=rng)

    def _waveform_override_supersedes_draw(self) -> bool:
        """Whether ``generate_waveform`` was overridden below the class defining ``_draw``.

        Compares the two methods' owning classes in the MRO rather than testing
        either against a fixed class, so it answers the question for a subclass
        of any model, built-in or not: an override sitting *under* the class that
        supplies the draw is a replacement of it, while one sitting at or above
        that class is the delegation the built-in models themselves install.
        """
        cls = type(self)
        waveform_owner = next(base for base in cls.__mro__ if "generate_waveform" in base.__dict__)
        draw_owner = next(base for base in cls.__mro__ if "_draw" in base.__dict__)
        return waveform_owner is not draw_owner and issubclass(waveform_owner, draw_owner)

    def resolve(self) -> str | None:
        """Pin any external, mutable dependency to an immutable version.

        Parametric models are fully specified by their configuration and have
        nothing external to resolve, so the base implementation is a no-op that
        returns ``None``. Models backed by a downloaded dataset (e.g.
        ``DeepExtractorGlitch``) override this to fetch and return the concrete
        version their run is pinned to. Exposing it on the base lets a caller
        pin every model in a heterogeneous glitch list uniformly.
        """
        return None

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return {
            "kind": self.kind,
            "rate": self.rate,
            "detectors": None if self.detectors is None else list(self.detectors),
            "network": None if self.network is None else self.network.serialize(),
            "amplitude_distribution": {
                "distribution": self.amplitude_distribution.distribution,
                "mean": self.amplitude_distribution.mean,
                "std": self.amplitude_distribution.std,
            },
        }


@dataclass(slots=True)
class BlipGlitch(GlitchModel):
    """Gaussian-windowed broadband burst."""

    width: float = 0.01
    psd_file: str | Path | None = None
    snr: float | SNRDistribution | dict[str, Any] | None = None
    low_frequency_cutoff: float = 2.0
    high_frequency_cutoff: float | None = None
    kind: Literal["blip"] = field(init=False, default="blip")
    _psd_frequencies: np.ndarray | None = field(init=False, default=None, repr=False)
    _psd_values: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate blip-specific parameters."""
        GlitchModel.__post_init__(self)
        if self.width <= 0.0:
            raise ValueError("blip width must be greater than zero.")
        if self.snr is not None:
            self.snr = normalize_snr(self.snr)
        from gwmock_noise.glitches._coloring import prepare_coloring  # noqa: PLC0415

        self._psd_frequencies, self._psd_values = prepare_coloring(
            self.psd_file, self.snr, self.low_frequency_cutoff, self.high_frequency_cutoff
        )

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a Gaussian-windowed white-noise burst."""
        return self._draw(sampling_frequency, rng=rng).waveform

    def _draw(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> GlitchDraw:
        """Draw a Gaussian-windowed white-noise burst and its parameters."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        half_span = max(1, int(np.ceil(3.0 * self.width * sampling_frequency)))
        sample_offsets = np.arange(-half_span, half_span + 1, dtype=float)
        sample_times = sample_offsets / sampling_frequency
        sigma = self.width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        envelope = np.exp(-0.5 * np.square(sample_times / sigma))

        carrier = generator.normal(size=envelope.size)
        carrier_std = float(np.std(carrier))
        if carrier_std > 0.0:
            carrier /= carrier_std

        amplitude = self.amplitude_distribution.sample(generator)
        base_waveform = carrier * envelope
        if self._psd_values is not None:
            from gwmock_noise.glitches._coloring import color_and_scale  # noqa: PLC0415

            # Drawn after the amplitude so a scalar target leaves the stream where it
            # has always been: a number consumes nothing from the generator, and only a
            # configured distribution advances it.
            target_snr = None if self.snr is None else draw_target_snr(self.snr, generator)
            scaled = color_and_scale(
                base_waveform,
                sampling_frequency=sampling_frequency,
                psd_frequencies=self._psd_frequencies,
                psd_values=self._psd_values,
                low_frequency_cutoff=self.low_frequency_cutoff,
                high_frequency_cutoff=self.high_frequency_cutoff,
                amplitude=amplitude,
                target_snr=target_snr,
            )
            return GlitchDraw(
                waveform=scaled.waveform,
                amplitude=amplitude,
                target_snr=target_snr,
                realized_snr=scaled.realized_snr,
            )
        return GlitchDraw(waveform=amplitude * base_waveform, amplitude=amplitude)

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return GlitchModel.serialize(self) | {
            "width": self.width,
            "psd_file": None if self.psd_file is None else str(self.psd_file),
            "snr": serialize_snr(self.snr),
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "high_frequency_cutoff": self.high_frequency_cutoff,
        }


@dataclass(slots=True)
class ScatteredLightGlitch(GlitchModel):
    """Arch-shaped scattered-light transient with a Gaussian envelope."""

    duration: float = 0.5
    peak_frequency: float = 24.0
    arch_exponent: float = 1.0
    phase: float = 0.0
    psd_file: str | Path | None = None
    snr: float | SNRDistribution | dict[str, Any] | None = None
    low_frequency_cutoff: float = 2.0
    high_frequency_cutoff: float | None = None
    kind: Literal["scattered_light"] = field(init=False, default="scattered_light")
    _psd_frequencies: np.ndarray | None = field(init=False, default=None, repr=False)
    _psd_values: np.ndarray | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate scattered-light parameters."""
        GlitchModel.__post_init__(self)
        if self.duration <= 0.0:
            raise ValueError("scattered-light duration must be greater than zero.")
        if self.peak_frequency <= 0.0:
            raise ValueError("scattered-light peak_frequency must be greater than zero.")
        if self.arch_exponent <= 0.0:
            raise ValueError("scattered-light arch_exponent must be greater than zero.")
        if self.snr is not None:
            self.snr = normalize_snr(self.snr)
        from gwmock_noise.glitches._coloring import prepare_coloring  # noqa: PLC0415

        self._psd_frequencies, self._psd_values = prepare_coloring(
            self.psd_file, self.snr, self.low_frequency_cutoff, self.high_frequency_cutoff
        )

    def generate_waveform(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Generate a chirping scattered-light glitch."""
        return self._draw(sampling_frequency, rng=rng).waveform

    def _draw(
        self,
        sampling_frequency: float,
        rng: np.random.Generator | None = None,
    ) -> GlitchDraw:
        """Draw a chirping scattered-light glitch and its parameters."""
        if sampling_frequency <= 0.0:
            raise ValueError("sampling_frequency must be greater than zero.")

        generator = np.random.default_rng() if rng is None else rng
        n_samples = max(1, round(self.duration * sampling_frequency))
        times = np.arange(n_samples, dtype=float) / sampling_frequency
        normalized_times = times / max(self.duration, 1.0 / sampling_frequency)
        centered_times = times - (0.5 * self.duration)
        envelope_sigma = self.duration / 6.0
        envelope = np.exp(-0.5 * np.square(centered_times / envelope_sigma))
        instantaneous_frequency = self.peak_frequency * np.power(
            np.abs(np.sin(np.pi * normalized_times)),
            self.arch_exponent,
        )
        phase = self.phase + (TWO_PI * np.cumsum(instantaneous_frequency) / sampling_frequency)
        amplitude = self.amplitude_distribution.sample(generator)
        base_waveform = envelope * np.sin(phase)
        if self._psd_values is not None:
            from gwmock_noise.glitches._coloring import color_and_scale  # noqa: PLC0415

            # Drawn after the amplitude so a scalar target leaves the stream where it
            # has always been: a number consumes nothing from the generator, and only a
            # configured distribution advances it.
            target_snr = None if self.snr is None else draw_target_snr(self.snr, generator)
            scaled = color_and_scale(
                base_waveform,
                sampling_frequency=sampling_frequency,
                psd_frequencies=self._psd_frequencies,
                psd_values=self._psd_values,
                low_frequency_cutoff=self.low_frequency_cutoff,
                high_frequency_cutoff=self.high_frequency_cutoff,
                amplitude=amplitude,
                target_snr=target_snr,
            )
            return GlitchDraw(
                waveform=scaled.waveform,
                amplitude=amplitude,
                target_snr=target_snr,
                realized_snr=scaled.realized_snr,
            )
        return GlitchDraw(waveform=amplitude * base_waveform, amplitude=amplitude)

    def serialize(self) -> dict[str, Any]:
        """Return metadata-friendly model parameters."""
        return GlitchModel.serialize(self) | {
            "duration": self.duration,
            "peak_frequency": self.peak_frequency,
            "arch_exponent": self.arch_exponent,
            "phase": self.phase,
            "psd_file": None if self.psd_file is None else str(self.psd_file),
            "snr": serialize_snr(self.snr),
            "low_frequency_cutoff": self.low_frequency_cutoff,
            "high_frequency_cutoff": self.high_frequency_cutoff,
        }


def supported_glitch_kinds() -> dict[str, type[GlitchModel]]:
    """Return all supported glitch-model kinds."""
    return {
        "blip": BlipGlitch,
        "scattered_light": ScatteredLightGlitch,
        "gengli_blip": importlib.import_module("gwmock_noise.glitches.gengli").GengliBlipGlitch,
        "deepextractor": importlib.import_module("gwmock_noise.glitches.deepextractor").DeepExtractorGlitch,
    }


def _parse_amplitude_distribution(value: Any) -> LogNormalAmplitudeDistribution:
    """Normalize glitch amplitude-distribution inputs."""
    if isinstance(value, LogNormalAmplitudeDistribution):
        return value
    if not isinstance(value, dict):
        raise TypeError("amplitude_distribution must be a mapping or LogNormalAmplitudeDistribution.")
    return LogNormalAmplitudeDistribution(**value)


def _scoped_to_absent_interferometers(models: Sequence[GlitchModel], known: set[str]) -> list[tuple[int, list[str]]]:
    """Return each model's selector entries naming an interferometer not in the run.

    Args:
        models: The run's glitch models.
        known: The interferometers the run actually has.

    Returns:
        One ``(model index, names)`` pair per offending model, in configuration order.
    """
    offenders = []
    for index, model in enumerate(models):
        if model.detectors is None:
            continue
        absent = [name for name in model.detectors if name not in known]
        if absent:
            offenders.append((index, absent))
    return offenders


def _coloring_claims(models: Sequence[GlitchModel], detectors: Sequence[str]) -> dict[str, dict[str, str]]:
    """Map each interferometer to the coloring PSDs claiming it, by resolved identity.

    The value is keyed by the resolved identity -- what decides whether two models
    name the same curve -- and holds the configured spelling, which is what a
    message has to quote back: a reader recognizes ``ET_15_full_cryo_psd``, not the
    installed path it resolves to.

    Args:
        models: The run's glitch models.
        detectors: The interferometers the run will generate.

    Returns:
        For each interferometer any coloring model claims, its PSDs keyed by resolved
        identity. An interferometer claimed only by uncolored models is absent.
    """
    claims: dict[str, dict[str, str]] = {}
    for model in models:
        reference = model.coloring_reference()
        if reference is None:
            continue
        configured = str(getattr(model, "psd_file", reference))
        for detector in detectors:
            if model.applies_to(detector):
                claims.setdefault(detector, {}).setdefault(reference, configured)
    return claims


def validate_glitch_detector_coverage(models: Sequence[GlitchModel], detectors: Sequence[str]) -> None:
    """Refuse a glitch configuration that does not say what every interferometer gets.

    A glitch model used to apply to every interferometer in the run because there
    was no way to say otherwise, so one ``psd_file`` colored all of them. A run
    naming both Einstein Telescope designs resolves to interferometers of two arm
    lengths, and the 10 km curve was applied to the 15 km instruments as well: SNRs
    23-44% away from what the configuration asked for, morphology-dependent, with
    nothing in the output or the logs to say so. **The silence was the defect**, so
    a configuration that cannot be read one way is refused here rather than run.

    Three things are refused, in the order a reader can act on them:

    1. **A selector naming an interferometer the run does not have.** Usually a typo
       or a leftover from another network, and the model then never fires -- a
       configured rate and PSD that inject nothing, which nothing in the output
       distinguishes from a model that fired and drew no events.
    2. **An interferometer no model claims.** Its strain is written with no glitches
       while its neighbours carry them. Where that is the intent, say it: a model
       scoped to it with ``rate = 0.0`` states "this interferometer carries no
       glitches" in the configuration, where a reader can see it.
    3. **Two coloring PSDs claiming one interferometer.** An interferometer has one
       noise floor; two models coloring it against different curves disagree about
       what instrument it is, and whichever ran second would not make that true.
       Models that do no coloring impose no floor and are not part of this.

    Args:
        models: The run's glitch models, already normalized.
        detectors: The interferometers the run will generate.

    Raises:
        ValueError: If any of the three cases above holds. The message names the
            interferometers involved.
    """
    run_detectors = list(dict.fromkeys(detectors))

    absent = _scoped_to_absent_interferometers(models, set(run_detectors))
    if absent:
        details = "; ".join(f"model {index} names {', '.join(names)}" for index, names in absent)
        raise ValueError(
            f"glitch models are scoped to interferometers this run does not have: {details}. "
            f"The run's interferometers are {', '.join(run_detectors)}."
        )

    uncovered = [detector for detector in run_detectors if not any(model.applies_to(detector) for model in models)]
    if uncovered:
        raise ValueError(
            f"no glitch model applies to {', '.join(uncovered)}, so this run would write their strain "
            "without glitches while the rest of the network carries them. Cover every interferometer: "
            "drop a model's detectors selector so it applies to all of them, extend a selector to name "
            "them, or -- if they genuinely carry no glitches -- add a model scoped to them with "
            "rate = 0.0, so the configuration says so."
        )

    conflicts = {
        detector: sorted(claims.values())
        for detector, claims in _coloring_claims(models, run_detectors).items()
        if len(claims) > 1
    }
    if conflicts:
        details = "; ".join(f"{detector} against {', '.join(files)}" for detector, files in sorted(conflicts.items()))
        raise ValueError(
            f"glitch models color one interferometer against more than one PSD: {details}. An "
            "interferometer has a single noise floor, so scope each model with detectors to the "
            "interferometers its psd_file describes."
        )


def normalize_glitch_models(value: Any) -> list[GlitchModel]:
    """Normalize heterogeneous glitch-config inputs.

    Entries carry an optional ``detectors`` selector -- one interferometer name or a
    list of them -- which scopes the model to those interferometers; an entry
    without one applies to every interferometer in the run, as it always has.
    Whether the resulting list accounts for the run is
    :func:`validate_glitch_detector_coverage`'s question, because it is the run that
    supplies the interferometers.

    Args:
        value: The configured list of models or model mappings.

    Returns:
        The normalized glitch models.

    Raises:
        TypeError: If the value is not a list, or an entry is not a mapping.
        ValueError: If an entry names an unsupported kind or omits its amplitude
            distribution.
    """
    if not isinstance(value, list):
        raise TypeError("glitch component options must provide a list of models.")

    kinds = supported_glitch_kinds()
    normalized: list[GlitchModel] = []
    for entry in value:
        if isinstance(entry, GlitchModel):
            normalized.append(entry)
            continue
        if not isinstance(entry, dict):
            raise TypeError("glitch model entries must be mappings or GlitchModel instances.")

        kind = entry.get("kind")
        if kind not in kinds:
            raise ValueError(f"glitch kind must be one of {', '.join(sorted(kinds))}.")

        parsed = dict(entry)
        if "amplitude_distribution" not in parsed:
            raise ValueError("glitch configs require an amplitude_distribution mapping.")
        parsed["amplitude_distribution"] = _parse_amplitude_distribution(parsed["amplitude_distribution"])
        parsed.pop("kind", None)
        normalized.append(kinds[kind](**parsed))
    return normalized
