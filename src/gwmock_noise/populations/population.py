"""Registered glitch populations: a named, serializable, digestible set of glitch classes.

A glitch model says how one class of transient is drawn. A *population* says which classes
a campaign injects, at what rates, into which interferometers, under one name and one
digest -- so that every arm of that campaign can be shown to have used the same model
rather than asserted to have.

Three things it adds over a bare list of models:

- **A digest.** :meth:`GlitchPopulation.digest` is a SHA-256 over the canonical
  serialization, so a run stamp can pin the population and a later run can prove it used
  the same one. Any change to any registered quantity changes it.
- **An expected-count decomposition.** :meth:`GlitchPopulation.expected_counts` returns the
  rate, the livetime, the detector-participation and the selection factor separately,
  alongside their product. A total that comes out wrong is then traceable to the factor
  that is wrong, which a single number is not.
- **A reusable realization.** :meth:`GlitchPopulation.realize` returns the glitch strain on
  its own, with no noise under it. Paired arms of a comparison add *that array*, so their
  glitch content is bit-identical by construction rather than by two generators agreeing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from gwmock_noise.glitches.models import GlitchModel, normalize_glitch_models, validate_glitch_detector_coverage
from gwmock_noise.glitches.snr import snr_survival
from gwmock_noise.simulators.glitches import InjectGlitches, _ZeroNoiseSimulator
from gwmock_noise.version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gwmock_noise.simulators.protocol import NoiseSimulator

#: Schema version of the serialized population. Bumped when the meaning of a field
#: changes, including when the shape does not -- a consumer reading a pinned population
#: has nothing else to read the change from. It is part of the digest, so a schema change
#: is visible as a digest change even when every registered number is untouched.
GLITCH_POPULATION_SCHEMA_VERSION = "1.0.0"


class GlitchRealization(NamedTuple):
    """One seeded glitch realization and the stamp that identifies it.

    Attributes:
        strain: The glitch contribution per interferometer, with nothing else in it. This
            is what paired arms add to their own base noise, and adding the same array to
            two arms is what makes their glitch content bit-identical.
        events: The per-event truth catalogue, in time order. See
            :data:`gwmock_noise.simulators.glitches.GLITCH_CATALOGUE_COLUMNS`.
        stamp: What a campaign manifest pins -- the population's name and digest, the
            package version, and the generation parameters. Enough to reproduce ``strain``
            and to prove another run used the same population.
    """

    strain: dict[str, np.ndarray]
    events: list[dict[str, Any]]
    stamp: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExpectedCount:
    """The expected injected count for one class in one interferometer, factorized.

    The product is :attr:`expected`, and every factor is carried beside it. A campaign that
    finds the total surprising can then read off *which* factor is surprising -- a rate
    transplanted from another run, a livetime that is calendar time rather than observing
    time, a participation probability nobody meant to set -- instead of re-deriving the
    whole chain.

    Attributes:
        glitch_class: Name of the class within its population.
        detector: The interferometer this row is for.
        rate_hz: The class's Poisson rate. For a class whose interferometers glitch
            independently this is the rate *each* of them sees; for a network-coherent
            class it is the rate of the shared network process, which is not the same
            number unless every interferometer takes every event.
        livetime_seconds: The analysed time this count is over.
        participation: Probability that this interferometer receives a given event of the
            class. One for a class with no network coherence.
        selection: Fraction of the class's events that survive the SNR cut the count was
            asked for. One when no cut was given.
        expected: ``rate_hz * livetime_seconds * participation * selection``.
    """

    glitch_class: str
    detector: str
    rate_hz: float
    livetime_seconds: float
    participation: float
    selection: float
    expected: float

    @property
    def network_events(self) -> float:
        """Expected number of *network* events of this class over the same livetime.

        The same arithmetic with the participation factor divided out: the events the
        class's process produced, rather than the subset this interferometer received. For
        a class with no network coherence the two coincide, because every event it draws
        for an interferometer is that interferometer's.
        """
        return self.rate_hz * self.livetime_seconds * self.selection


@dataclass(frozen=True, slots=True)
class GlitchClass:
    """One named class within a registered population.

    The prose fields are part of the registered definition and therefore part of the
    digest, deliberately. A population whose numbers are unchanged but whose *meaning* has
    been rewritten is a different population, and a digest that did not move would say
    otherwise.

    Attributes:
        name: Stable identifier of the class within its population.
        morphology: Short slug naming the waveform family, e.g.
            ``"gaussian-windowed-broadband-burst"``.
        description: One sentence saying what the class represents and what it is drawn
            from.
        model: The glitch model that generates it, already normalized.
    """

    name: str
    morphology: str
    description: str
    model: GlitchModel

    def __post_init__(self) -> None:
        """Validate the class definition.

        Raises:
            ValueError: If the name, morphology or description is blank.
        """
        for field_name in ("name", "morphology", "description"):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"glitch class {field_name} must be a non-empty string.")

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this class."""
        return {
            "name": self.name,
            "morphology": self.morphology,
            "description": self.description,
            "model": self.model.serialize(),
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> GlitchClass:
        """Rebuild a class from :meth:`serialize` output.

        Args:
            payload: The serialized class.

        Returns:
            The reconstructed class.

        Raises:
            KeyError: If a required key is missing.
        """
        (model,) = normalize_glitch_models([payload["model"]])
        return cls(
            name=payload["name"],
            morphology=payload["morphology"],
            description=payload["description"],
            model=model,
        )


@dataclass(frozen=True, slots=True)
class GlitchPopulation:
    """A named, versioned, digestible set of glitch classes.

    **The digest is only as portable as what goes into it.** A model's ``psd_file`` is
    serialized as written, so a population built against ``/scratch/me/et.txt`` carries
    that path into its digest and no other machine can reproduce it. Name a bundled preset,
    or a path relative to the campaign's own tree, and the digest travels.

    Attributes:
        name: Stable registered identifier, e.g. ``"et-o3-anchored-v1"``.
        description: One paragraph saying what the population is for and what it is
            anchored to.
        classes: The classes, in registration order.
        references: Published sources the registered quantities are anchored to, one
            citation per entry.
        unanchored: Quantities that could **not** be anchored to a published measurement,
            one per entry, each saying what it is and why. Carried in the population rather
            than only in its documentation, so a consumer reading a pinned population reads
            the caveats with it. An empty tuple is a claim that everything is anchored, so
            it is a claim to be able to defend.
        schema_version: Serialization schema, part of the digest.
    """

    name: str
    description: str
    classes: tuple[GlitchClass, ...]
    references: tuple[str, ...] = ()
    unanchored: tuple[str, ...] = ()
    schema_version: str = GLITCH_POPULATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate the population definition.

        Raises:
            ValueError: If the name or description is blank, the population has no classes,
                or two classes share a name.
        """
        if not self.name.strip():
            raise ValueError("glitch population name must be a non-empty string.")
        if not self.description.strip():
            raise ValueError("glitch population description must be a non-empty string.")
        if not self.classes:
            raise ValueError("a glitch population must define at least one class.")
        names = [entry.name for entry in self.classes]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"glitch population class names repeat: {', '.join(repeated)}.")

    def model_for(self, glitch_class: str) -> GlitchModel:
        """Return one class's glitch model by name.

        Args:
            glitch_class: The class name.

        Returns:
            The model that generates it.

        Raises:
            KeyError: If the population has no such class.
        """
        for entry in self.classes:
            if entry.name == glitch_class:
                return entry.model
        raise KeyError(f"glitch population {self.name!r} has no class {glitch_class!r}.")

    def models(self) -> list[GlitchModel]:
        """Return the population's glitch models in registration order."""
        return [entry.model for entry in self.classes]

    def serialize(self) -> dict[str, Any]:
        """Return the mapping that reconstructs this population."""
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "classes": [entry.serialize() for entry in self.classes],
            "references": list(self.references),
            "unanchored": list(self.unanchored),
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> GlitchPopulation:
        """Rebuild a population from :meth:`serialize` output.

        Args:
            payload: The serialized population.

        Returns:
            The reconstructed population, which serializes and digests identically.

        Raises:
            KeyError: If a required key is missing.
            ValueError: If the payload declares a schema version this package cannot read.
        """
        schema_version = payload.get("schema_version", GLITCH_POPULATION_SCHEMA_VERSION)
        if schema_version != GLITCH_POPULATION_SCHEMA_VERSION:
            raise ValueError(
                f"glitch population schema version {schema_version!r} cannot be read by this package, "
                f"which writes {GLITCH_POPULATION_SCHEMA_VERSION!r}."
            )
        return cls(
            name=payload["name"],
            description=payload["description"],
            classes=tuple(GlitchClass.deserialize(entry) for entry in payload["classes"]),
            references=tuple(payload.get("references", ())),
            unanchored=tuple(payload.get("unanchored", ())),
            schema_version=schema_version,
        )

    def canonical_json(self) -> str:
        """Return the exact bytes the digest is taken over.

        Sorted keys, no insignificant whitespace, ASCII only and no ``NaN``: a mapping that
        two processes can be relied on to render identically. Exposed rather than hidden so
        a campaign that disagrees about a digest can diff what was hashed instead of
        guessing.

        Returns:
            The canonical serialization.
        """
        return json.dumps(
            self.serialize(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    def digest(self) -> str:
        """Return the population's ``sha256:...`` digest.

        What a campaign manifest pins. Every registered quantity, every prose field and the
        schema version go into it, so a population that has been edited cannot present
        itself as the one a finished run used.

        Returns:
            The digest, algorithm-prefixed.
        """
        return "sha256:" + hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def expected_counts(
        self,
        *,
        livetime_seconds: float,
        detectors: Sequence[str],
        snr_threshold: float | None = None,
    ) -> list[ExpectedCount]:
        """Decompose the expected injected count per class and interferometer.

        Args:
            livetime_seconds: Analysed time the count is over. This is *observing* time, not
                calendar time: the distinction is the commonest factor-of-1.3 error in a
                count derived from a published rate.
            detectors: The run's interferometers. Checked against the population's own
                scoping, so a decomposition cannot be computed for a network the population
                does not account for.
            snr_threshold: Optional recording or detection cut. When given, the selection
                factor is the fraction of each class's target-SNR distribution at or above
                it, read off the distribution in closed form.

        Returns:
            One row per (class, interferometer) pair, in registration order.

        Raises:
            ValueError: If the livetime is not positive, no interferometers were given, or
                the population does not account for every interferometer.
        """
        if not np.isfinite(livetime_seconds) or livetime_seconds <= 0.0:
            raise ValueError("livetime_seconds must be finite and greater than zero.")
        run_detectors = list(dict.fromkeys(detectors))
        if not run_detectors:
            raise ValueError("expected_counts requires at least one interferometer.")
        validate_glitch_detector_coverage(self.models(), run_detectors)

        rows: list[ExpectedCount] = []
        for entry in self.classes:
            model = entry.model
            participation = 1.0 if model.network is None else model.network.participation_probability
            selection = 1.0 if snr_threshold is None else snr_survival(getattr(model, "snr", None), snr_threshold)
            for detector in run_detectors:
                if not model.applies_to(detector):
                    continue
                rows.append(
                    ExpectedCount(
                        glitch_class=entry.name,
                        detector=detector,
                        rate_hz=float(model.rate),
                        livetime_seconds=float(livetime_seconds),
                        participation=float(participation),
                        selection=float(selection),
                        expected=float(model.rate * livetime_seconds * participation * selection),
                    )
                )
        return rows

    def build_injector(self, base: NoiseSimulator, *, gps_start: float = 0.0) -> InjectGlitches:
        """Wrap a base simulator so it also injects this population.

        Args:
            base: The simulator producing the noise the glitches are added to.
            gps_start: GPS time of the first sample, stamped onto the truth catalogue.

        Returns:
            The wrapped simulator.
        """
        return InjectGlitches(base, self.models(), gps_start=gps_start)

    def realize(
        self,
        *,
        detectors: Sequence[str],
        duration: float,
        sampling_frequency: float,
        seed: int,
        gps_start: float = 0.0,
    ) -> GlitchRealization:
        """Generate this population's glitch strain on its own, with no noise under it.

        **This is the object paired arms share.** A comparison whose arms differ in their
        signal content, their geometry, their detection threshold or their ranking
        statistic must not differ in their glitches, and the way to guarantee that is for
        every arm to add *the same array* rather than to re-run a generator and trust it to
        agree. The returned stamp carries the population digest and the seed, so an arm can
        record which realization it added.

        The realization is reproducible for a fixed (package version, population, seed) and
        does not depend on the base noise, on the other interferometers in the run, or on
        whether it was generated in one call or streamed -- see the module's tests, which
        assert each of those separately.

        Args:
            detectors: The interferometers to realize.
            duration: Length of the realization in seconds.
            sampling_frequency: Sampling frequency in Hz. Note that a class with a
                ``high_frequency_cutoff`` above the Nyquist frequency is refused rather
                than silently narrowed, so a population registered over a wide band needs a
                rate that can carry it.
            seed: The realization's seed.
            gps_start: GPS time of the first sample.

        Returns:
            The glitch strain, its truth catalogue and the stamp identifying it.

        Raises:
            ValueError: If the population does not account for every interferometer.
        """
        run_detectors = list(dict.fromkeys(detectors))
        validate_glitch_detector_coverage(self.models(), run_detectors)
        injector = self.build_injector(
            _ZeroNoiseSimulator(
                detectors=run_detectors,
                duration=duration,
                sampling_frequency=sampling_frequency,
                seed=seed,
            ),
            gps_start=gps_start,
        )
        strain = injector.generate(duration, sampling_frequency, run_detectors, seed=seed)
        return GlitchRealization(
            strain=strain,
            events=injector.glitch_events,
            stamp={
                "population": self.name,
                "schema_version": self.schema_version,
                "digest": self.digest(),
                "gwmock_noise_version": __version__,
                "seed": seed,
                "detectors": run_detectors,
                "duration": float(duration),
                "sampling_frequency": float(sampling_frequency),
                "gps_start": float(gps_start),
                "event_count": len(injector.glitch_events),
            },
        )
