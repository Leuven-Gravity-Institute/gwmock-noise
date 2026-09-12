"""Runtime glitch-model definitions."""

from __future__ import annotations

import importlib
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
    """Base dataclass for transient glitch generators."""

    rate: float
    amplitude_distribution: LogNormalAmplitudeDistribution
    kind: str = field(init=False)

    def __post_init__(self) -> None:
        """Validate common glitch parameters."""
        if self.rate < 0.0:
            raise ValueError("glitch rate must be non-negative.")

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


def normalize_glitch_models(value: Any) -> list[GlitchModel]:
    """Normalize heterogeneous glitch-config inputs."""
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
