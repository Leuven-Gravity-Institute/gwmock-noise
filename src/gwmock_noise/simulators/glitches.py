"""Transient glitch injection wrappers."""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.glitches.models import GlitchDraw, GlitchModel
from gwmock_noise.simulators.protocol import NoiseSimulator

#: Schema version of the per-event glitch truth catalogue. Bumped when the meaning of a
#: column changes, including when the shape does not -- a consumer has nothing else to
#: read the change from.
GLITCH_CATALOGUE_SCHEMA_VERSION = "1.0.0"

#: Which instant each time column refers to, stated rather than left to be inferred. The
#: Poisson process draws the time the waveform *starts*, so for a 2 s DeepExtractor sample
#: the visible transient sits about a second after ``gps_start_time``; ``gps_peak_time`` is
#: where it actually peaks. A consumer cutting a window around the wrong one of the two
#: misses the glitch, and neither column's name alone says which it is.
GLITCH_CATALOGUE_TIME_CONVENTION = (
    "gps_start_time is the GPS time of the FIRST SAMPLE of the injected waveform -- the "
    "Poisson event time, snapped down to the sample grid the strain is written on. It is "
    "not the peak. gps_peak_time is the GPS time of the largest |strain| sample of the same "
    "waveform, measured on the waveform as drawn (before any truncation at the end of the "
    "data)."
)

#: One line per catalogue column, carried in the metadata beside the rows so a file
#: describes its own columns.
GLITCH_CATALOGUE_COLUMNS: dict[str, str] = {
    "event_id": (
        "Stable identifier, '<detector>-<model_index>-<ordinal>', where ordinal counts this "
        "(model, detector) pair's events from zero. Reproducible for a fixed "
        "(version, config, seed)."
    ),
    "detector": "Interferometer the glitch was injected into.",
    "model_index": "Position of the glitch model in the configured model list.",
    "kind": "Glitch-model kind, e.g. 'blip', 'scattered_light', 'deepextractor'.",
    "glitch_class": (
        "Drawn morphological class (e.g. a Gravity Spy class name), or null for a model with a single morphology."
    ),
    "gps_start_time": "GPS time of the first sample of the injected waveform. Not the peak.",
    "gps_peak_time": "GPS time of the largest |strain| sample of the injected waveform.",
    "duration_seconds": "Length of the drawn waveform, n_samples / sampling_frequency.",
    "n_samples": (
        "Samples in the drawn waveform. A waveform still running when the generated data ends "
        "is truncated there, so the strain may hold fewer than this."
    ),
    "segment_index": "Zero-based index of the generate() call that injected the event.",
    "sample_index": "Index of the waveform's first sample within that segment.",
    "target_snr": (
        "Optimal SNR the draw was calibrated to, before the amplitude multiplier, or null for "
        "a model that does not calibrate SNR."
    ),
    "realized_snr": (
        "Optimal SNR the injected waveform actually carries against the PSD it was colored "
        "with, or null for an uncolored model."
    ),
    "amplitude": "Amplitude multiplier drawn from the model's amplitude distribution.",
}


class _ZeroNoiseSimulator:
    """Minimal zero-noise base simulator for glitch-only realizations."""

    def __init__(
        self,
        *,
        detectors: list[str],
        duration: float,
        sampling_frequency: float,
        seed: int | None,
    ) -> None:
        """Initialize the zero-valued protocol-compatible state."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed

    def reset(self) -> None:
        """Reset the zero-noise base state."""
        return

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate zero-valued strain arrays."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        n_samples = round(duration * sampling_frequency)
        return {detector: np.zeros(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield zero-valued chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return metadata for the zero-valued base simulator."""
        return {
            "implementation": "zero",
            "duration": self.duration,
            "sampling_frequency": self.sampling_frequency,
            "detectors": list(self.detectors),
            "seed": self.seed,
        }


class InjectGlitches:
    """Wrap a base simulator and inject transient glitches additively.

    Each glitch model runs an independent Poisson process per detector, so
    every detector receives its own event times and waveform realizations.
    ``GlitchModel.rate`` is therefore the event rate seen by each individual
    detector.

    Every ``(model, detector)`` pair draws from its own random generator,
    derived from the top-level seed and the detector name via an independent
    ``SeedSequence`` stream. A detector's realization is therefore reproducible
    and independent of which other detectors are present or the order in which
    they are requested.

    A glitch whose waveform extends past the end of a chunk has its remainder
    carried into the next chunk and replayed in event order, so the injected
    glitch series is identical, sample for sample, to a single generate call of
    the same total duration (the per-sample addition order is preserved exactly,
    not merely up to rounding). A tail is dropped only when it overflows the last
    generated chunk, where the data window genuinely ends.

    Every event that lands in the strain is recorded as it fires, not merely
    tallied: see :attr:`glitch_events` for the cumulative truth catalogue and
    :attr:`segment_glitch_events` for the events the most recent generate call
    injected. A row is written where a glitch *starts*, so a waveform straddling
    a chunk boundary appears once, in the chunk that holds its first sample, at
    its true time -- the carried-over tail adds samples to the next chunk but no
    second row. ``GLITCH_CATALOGUE_COLUMNS`` documents the columns and
    ``GLITCH_CATALOGUE_TIME_CONVENTION`` which instant each time column names.

    ``gps_start`` is the GPS time of the first sample of the next segment, and it
    belongs to the caller. It advances by each generated segment's duration, so a
    stream's catalogue reads in real GPS time on its own; a caller that knows
    better -- a writer placing non-contiguous segments, or one driving the stream
    batch by batch -- assigns it before each generate call and that assignment is
    honoured, including on a call that also carries a seed. Only ``reset``, the
    explicit start-over, rewinds it to the epoch the wrapper was constructed with.
    """

    def __init__(
        self,
        base: NoiseSimulator,
        glitch_models: list[GlitchModel],
        gps_start: float = 0.0,
    ) -> None:
        """Initialize the additive glitch wrapper."""
        if not glitch_models:
            raise ValueError("glitch_models must contain at least one glitch model.")

        self.base = base
        self.glitch_models = list(glitch_models)
        self.duration = base.duration
        self.sampling_frequency = base.sampling_frequency
        self.detectors = list(base.detectors)
        self.seed = base.seed
        self.gps_start = float(gps_start)

        self._epoch = float(gps_start)
        self._elapsed_time = 0.0
        self._segment_index = 0
        self._seed_sequence: np.random.SeedSequence | None = None
        self._rngs: dict[tuple[int, str], np.random.Generator] = {}
        self._next_event_times: dict[tuple[int, str], float] = {}
        self._event_counts: dict[tuple[int, str], int] = {}
        self._pending_tails: dict[tuple[int, str], list[np.ndarray]] = {}
        self._events: list[dict[str, Any]] = []
        self._segment_events: list[dict[str, Any]] = []

    def _initialize_process(self, seed: int | None) -> None:
        """Reset the per-model, per-detector Poisson-process state.

        Deliberately does **not** touch ``gps_start``. This runs from inside
        ``generate`` whenever a seed is passed, which is after a segment writer has
        already said where the segment sits in GPS time -- so rewinding the epoch here
        overwrote that statement. It timed the first seeded segment against the
        constructor's epoch and then auto-advanced from there, so on the streaming path
        (a seed on the first chunk and none after) *every* chunk of a run was timed from
        the constructor's epoch rather than the writer's. Measured: a stream told to
        start at GPS 1256655618 recorded its first glitch at 0.03 s.
        """
        self.seed = seed
        self._elapsed_time = 0.0
        self._segment_index = 0
        self._seed_sequence = np.random.SeedSequence(seed)
        self._rngs = {}
        self._next_event_times = {}
        self._event_counts = {}
        self._pending_tails = {}
        self._events = []
        self._segment_events = []

    def _add_pending_tail(self, key: tuple[int, str], tail: np.ndarray) -> None:
        """Queue a waveform tail to inject at the start of the next chunk.

        Tails are aligned to sample 0 of the next chunk and kept as an ordered
        list, appended in event order. They are injected sequentially rather
        than pre-summed so the per-sample addition order matches a single
        generate call exactly (floating-point addition is not associative, so
        summing overlapping tails first would drift by ~1 ULP against the base).
        """
        if tail.size == 0:
            return
        self._pending_tails.setdefault(key, []).append(tail.astype(float, copy=True))

    def _rng_for(self, model_index: int, detector: str) -> np.random.Generator:
        """Return the dedicated generator for one (model, detector) pair.

        The stream is spawned from the top-level seed and a stable hash of the
        detector name, so it does not depend on the presence or ordering of any
        other detector.
        """
        key = (model_index, detector)
        rng = self._rngs.get(key)
        if rng is None:
            if self._seed_sequence is None:
                raise RuntimeError("glitch RNG was not initialized.")
            detector_key = zlib.crc32(detector.encode("utf-8"))
            child = np.random.SeedSequence(
                entropy=self._seed_sequence.entropy,
                spawn_key=(model_index, detector_key),
            )
            rng = np.random.default_rng(child)
            self._rngs[key] = rng
        return rng

    @staticmethod
    def _draw_interarrival(rate: float, rng: np.random.Generator) -> float:
        """Draw the next waiting time for one glitch process."""
        if rate == 0.0:
            return float(np.inf)
        return float(rng.exponential(1.0 / rate))

    def _record_event(  # noqa: PLR0913
        self,
        *,
        model: GlitchModel,
        model_index: int,
        detector: str,
        ordinal: int,
        draw: GlitchDraw,
        sample_index: int,
        segment_gps_start: float,
        sampling_frequency: float,
    ) -> None:
        """Append one catalogue row for a glitch that reached the strain.

        Called where the waveform is added, so the catalogue cannot claim an event
        the strain does not carry: a draw the chunk has no room for is never
        recorded, and a draw carried into the next chunk is recorded once, here,
        against the chunk it starts in.

        The peak is measured on the drawn waveform rather than inferred from the
        model's shape. A blip peaks at its centre, a scattered-light arch at its
        own centre, and a DeepExtractor sample wherever that reconstruction
        happens to peak -- one rule per model would be three chances to get a
        column wrong, and the drawn samples answer it directly for all three.
        """
        waveform = draw.waveform
        gps_start_time = segment_gps_start + (sample_index / sampling_frequency)
        peak_offset = int(np.argmax(np.abs(waveform))) / sampling_frequency
        record = {
            "event_id": f"{detector}-{model_index}-{ordinal}",
            "detector": detector,
            "model_index": model_index,
            "kind": model.kind,
            "glitch_class": draw.glitch_class,
            "gps_start_time": float(gps_start_time),
            "gps_peak_time": float(gps_start_time + peak_offset),
            "duration_seconds": float(waveform.size / sampling_frequency),
            "n_samples": int(waveform.size),
            "segment_index": int(self._segment_index),
            "sample_index": int(sample_index),
            "target_snr": None if draw.target_snr is None else float(draw.target_snr),
            "realized_snr": None if draw.realized_snr is None else float(draw.realized_snr),
            "amplitude": None if draw.amplitude is None else float(draw.amplitude),
        }
        # Appended to the segment's list only; `generate` sorts that list and extends the
        # cumulative catalogue with it, so both are in time order rather than in the
        # model-then-detector order the injection loop happens to run in.
        self._segment_events.append(record)

    @property
    def glitch_events(self) -> list[dict[str, Any]]:
        """Return every glitch injected since the last (re)seed, in time order.

        Copies, so a consumer cannot edit the run's own record of itself.
        """
        return [dict(record) for record in self._events]

    @property
    def segment_glitch_events(self) -> list[dict[str, Any]]:
        """Return the glitches the most recent generate call injected.

        What a per-batch consumer wants: the rows belonging to the chunk it is
        about to write, rather than the whole stream's catalogue re-filtered by
        time. Empty before the first generate call, and after a call that fired
        no events.
        """
        return [dict(record) for record in self._segment_events]

    def reset(self) -> None:
        """Reset the additive wrapper and any resettable base state.

        Rewinds the epoch to the one the wrapper was constructed with, which
        ``_initialize_process`` does not: this is the caller saying "start over", where a
        seed passed to ``generate`` is the caller saying "this segment, whose epoch I have
        just told you, begins a new realization". The two need different answers, and
        conflating them is what put every streamed segment on the wrong epoch.
        """
        self.gps_start = self._epoch
        self._initialize_process(self.seed)
        if hasattr(self.base, "reset"):
            self.base.reset()

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate base noise and inject the configured glitches."""
        runtime_detectors = list(detectors)
        base_result = self.base.generate(duration, sampling_frequency, runtime_detectors, seed=seed)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        if seed is not None or self._seed_sequence is None:
            self._initialize_process(seed if seed is not None else self.seed)

        n_samples = round(duration * sampling_frequency)
        combined: dict[str, np.ndarray] = {}
        for detector in runtime_detectors:
            if detector not in base_result:
                raise KeyError(f"Base simulator did not return detector '{detector}'.")
            base_strain = np.asarray(base_result[detector], dtype=float)
            if base_strain.shape != (n_samples,):
                raise ValueError(
                    "Base simulator output shape must match the requested duration and sampling_frequency."
                )
            combined[detector] = base_strain.copy()

        segment_start = self._elapsed_time
        segment_end = segment_start + duration
        # Read before anything is injected and used for every row this segment writes, so
        # a caller reassigning `gps_start` mid-segment cannot land two events in one chunk
        # on two different epochs.
        segment_gps_start = float(self.gps_start)
        if self._seed_sequence is None:
            raise RuntimeError("glitch RNG was not initialized.")
        self._segment_events = []

        for index, model in enumerate(self.glitch_models):
            for detector in runtime_detectors:
                key = (index, detector)
                rng = self._rng_for(index, detector)

                # Lay down any waveform tails carried over from the previous chunk
                # before processing this chunk's events, so a glitch straddling a
                # chunk boundary is continuous rather than truncated. Fragments are
                # replayed in their original event order; anything still overflowing
                # this chunk is re-queued (again in order) for the next one.
                for tail in self._pending_tails.pop(key, []):
                    stop = min(n_samples, tail.size)
                    combined[detector][:stop] += tail[:stop]
                    if tail.size > n_samples:
                        self._add_pending_tail(key, tail[n_samples:])

                if key not in self._next_event_times:
                    # A pair first seen mid-stream (e.g. a detector added between
                    # chunks) starts its Poisson clock at the current segment start.
                    self._next_event_times[key] = segment_start + self._draw_interarrival(model.rate, rng)
                event_time = self._next_event_times[key]
                while event_time < segment_end:
                    sample_index = int((event_time - segment_start) * sampling_frequency)
                    draw = model.draw(sampling_frequency, rng=rng)
                    waveform = draw.waveform
                    stop_index = min(n_samples, sample_index + waveform.size)
                    if stop_index > sample_index:
                        combined[detector][sample_index:stop_index] += waveform[: stop_index - sample_index]
                        ordinal = self._event_counts.get(key, 0)
                        self._event_counts[key] = ordinal + 1
                        self._record_event(
                            model=model,
                            model_index=index,
                            detector=detector,
                            ordinal=ordinal,
                            draw=draw,
                            sample_index=sample_index,
                            segment_gps_start=segment_gps_start,
                            sampling_frequency=sampling_frequency,
                        )
                        # Carry the part that overflows this chunk into the next one.
                        self._add_pending_tail(key, waveform[stop_index - sample_index :])
                    event_time += self._draw_interarrival(model.rate, rng)
                self._next_event_times[key] = event_time

        self._elapsed_time = segment_end
        self._segment_index += 1
        self.gps_start = segment_gps_start + duration
        # In injection order across models and detectors, then by time within a pair. A
        # consumer reading a catalogue expects it in time order, and sorting here is the
        # one place it can be done without a second pass over the run.
        self._segment_events.sort(key=lambda record: (record["gps_start_time"], record["event_id"]))
        self._events.extend(self._segment_events)
        return combined

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Yield glitch-injected chunks lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return additive-wrapper metadata."""
        base_metadata = dict(self.base.metadata)
        counts = []
        for index, model in enumerate(self.glitch_models):
            count_by_detector = {
                detector: int(self._event_counts.get((model_index, detector), 0))
                for model_index, detector in sorted(self._next_event_times)
                if model_index == index
            }
            counts.append(
                {
                    "kind": model.kind,
                    "count": sum(count_by_detector.values()),
                    "count_by_detector": count_by_detector,
                }
            )
        return base_metadata | {
            "implementation": "inject_glitches",
            "base_implementation": base_metadata.get("implementation"),
            "glitches": {
                "models": [model.serialize() for model in self.glitch_models],
                "elapsed_time_seconds": self._elapsed_time,
                "counts": counts,
                # The per-event truth catalogue, beside the counts rather than replacing
                # them: the counts are a run-level summary and these are the events, and a
                # consumer scoring injections needs the rows.
                "catalogue": {
                    "schema_version": GLITCH_CATALOGUE_SCHEMA_VERSION,
                    "time_convention": GLITCH_CATALOGUE_TIME_CONVENTION,
                    "columns": dict(GLITCH_CATALOGUE_COLUMNS),
                    "events": self.glitch_events,
                },
            },
        }


def apply_segment_gps_start(simulator: Any, gps_start: float) -> None:
    """Point every glitch injector inside a simulator chain at one segment epoch.

    A writer that assigns a per-segment epoch -- ``FrameWriter``, whose frame names
    carry it -- is the authority on where a segment sits in GPS time, and the glitch
    truth catalogue has to be stamped with that same instant or the catalogue and the
    frame name describe different data. Threading the writer's epoch through is what
    keeps it one number rather than two that agree only while the segments happen to be
    contiguous.

    Walks the wrapper chain (``base``) and a composite's components, because the
    injector is rarely the outermost simulator: a glitch component sits inside
    ``CompositeNoiseSimulator``, which sits inside whatever adapter the writer holds.
    A simulator with no injector anywhere beneath it is left alone.
    """
    if isinstance(simulator, InjectGlitches):
        simulator.gps_start = float(gps_start)
    base = getattr(simulator, "base", None)
    if base is not None and base is not simulator:
        apply_segment_gps_start(base, gps_start)
    for component in getattr(simulator, "_components", None) or ():
        # A composite stores (name, simulator) pairs.
        apply_segment_gps_start(component[1], gps_start)
