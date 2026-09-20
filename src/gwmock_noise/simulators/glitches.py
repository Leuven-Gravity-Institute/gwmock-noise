"""Transient glitch injection wrappers."""

from __future__ import annotations

import math
import zlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise.glitches.models import (
    GlitchDraw,
    GlitchModel,
    NetworkCoherence,
    validate_glitch_detector_coverage,
)
from gwmock_noise.simulators.protocol import NoiseSimulator

#: Schema version of the per-event glitch truth catalogue. Bumped when the meaning of a
#: column changes, including when the shape does not -- a consumer has nothing else to
#: read the change from.
GLITCH_CATALOGUE_SCHEMA_VERSION = "1.1.0"

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
        "Stable identifier, '<detector>-<model_index>-<ordinal>'. For a model with no network "
        "coherence the ordinal counts this (model, detector) pair's events from zero. For a "
        "network-coherent model it is the ordinal of the shared network event, so a detector "
        "that skipped an event skips its ordinal and the numbering is not contiguous per "
        "detector. Reproducible either way for a fixed (version, config, seed)."
    ),
    "network_event_id": (
        "Identifier of the shared network event, '<model_index>-<ordinal>', for a "
        "network-coherent model; null for a model whose detectors are independent. Rows "
        "sharing it are the same transient seen in different interferometers."
    ),
    "amplitude_ratio": (
        "Per-interferometer multiplier applied to a network-coherent event's shared waveform; "
        "null for a model whose detectors are independent."
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
        "Samples in the drawn waveform. A waveform still running where the generated data ends "
        "-- past the last segment, or past a segment whose successor sits at a discontinuous "
        "epoch -- is truncated there, so the strain may hold fewer than this."
    ),
    "segment_index": "Zero-based index of the generate() call that injected the event.",
    "sample_index": "Index of the waveform's first sample within that segment.",
    "target_snr": (
        "Optimal SNR the draw was calibrated to, before the amplitude multiplier, or null for "
        "a model that does not calibrate SNR."
    ),
    "realized_snr": (
        "Optimal SNR of the waveform as drawn, against the PSD it was colored with, or null "
        "for an uncolored model. The strain normally carries all of it, because a waveform "
        "crossing a segment boundary has its remainder injected into the next segment. The "
        "exception is the end of the generated data, where a still-running waveform is cut "
        "and the strain then holds less than this figure."
    ),
    "amplitude": "Amplitude multiplier drawn from the model's amplitude distribution.",
}


#: Spawn-key slot for a network-coherent model's shared arrival-and-waveform stream. It sits
#: one past the top of the CRC-32 range the per-detector streams are keyed by, so a network
#: stream can never collide with an interferometer's own stream however the name hashes.
NETWORK_STREAM_KEY = 1 << 32

#: Spawn-key tag for the per-(event, interferometer) participation stream of a
#: network-coherent model. Also outside the CRC-32 range, and it sits in a third slot, so a
#: participation stream is distinct from both of the above.
NETWORK_PARTICIPATION_KEY = (1 << 32) + 1


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

    A glitch model with no ``network`` coherence runs an independent Poisson process
    per detector, so every detector receives its own event times and waveform
    realizations, and ``GlitchModel.rate`` is the event rate seen by each individual
    detector. Every ``(model, detector)`` pair draws from its own random generator,
    derived from the top-level seed and the detector name via an independent
    ``SeedSequence`` stream.

    A model carrying a :class:`~gwmock_noise.glitches.models.NetworkCoherence` runs
    **one** Poisson process for the whole network and draws **one** waveform per
    event, offering it to every detector it applies to. ``GlitchModel.rate`` is then
    the *network* event rate, not the per-detector one, and each detector sees it
    times the coherence's participation probability. Arrival times and waveforms come
    from a stream keyed by the model alone; each detector's participation and
    amplitude ratio come from a stream keyed by its own name and the event's ordinal.

    Either way, **a detector's realization is reproducible and independent of which
    other detectors are present or the order in which they are requested** -- which is
    what lets two runs over different networks be compared on the channels they share.

    A glitch whose waveform extends past the end of a chunk has its remainder
    carried into the next chunk and replayed in event order, so the injected
    glitch series is identical, sample for sample, to a single generate call of
    the same total duration (the per-sample addition order is preserved exactly,
    not merely up to rounding). A tail is dropped where the data it would continue
    into does not exist: past the last generated chunk, and past a segment whose
    successor the caller places at a discontinuous epoch.

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
        self._contiguous_gps_start: float | None = None
        self._network_rngs: dict[int, np.random.Generator] = {}
        self._network_next_event_times: dict[int, float] = {}
        self._network_event_counts: dict[int, int] = {}
        # Every (model, interferometer) pair the run has reached, network-coherent or not.
        # `metadata` reports counts over this rather than over the per-pair Poisson clocks,
        # which a network-coherent model does not have: it has one clock for the network.
        self._seen_pairs: set[tuple[int, str]] = set()

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
        self._contiguous_gps_start = None
        self._network_rngs = {}
        self._network_next_event_times = {}
        self._network_event_counts = {}
        self._seen_pairs = set()

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

    def _network_rng_for(self, model_index: int) -> np.random.Generator:
        """Return the shared arrival-and-waveform generator for one network-coherent model.

        Keyed by the model alone. A network-coherent event belongs to the network rather
        than to any one interferometer, so its time and its waveform have to be drawn from a
        stream that does not know which interferometers the run contains -- otherwise the
        same seed would produce different transients in a three-interferometer run and in
        the two-interferometer one it is being compared against, and the comparison would be
        measuring the generator rather than the geometry.
        """
        rng = self._network_rngs.get(model_index)
        if rng is None:
            if self._seed_sequence is None:
                raise RuntimeError("glitch RNG was not initialized.")
            child = np.random.SeedSequence(
                entropy=self._seed_sequence.entropy,
                spawn_key=(model_index, NETWORK_STREAM_KEY),
            )
            rng = np.random.default_rng(child)
            self._network_rngs[model_index] = rng
        return rng

    def _network_share(
        self,
        model_index: int,
        detector: str,
        ordinal: int,
        coherence: NetworkCoherence,
    ) -> float | None:
        """Decide whether one interferometer takes a network event, and how loudly.

        The generator is derived from the seed, the model, the interferometer name and the
        network event's own ordinal, and is used once and discarded. That costs a
        ``SeedSequence`` spawn per (event, interferometer) -- microseconds, against a drawn
        and colored waveform -- and buys the property the pairing rests on: one
        interferometer's decisions depend on nothing but its own name and the event's index,
        so adding or removing another interferometer, or joining one part-way through a
        stream, leaves every other interferometer's strain bit-for-bit unchanged. A single
        long-lived per-interferometer stream advanced once per event would have the same
        property only while every event advanced it, which a mid-stream join breaks.

        Args:
            model_index: Position of the model in the configured list.
            detector: The interferometer being offered the event.
            ordinal: Index of the network event, counted from zero across the whole run.
            coherence: The model's coherence specification.

        Returns:
            The multiplier to apply to the shared waveform, or ``None`` when this
            interferometer does not take this event.

        Raises:
            RuntimeError: If the wrapper's seed sequence was never initialized.
        """
        if self._seed_sequence is None:
            raise RuntimeError("glitch RNG was not initialized.")
        child = np.random.SeedSequence(
            entropy=self._seed_sequence.entropy,
            spawn_key=(
                model_index,
                zlib.crc32(detector.encode("utf-8")),
                NETWORK_PARTICIPATION_KEY,
                ordinal,
            ),
        )
        rng = np.random.default_rng(child)
        # `random()` is uniform on [0, 1), so a probability of 1.0 always participates and a
        # probability of 0.0 never does -- both without a special case.
        if rng.random() >= coherence.participation_probability:
            return None
        return coherence.amplitude_ratio(rng)

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
        network_event_id: str | None = None,
        amplitude_ratio: float | None = None,
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

        ``realized_snr`` is likewise the drawn waveform's, and deliberately not the
        SNR of the samples this segment happens to receive. Only part of a
        boundary-crossing waveform lands here, but the remainder is injected into the
        next segment rather than lost, so an SNR computed over this segment's prefix
        would understate a glitch the data holds in full -- and would differ between a
        streamed run and a single call over the same span, which nothing else about the
        catalogue does. A waveform is genuinely cut only where the generated data ends,
        which is the caveat the column's own description carries.

        ``amplitude_ratio`` scales it. A network-coherent event is drawn once and offered
        to several interferometers, so the draw's own figure describes the shared waveform
        and not what any one of them received; SNR is linear in amplitude, so the row
        reports the drawn figure times this interferometer's ratio. The ratio is kept in its
        own column rather than folded into ``amplitude``, which is the model's own
        amplitude-distribution draw and is shared by every interferometer taking the event.
        """
        waveform = draw.waveform
        gps_start_time = segment_gps_start + (sample_index / sampling_frequency)
        peak_offset = int(np.argmax(np.abs(waveform))) / sampling_frequency
        record = {
            "event_id": f"{detector}-{model_index}-{ordinal}",
            "network_event_id": network_event_id,
            "amplitude_ratio": None if amplitude_ratio is None else float(amplitude_ratio),
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
            "realized_snr": (
                None
                if draw.realized_snr is None
                else float(draw.realized_snr if amplitude_ratio is None else draw.realized_snr * amplitude_ratio)
            ),
            "amplitude": None if draw.amplitude is None else float(draw.amplitude),
        }
        # Appended to the segment's list only; `generate` sorts that list and extends the
        # cumulative catalogue with it, so both are in time order rather than in the
        # model-then-detector order the injection loop happens to run in.
        self._segment_events.append(record)

    def _drop_tails_across_a_gap(self, segment_gps_start: float, sampling_frequency: float) -> None:
        """Discard carried-over tails when this segment does not adjoin the previous one.

        A tail is the remainder of a waveform, replayed at sample zero of the next segment
        because that is where the waveform continues. It only continues there if the next
        segment *is* the next data: ``FrameWriter.write_segments`` permits gaps, so a caller
        can place this segment a thousand seconds after the last, and the interval between
        them holds no data for a waveform to run through.

        Replaying the tail anyway put strain into the new segment that no row of its
        catalogue accounted for, while the row that produced those samples carried a
        timestamp from before the gap -- so the strain and the truth disagreed about when
        the glitch was. Measured: a 112-sample tail from a segment at GPS 1256655618 was
        written at the start of a segment at GPS 1256656618.

        Dropping it is the same rule that already applies past the last generated chunk, and
        for the same reason. Contiguity is judged against the epoch this wrapper advanced to
        after the previous segment, so a caller that leaves ``gps_start`` alone keeps its
        tails.

        **Judged on the sample grid, not by exact equality.** A writer computes each epoch
        as ``start + index * duration`` while this wrapper advances by adding ``duration``
        once per segment; the two take different rounding paths, so for a duration that
        binary floating point cannot hold exactly they differ by an ULP. At GPS magnitudes
        that is ~2.4e-7 s, four orders of magnitude below a sample period at any realistic
        rate -- unmistakably the same instant -- and an exact comparison read it as a gap
        and discarded valid tails. Measured with ``duration=0.1``: 38 samples of waveform
        lost from an eight-segment run, on segments 2 and 4 where the rounding diverged.

        Half a sample period is the meaningful threshold: below it the two epochs name the
        same sample, and at or beyond it the data is offset by a whole sample or more, which
        is a real discontinuity. The ULP floor keeps that comparison honest above about
        2 MHz, where half a sample falls below what a GPS-magnitude double can represent at
        all -- rather than resting on the assumption that nobody samples that fast.
        """
        if self._contiguous_gps_start is None:
            return
        tolerance = max(
            0.5 / sampling_frequency,
            4.0 * math.ulp(max(abs(segment_gps_start), abs(self._contiguous_gps_start))),
        )
        if abs(segment_gps_start - self._contiguous_gps_start) < tolerance:
            return
        self._pending_tails = {}

    @staticmethod
    def _event_order(record: dict[str, Any]) -> tuple[float, str]:
        """Return the sort key that puts catalogue rows in time order.

        The id breaks a tie, so two glitches landing on the same sample in different
        detectors order deterministically rather than by whichever the injection loop
        reached first.
        """
        return (record["gps_start_time"], record["event_id"])

    def _commit_segment_events(self) -> None:
        """Close out the segment's rows and fold them into the run's catalogue.

        Sorted here, once, rather than on every read: a consumer of a truth catalogue
        expects time order, and this is the only place that can establish it without a
        second pass over the run.

        The cumulative list is extended rather than merged, because segments normally
        advance in time and it is then already ordered by construction. A caller may
        assign a *decreasing* epoch, though -- ``FrameWriter.write_segments`` accepts
        segments in any order -- and ``glitch_events`` promises time order, so that case
        is re-sorted. The check is O(1), so an ordinary run does not pay O(n log n) per
        segment for a case it never hits.
        """
        self._segment_events.sort(key=self._event_order)
        out_of_order = bool(
            self._events
            and self._segment_events
            and self._segment_events[0]["gps_start_time"] < self._events[-1]["gps_start_time"]
        )
        self._events.extend(self._segment_events)
        if out_of_order:
            self._events.sort(key=self._event_order)

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

    @staticmethod
    def _copy_checked_base_strain(
        base_result: dict[str, np.ndarray],
        detectors: list[str],
        n_samples: int,
    ) -> dict[str, np.ndarray]:
        """Return a writable copy of the base strain, checked against what was asked for.

        Copied rather than injected into in place, so a base simulator handing back a
        cached or shared array does not accumulate this run's glitches.

        Args:
            base_result: What the base simulator returned.
            detectors: The interferometers this segment asked for.
            n_samples: The sample count the requested duration and rate imply.

        Returns:
            One writable array per requested detector.

        Raises:
            KeyError: If the base simulator omitted a requested detector.
            ValueError: If its output is not the requested length.
        """
        combined: dict[str, np.ndarray] = {}
        for detector in detectors:
            if detector not in base_result:
                raise KeyError(f"Base simulator did not return detector '{detector}'.")
            base_strain = np.asarray(base_result[detector], dtype=float)
            if base_strain.shape != (n_samples,):
                raise ValueError(
                    "Base simulator output shape must match the requested duration and sampling_frequency."
                )
            combined[detector] = base_strain.copy()
        return combined

    def _lay_pending_tails(self, key: tuple[int, str], strain: np.ndarray, n_samples: int) -> None:
        """Replay the waveform fragments carried over from the previous chunk.

        Laid down before this chunk's own events, so a glitch straddling a chunk boundary is
        continuous rather than truncated. Fragments are replayed in their original event
        order; anything still overflowing this chunk is re-queued, again in order, for the
        next one.
        """
        for tail in self._pending_tails.pop(key, []):
            stop = min(n_samples, tail.size)
            strain[:stop] += tail[:stop]
            if tail.size > n_samples:
                self._add_pending_tail(key, tail[n_samples:])

    def _inject_independent(  # noqa: PLR0913
        self,
        *,
        model: GlitchModel,
        model_index: int,
        detectors: list[str],
        combined: dict[str, np.ndarray],
        n_samples: int,
        segment_start: float,
        segment_end: float,
        segment_gps_start: float,
        sampling_frequency: float,
    ) -> None:
        """Inject one model whose interferometers glitch independently of one another.

        Each interferometer runs its own Poisson process and draws from its own stream, so
        ``model.rate`` is the rate each of them sees and no two of them carry the same
        transient.
        """
        for detector in detectors:
            key = (model_index, detector)
            rng = self._rng_for(model_index, detector)
            self._lay_pending_tails(key, combined[detector], n_samples)

            if key not in self._next_event_times:
                # A pair first seen mid-stream (e.g. a detector added between
                # chunks) starts its Poisson clock at the current segment start.
                self._next_event_times[key] = segment_start + self._draw_interarrival(model.rate, rng)
            event_time = self._next_event_times[key]
            while event_time < segment_end:
                sample_index = int((event_time - segment_start) * sampling_frequency)
                draw = model.draw(sampling_frequency, rng=rng)
                ordinal = self._event_counts.get(key, 0)
                if self._place_waveform(
                    strain=combined[detector],
                    waveform=draw.waveform,
                    sample_index=sample_index,
                    n_samples=n_samples,
                    key=key,
                ):
                    self._event_counts[key] = ordinal + 1
                    self._record_event(
                        model=model,
                        model_index=model_index,
                        detector=detector,
                        ordinal=ordinal,
                        draw=draw,
                        sample_index=sample_index,
                        segment_gps_start=segment_gps_start,
                        sampling_frequency=sampling_frequency,
                    )
                event_time += self._draw_interarrival(model.rate, rng)
            self._next_event_times[key] = event_time

    def _inject_network_coherent(  # noqa: PLR0913
        self,
        *,
        model: GlitchModel,
        model_index: int,
        detectors: list[str],
        combined: dict[str, np.ndarray],
        n_samples: int,
        segment_start: float,
        segment_end: float,
        segment_gps_start: float,
        sampling_frequency: float,
    ) -> None:
        """Inject one model whose events are shared across the interferometers it reaches.

        One Poisson process for the whole network -- ``model.rate`` is the *network* event
        rate, not the per-interferometer one -- and one waveform per event, offered to each
        scoped interferometer at the same sample index. Both the arrival times and the
        waveforms come from a stream keyed by the model alone, and each interferometer's
        participation from a stream keyed by its own name and the event's ordinal, so what
        an interferometer receives does not depend on which others are in the run.

        The event's ordinal counts *network* events, including ones no interferometer took,
        for the same reason: an ordinal that counted only the events some interferometer
        received would depend on the run's interferometer list, and the participation stream
        is keyed by it.
        """
        for detector in detectors:
            self._lay_pending_tails((model_index, detector), combined[detector], n_samples)

        coherence = model.network
        if coherence is None:  # pragma: no cover - the caller selects this branch
            raise RuntimeError("_inject_network_coherent requires a model with network coherence.")
        rng = self._network_rng_for(model_index)
        if model_index not in self._network_next_event_times:
            self._network_next_event_times[model_index] = segment_start + self._draw_interarrival(model.rate, rng)
        event_time = self._network_next_event_times[model_index]
        while event_time < segment_end:
            sample_index = int((event_time - segment_start) * sampling_frequency)
            draw = model.draw(sampling_frequency, rng=rng)
            ordinal = self._network_event_counts.get(model_index, 0)
            self._network_event_counts[model_index] = ordinal + 1
            network_event_id = f"{model_index}-{ordinal}"
            for detector in detectors:
                ratio = self._network_share(model_index, detector, ordinal, coherence)
                if ratio is None:
                    continue
                key = (model_index, detector)
                waveform = draw.waveform if ratio == 1.0 else ratio * draw.waveform
                if self._place_waveform(
                    strain=combined[detector],
                    waveform=waveform,
                    sample_index=sample_index,
                    n_samples=n_samples,
                    key=key,
                ):
                    self._event_counts[key] = self._event_counts.get(key, 0) + 1
                    self._record_event(
                        model=model,
                        model_index=model_index,
                        detector=detector,
                        ordinal=ordinal,
                        draw=draw,
                        sample_index=sample_index,
                        segment_gps_start=segment_gps_start,
                        sampling_frequency=sampling_frequency,
                        network_event_id=network_event_id,
                        amplitude_ratio=ratio,
                    )
            event_time += self._draw_interarrival(model.rate, rng)
        self._network_next_event_times[model_index] = event_time

    def _refuse_a_band_the_sampling_rate_cannot_carry(self, sampling_frequency: float) -> None:
        """Refuse a run whose rate cannot represent a model's configured frequency support.

        Coloring already raises on a band above the Nyquist frequency, but it raises on the
        *first drawn waveform*, and a class configured at a realistic rate draws its first
        waveform hours into a campaign -- so a run that could never have been valid
        generates and writes data until it happens to fire. Asked and answered here instead,
        before the base simulator runs, for the same reason the coverage check is: a
        configuration that cannot be honoured should not start.

        Narrowing the band to the Nyquist frequency instead would be worse than either: the
        run would then inject a class that is not the configured one while reporting the
        configuration it was asked for.

        Args:
            sampling_frequency: The rate this segment is being generated at.

        Raises:
            ValueError: If any model's ``high_frequency_cutoff`` exceeds the Nyquist
                frequency. The message names the models and the rate that would carry them.
        """
        nyquist = sampling_frequency / 2.0
        offenders = [
            (index, float(cutoff))
            for index, model in enumerate(self.glitch_models)
            if (cutoff := getattr(model, "high_frequency_cutoff", None)) is not None and cutoff > nyquist
        ]
        if not offenders:
            return
        details = "; ".join(f"model {index} is supported to {cutoff:g} Hz" for index, cutoff in offenders)
        required = 2.0 * max(cutoff for _, cutoff in offenders)
        raise ValueError(
            f"glitch models are configured above the Nyquist frequency of {nyquist:g} Hz: {details}. "
            f"Generate at {required:g} Hz or above, or lower the models' high_frequency_cutoff -- "
            f"narrowing it here would inject a different class than the one configured."
        )

    def _place_waveform(
        self,
        *,
        strain: np.ndarray,
        waveform: np.ndarray,
        sample_index: int,
        n_samples: int,
        key: tuple[int, str],
    ) -> bool:
        """Add one waveform to a chunk and queue whatever overflows it.

        Args:
            strain: The interferometer's strain for this chunk, added to in place.
            waveform: The waveform to inject.
            sample_index: Where in the chunk its first sample lands.
            n_samples: The chunk's length.
            key: The ``(model index, interferometer)`` pair whose tail queue the overflow
                belongs to.

        Returns:
            Whether any of the waveform reached the strain. ``False`` means the chunk had no
            room for it at all, and it is therefore not recorded either.
        """
        stop_index = min(n_samples, sample_index + waveform.size)
        if stop_index <= sample_index:
            return False
        strain[sample_index:stop_index] += waveform[: stop_index - sample_index]
        # Carry the part that overflows this chunk into the next one.
        self._add_pending_tail(key, waveform[stop_index - sample_index :])
        return True

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate base noise and inject the configured glitches."""
        runtime_detectors = list(detectors)
        # Before the base runs, so a configuration that does not say what every
        # interferometer gets is refused rather than half-generated. The detectors
        # asked for here are the run's, which a caller may narrow between segments,
        # so the question is settled against them rather than against whatever set
        # the models were built beside.
        validate_glitch_detector_coverage(self.glitch_models, runtime_detectors)
        self._refuse_a_band_the_sampling_rate_cannot_carry(sampling_frequency)
        base_result = self.base.generate(duration, sampling_frequency, runtime_detectors, seed=seed)

        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = runtime_detectors
        if seed is not None or self._seed_sequence is None:
            self._initialize_process(seed if seed is not None else self.seed)

        n_samples = round(duration * sampling_frequency)
        combined = self._copy_checked_base_strain(base_result, runtime_detectors, n_samples)

        segment_start = self._elapsed_time
        segment_end = segment_start + duration
        # Read before anything is injected and used for every row this segment writes, so
        # a caller reassigning `gps_start` mid-segment cannot land two events in one chunk
        # on two different epochs.
        segment_gps_start = float(self.gps_start)
        if self._seed_sequence is None:
            raise RuntimeError("glitch RNG was not initialized.")
        self._drop_tails_across_a_gap(segment_gps_start, sampling_frequency)
        self._segment_events = []

        for index, model in enumerate(self.glitch_models):
            # Only the interferometers this model is scoped to. A model the run does
            # not reach here draws nothing and consumes no stream, so the detectors it
            # does reach realize exactly what they would have without the selector.
            scoped = [detector for detector in runtime_detectors if model.applies_to(detector)]
            self._seen_pairs.update((index, detector) for detector in scoped)
            if model.network is None:
                self._inject_independent(
                    model=model,
                    model_index=index,
                    detectors=scoped,
                    combined=combined,
                    n_samples=n_samples,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    segment_gps_start=segment_gps_start,
                    sampling_frequency=sampling_frequency,
                )
            else:
                self._inject_network_coherent(
                    model=model,
                    model_index=index,
                    detectors=scoped,
                    combined=combined,
                    n_samples=n_samples,
                    segment_start=segment_start,
                    segment_end=segment_end,
                    segment_gps_start=segment_gps_start,
                    sampling_frequency=sampling_frequency,
                )

        self._elapsed_time = segment_end
        self._segment_index += 1
        self.gps_start = segment_gps_start + duration
        # Remembered apart from `gps_start`, which the caller may overwrite: this is the
        # epoch that would continue this segment without a gap, and comparing the next
        # segment's epoch against it is what tells a contiguous stream from a jump.
        self._contiguous_gps_start = self.gps_start
        # In injection order across models and detectors, then by time within a pair. A
        # consumer reading a catalogue expects it in time order, and sorting here is the
        # one place it can be done without a second pass over the run.
        self._commit_segment_events()
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
                for model_index, detector in sorted(self._seen_pairs)
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

    A wrapper that *builds* its simulators rather than holding one -- ``ParallelAdapter``,
    which constructs and caches one per detector from a factory -- cannot be reached by
    walking attributes, and an epoch set on such a wrapper alone reached none of its
    workers. Those declare a writable ``segment_gps_start`` and own the fan-out, so this
    hands the epoch over and lets them place both the workers they already hold and any
    they build later in the same segment.
    """
    if isinstance(simulator, InjectGlitches):
        simulator.gps_start = float(gps_start)
    if hasattr(type(simulator), "segment_gps_start"):
        simulator.segment_gps_start = float(gps_start)
    base = getattr(simulator, "base", None)
    if base is not None and base is not simulator:
        apply_segment_gps_start(base, gps_start)
    for component in getattr(simulator, "_components", None) or ():
        # A composite stores (name, simulator) pairs.
        apply_segment_gps_start(component[1], gps_start)
