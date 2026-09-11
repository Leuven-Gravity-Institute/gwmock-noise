"""Tests for the per-event glitch truth catalogue.

The catalogue is what makes a glitch frame set scorable: without it a run can be
produced but not measured against, because nothing records which glitch went in
where. These tests hold it to the strain rather than to itself -- a row must
correspond to samples actually present in the output, at the time it claims.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.config import NoiseComponentConfig, NoiseConfig, OutputConfig
from gwmock_noise.glitches import BlipGlitch, LogNormalAmplitudeDistribution, ScatteredLightGlitch
from gwmock_noise.parallel import ParallelAdapter
from gwmock_noise.simulators import DefaultNoiseSimulator, InjectGlitches, apply_segment_gps_start
from gwmock_noise.simulators.glitches import (
    GLITCH_CATALOGUE_COLUMNS,
    GLITCH_CATALOGUE_SCHEMA_VERSION,
    _ZeroNoiseSimulator,
)

O3_EPOCH = 1256655618.0


def _zero_base(detectors: list[str], sampling_frequency: float = 256.0) -> _ZeroNoiseSimulator:
    return _ZeroNoiseSimulator(
        detectors=detectors,
        duration=4.0,
        sampling_frequency=sampling_frequency,
        seed=None,
    )


def _write_flat_psd(path: Path, *, value: float = 1.0) -> Path:
    frequencies = np.linspace(0.0, 4096.0, 129)
    np.savetxt(path, np.column_stack((frequencies, np.full_like(frequencies, value))))
    return path


def _nonzero_runs(strain: np.ndarray) -> list[tuple[int, int]]:
    """Return the half-open [start, stop) index ranges where the strain is non-zero."""
    occupied = strain != 0.0
    if not np.any(occupied):
        return []
    padded = np.concatenate(([False], occupied, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _merged_spans(events: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Merge the catalogue's sample spans, so overlapping glitches make one span.

    The strain cannot distinguish two overlapping glitches from one longer one, so
    comparing the catalogue's spans to the strain's non-zero runs has to merge first.
    Without this the assertion would only hold for a realization that happened to draw
    no overlaps, which is the kind of test that passes on one seed.
    """
    spans = sorted(
        (int(event["sample_index"]), int(event["sample_index"]) + int(event["n_samples"])) for event in events
    )
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def test_catalogue_spans_match_the_injected_strain_exactly() -> None:
    """Every catalogue row corresponds to samples the strain actually carries."""
    sampling_frequency = 256.0
    duration = 120.0
    model = BlipGlitch(
        rate=0.2,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)

    strain = simulator.generate(duration, sampling_frequency, ["H1"], seed=11)["H1"]
    events = simulator.glitch_events

    assert events, "test is vacuous: no glitch was injected"
    assert _merged_spans(events) == _nonzero_runs(strain)
    for event in events:
        assert event["detector"] == "H1"
        assert event["kind"] == "blip"
        assert event["gps_start_time"] == O3_EPOCH + (event["sample_index"] / sampling_frequency)
        span = strain[event["sample_index"] : event["sample_index"] + event["n_samples"]]
        peak_index = event["sample_index"] + int(np.argmax(np.abs(span)))
        assert event["gps_peak_time"] == O3_EPOCH + (peak_index / sampling_frequency)


def test_catalogue_peak_is_not_the_start_time() -> None:
    """The two time columns name different instants, as the schema says they do."""
    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.05,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    simulator.generate(60.0, 256.0, ["H1"], seed=3)

    events = simulator.glitch_events
    assert events, "test is vacuous: no glitch was injected"
    # A blip's envelope peaks at the centre of its support, so the peak sits well after
    # the Poisson event time. A consumer that read gps_start_time as the peak would cut
    # its window in the wrong place.
    for event in events:
        offset = event["gps_peak_time"] - event["gps_start_time"]
        assert offset > 0.0
        # The carrier is white noise, so the peak jitters around the envelope's centre
        # rather than landing on it; what matters is that it is nowhere near the start.
        assert offset == pytest.approx(0.5 * event["duration_seconds"], abs=0.25 * event["duration_seconds"])


def test_catalogue_survives_batching_and_boundary_carry_over() -> None:
    """A streamed run records the same events, once each, at their true times."""
    sampling_frequency = 256.0
    chunk = 2.0
    n_chunks = 4
    detectors = ["H1", "L1"]
    # 1 s waveforms at ~1/s straddle the 2 s chunk boundaries; std=0 keeps the
    # scattered-light waveform deterministic.
    model = ScatteredLightGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=1.0,
        peak_frequency=20.0,
    )

    streamed_simulator = InjectGlitches(_zero_base(detectors), [model], gps_start=O3_EPOCH)
    stream = streamed_simulator.generate_stream(chunk, sampling_frequency, detectors, seed=5)
    for _ in range(n_chunks):
        next(stream)

    single_simulator = InjectGlitches(_zero_base(detectors), [model], gps_start=O3_EPOCH)
    single_simulator.generate(chunk * n_chunks, sampling_frequency, detectors, seed=5)

    streamed = streamed_simulator.glitch_events
    single = single_simulator.glitch_events
    assert streamed, "test is vacuous: no glitch was injected"

    chunk_samples = int(chunk * sampling_frequency)
    straddling = [event for event in streamed if event["sample_index"] + event["n_samples"] > chunk_samples]
    assert straddling, "test is vacuous: no glitch straddled a chunk boundary"

    # Once each, not once per chunk its samples reach.
    event_ids = [event["event_id"] for event in streamed]
    assert len(event_ids) == len(set(event_ids))

    # The batched and single-shot catalogues agree on every column that does not name
    # the batching itself.
    def _identity(event: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in event.items() if key not in {"segment_index", "sample_index"}}

    assert [_identity(event) for event in streamed] == [_identity(event) for event in single]

    # A straddling glitch is filed against the chunk holding its first sample, at its
    # true time -- not against the chunk its tail lands in.
    for event in straddling:
        # `sample_index` is an offset inside its own segment, and the GPS time is built
        # from the segment's epoch plus that offset -- so a glitch whose tail spills into
        # the next chunk is still filed at its true start, in the chunk that holds it.
        assert 0 <= event["sample_index"] < chunk_samples
        expected_gps = O3_EPOCH + (event["segment_index"] * chunk) + (event["sample_index"] / sampling_frequency)
        assert event["gps_start_time"] == expected_gps


def test_segment_catalogue_reports_only_the_latest_chunk() -> None:
    """Per-batch consumers get the rows for the chunk they are about to write."""
    sampling_frequency = 256.0
    chunk = 5.0
    model = BlipGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    stream = simulator.generate_stream(chunk, sampling_frequency, ["H1"], seed=21)

    collected: list[dict[str, Any]] = []
    for index in range(4):
        next(stream)
        segment_events = simulator.segment_glitch_events
        for event in segment_events:
            assert event["segment_index"] == index
            assert O3_EPOCH + (index * chunk) <= event["gps_start_time"] < O3_EPOCH + ((index + 1) * chunk)
        collected.extend(segment_events)

    assert collected, "test is vacuous: no glitch was injected"
    assert collected == simulator.glitch_events


def test_catalogue_is_reproducible_for_a_fixed_seed() -> None:
    """The catalogue replays exactly, like the strain it describes."""
    models = [
        BlipGlitch(
            rate=0.5,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.75, std=0.3),
            width=0.015,
        ),
        ScatteredLightGlitch(
            rate=0.2,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.5, std=0.0),
            duration=0.25,
            peak_frequency=24.0,
        ),
    ]
    first = InjectGlitches(_zero_base(["H1", "L1"]), models, gps_start=O3_EPOCH)
    second = InjectGlitches(_zero_base(["H1", "L1"]), models, gps_start=O3_EPOCH)

    first.generate(30.0, 512.0, ["H1", "L1"], seed=1234)
    second.generate(30.0, 512.0, ["H1", "L1"], seed=1234)

    assert first.glitch_events, "test is vacuous: no glitch was injected"
    assert first.glitch_events == second.glitch_events
    # Both models and both detectors are represented, so the identity columns are
    # actually being exercised rather than trivially equal.
    assert {event["model_index"] for event in first.glitch_events} == {0, 1}
    assert {event["detector"] for event in first.glitch_events} == {"H1", "L1"}


def test_a_seeded_segment_keeps_the_epoch_its_caller_assigned() -> None:
    """The bug that shipped nothing: a seed must not rewind the caller's epoch.

    A segment writer states where the segment sits in GPS time and *then* generates,
    and the first segment of a run is the one that carries the seed. Re-initialising the
    Poisson process used to reset the epoch to the constructor's, so that segment was
    timed from the wrong instant and the auto-advance carried the error into every later
    segment. On the streaming path -- a seed on the first chunk and none after, which is
    how a batch-by-batch consumer drives this -- that put the *whole run* on the wrong
    epoch: a stream told to start at GPS 1256655618 recorded its first glitch at 0.03 s.
    """
    model = BlipGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    # Constructed with the default epoch, so only the assignment can put the rows in the
    # right place -- the constructor cannot be what makes this pass.
    simulator = InjectGlitches(_zero_base(["H1"]), [model])
    simulator.gps_start = O3_EPOCH

    simulator.generate(4.0, 256.0, ["H1"], seed=5)

    events = simulator.glitch_events
    assert events, "test is vacuous: no glitch was injected"
    for event in events:
        assert event["gps_start_time"] >= O3_EPOCH
        assert event["gps_start_time"] < O3_EPOCH + 4.0


def test_every_streamed_segment_is_timed_from_the_epoch_it_was_given() -> None:
    """Non-contiguous segments, each timed against its own epoch, seed on the first.

    This is `FrameWriter.write_segments`' shape: the writer assigns each segment's epoch
    and seeds only the first. A catalogue that ignored the assignment would put segment
    two's glitches under segment one's frame name.
    """
    sampling_frequency = 256.0
    duration = 4.0
    epochs = [O3_EPOCH, O3_EPOCH + 1000.0]
    model = BlipGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model])

    for index, epoch in enumerate(epochs):
        apply_segment_gps_start(simulator, epoch)
        simulator.generate(duration, sampling_frequency, ["H1"], seed=5 if index == 0 else None)
        segment_events = simulator.segment_glitch_events
        assert segment_events, f"test is vacuous: segment {index} injected nothing"
        for event in segment_events:
            assert epoch <= event["gps_start_time"] < epoch + duration


def test_reset_rewinds_the_epoch_and_restarts_the_catalogue() -> None:
    """`reset` is the explicit start-over, so it does rewind -- unlike a bare re-seed."""
    model = BlipGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)

    simulator.generate(10.0, 256.0, ["H1"], seed=7)
    first = simulator.glitch_events
    simulator.reset()
    assert simulator.gps_start == O3_EPOCH
    assert simulator.glitch_events == []
    simulator.generate(10.0, 256.0, ["H1"], seed=7)

    assert first, "test is vacuous: no glitch was injected"
    assert simulator.glitch_events == first


def test_catalogue_records_the_snr_it_calibrated_to_and_the_one_it_achieved(tmp_path: Path) -> None:
    """Target and realized SNR are both recorded, and the realized one is measurable."""
    psd_file = _write_flat_psd(tmp_path / "psd.txt")
    sampling_frequency = 1024.0
    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=2.0, std=0.0),
        width=0.01,
        psd_file=psd_file,
        snr=12.0,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    strain = simulator.generate(40.0, sampling_frequency, ["H1"], seed=5)["H1"]

    events = simulator.glitch_events
    assert events, "test is vacuous: no glitch was injected"
    table = np.loadtxt(psd_file)
    for event in events:
        assert event["target_snr"] == 12.0
        assert event["amplitude"] == 2.0
        # Recorded as what the strain holds, which for an amplitude multiplier of 2 is
        # twice the configured target, not the target itself.
        assert event["realized_snr"] == pytest.approx(24.0, rel=1e-9)

        waveform = strain[event["sample_index"] : event["sample_index"] + event["n_samples"]]
        frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / sampling_frequency)
        psd = np.interp(frequencies, table[:, 0], table[:, 1], left=0.0, right=0.0)
        band = (frequencies >= 2.0) & (frequencies < sampling_frequency / 2.0) & (psd > 0.0)
        waveform_fd = np.fft.rfft(waveform) / sampling_frequency
        delta_frequency = sampling_frequency / waveform.size
        measured = float(np.sqrt(4.0 * delta_frequency * np.sum(np.abs(waveform_fd[band]) ** 2 / psd[band])))
        assert measured == pytest.approx(event["realized_snr"], rel=1e-6)


def test_catalogue_reports_no_snr_for_an_uncalibrated_model() -> None:
    """A model with no PSD has no SNR, and the catalogue says so rather than inventing one."""
    model = BlipGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.5, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    simulator.generate(10.0, 256.0, ["H1"], seed=2)

    events = simulator.glitch_events
    assert events, "test is vacuous: no glitch was injected"
    for event in events:
        assert event["target_snr"] is None
        assert event["realized_snr"] is None
        assert event["amplitude"] == 1.5
        assert event["glitch_class"] is None


def test_metadata_carries_the_catalogue_and_documents_its_columns() -> None:
    """The public metadata surface holds the rows and what each column means."""
    model = BlipGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    simulator.generate(10.0, 256.0, ["H1"], seed=4)

    catalogue = simulator.metadata["glitches"]["catalogue"]
    assert catalogue["schema_version"] == GLITCH_CATALOGUE_SCHEMA_VERSION
    assert catalogue["events"] == simulator.glitch_events
    assert catalogue["events"], "test is vacuous: no glitch was injected"
    # Every column a row carries is documented, and nothing is documented that no row
    # carries -- a column with no description is one a consumer has to guess at.
    assert set(catalogue["columns"]) == set(catalogue["events"][0])
    assert set(catalogue["columns"]) == set(GLITCH_CATALOGUE_COLUMNS)
    # The start-versus-peak ambiguity is stated, not left to be inferred.
    convention = catalogue["time_convention"]
    assert "FIRST SAMPLE" in convention
    assert "not the peak" in convention

    # The counts stay: they are the run-level summary, and the rows do not replace them.
    assert simulator.metadata["glitches"]["counts"][0]["count"] == len(catalogue["events"])


def test_third_party_model_without_draw_still_reaches_the_catalogue() -> None:
    """A model implementing only generate_waveform is recorded, with blanks not guesses."""

    class BareModel(BlipGlitch):
        """A model that overrides generate_waveform and nothing else."""

        def generate_waveform(
            self,
            sampling_frequency: float,
            rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            _ = rng
            return np.ones(round(0.1 * sampling_frequency), dtype=float)

    model = BareModel(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    simulator.generate(10.0, 256.0, ["H1"], seed=8)

    events = simulator.glitch_events
    assert events, "test is vacuous: no glitch was injected"
    for event in events:
        assert event["kind"] == "blip"
        assert event["n_samples"] == round(0.1 * 256.0)
        assert event["amplitude"] is None
        assert event["target_snr"] is None
        assert event["realized_snr"] is None


def test_empty_waveform_events_are_not_recorded() -> None:
    """A draw the strain never received is absent from the catalogue, as from the counts."""

    class EmptyWaveformBlip(BlipGlitch):
        def generate_waveform(
            self,
            sampling_frequency: float,
            rng: np.random.Generator | None = None,
        ) -> np.ndarray:
            _ = (sampling_frequency, rng)
            return np.array([], dtype=float)

    model = EmptyWaveformBlip(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH)
    simulator.generate(5.0, 64.0, ["H1"], seed=3)

    assert simulator.glitch_events == []


def test_run_writes_the_catalogue_into_the_metadata_sidecar(tmp_path: Path) -> None:
    """A configured run's sidecar carries the catalogue in the run's own GPS epoch."""
    output_directory = tmp_path / "output"
    config = NoiseConfig(
        detectors=["H1"],
        duration=60.0,
        sampling_frequency=256.0,
        seed=19,
        output=OutputConfig(directory=output_directory, prefix="glitches", gps_start=O3_EPOCH),
        components=[
            NoiseComponentConfig(
                simulator="glitches",
                options={
                    "models": [
                        {
                            "kind": "blip",
                            "rate": 0.5,
                            "width": 0.01,
                            "amplitude_distribution": {"mean": 1.0, "std": 0.0},
                        }
                    ]
                },
            )
        ],
    )

    DefaultNoiseSimulator().run(config)

    sidecar = json.loads((output_directory / "glitches_H1.json").read_text())
    events = sidecar["glitches"]["catalogue"]["events"]
    assert events, "test is vacuous: no glitch was injected"
    strain = np.load(output_directory / "glitches_H1.npy")
    assert _merged_spans(events) == _nonzero_runs(strain)
    for event in events:
        assert event["gps_start_time"] >= O3_EPOCH
        assert event["gps_start_time"] < O3_EPOCH + config.duration
        assert event["gps_start_time"] == O3_EPOCH + (event["sample_index"] / config.sampling_frequency)


def test_the_run_catalogue_stays_in_time_order_for_out_of_order_segments() -> None:
    """`glitch_events` promises time order, and segments need not arrive in it.

    `FrameWriter.write_segments` accepts its segments in whatever order it is handed, so a
    caller can assign a decreasing epoch. Appending each segment's rows then relies on the
    epochs advancing, which is true of an ordinary run and not guaranteed.
    """
    sampling_frequency = 256.0
    duration = 4.0
    later, earlier = O3_EPOCH + 1000.0, O3_EPOCH
    model = BlipGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model])

    # The later segment first, then the earlier one.
    apply_segment_gps_start(simulator, later)
    simulator.generate(duration, sampling_frequency, ["H1"], seed=5)
    assert simulator.segment_glitch_events, "test is vacuous: the first segment injected nothing"
    apply_segment_gps_start(simulator, earlier)
    simulator.generate(duration, sampling_frequency, ["H1"], seed=None)
    assert simulator.segment_glitch_events, "test is vacuous: the second segment injected nothing"

    times = [event["gps_start_time"] for event in simulator.glitch_events]
    assert times == sorted(times)
    # Both segments are represented, so the ordering is over two blocks rather than one.
    assert min(times) < earlier + duration
    assert max(times) >= later


def test_parallel_workers_are_placed_on_the_segment_epoch() -> None:
    """A parallel run's per-detector workers must carry the writer's epoch, not the factory's.

    `ParallelAdapter` builds one simulator per detector from a factory and caches it, so an
    epoch set on the adapter reached none of them by attribute walking: each worker went on
    stamping its catalogue with the factory's default while the frame carried the writer's
    epoch. Measured before the fix -- a run told to start at GPS 1256655618 recorded its
    glitches at 0.17 s.
    """
    sampling_frequency = 64.0
    duration = 4.0
    detectors = ["H1", "L1"]

    def factory() -> InjectGlitches:
        base = _ZeroNoiseSimulator(
            detectors=detectors, duration=duration, sampling_frequency=sampling_frequency, seed=None
        )
        model = BlipGlitch(
            rate=3.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
            width=0.01,
        )
        return InjectGlitches(base, [model])

    adapter = ParallelAdapter(factory, max_workers=2, backend="thread")
    # Exactly what FrameWriter.write does before generating a segment, and before any
    # worker exists -- so the epoch has to survive until they are built.
    apply_segment_gps_start(adapter, O3_EPOCH)
    adapter.generate(duration, sampling_frequency, detectors, seed=5)

    events = [event for worker in adapter._worker_simulators.values() for event in worker.glitch_events]
    assert events, "test is vacuous: no glitch was injected"
    assert len(adapter._worker_simulators) == len(detectors), "test is vacuous: workers were not built per detector"
    for event in events:
        assert O3_EPOCH <= event["gps_start_time"] < O3_EPOCH + duration

    # A second segment at a new epoch reaches the workers that now exist.
    apply_segment_gps_start(adapter, O3_EPOCH + 1000.0)
    adapter.generate(duration, sampling_frequency, detectors, seed=None)
    second = [event for worker in adapter._worker_simulators.values() for event in worker.segment_glitch_events]
    assert second, "test is vacuous: the second segment injected nothing"
    for event in second:
        assert O3_EPOCH + 1000.0 <= event["gps_start_time"] < O3_EPOCH + 1000.0 + duration


def _spans_clipped_to(events: list[dict[str, Any]], n_samples: int) -> list[tuple[int, int]]:
    """Merge the catalogue's sample spans, clipped to one segment's length.

    A row records the whole drawn waveform, so a span can run past the segment that holds its
    start; comparing against that segment's samples has to clip.
    """
    spans = sorted(
        (int(event["sample_index"]), min(n_samples, int(event["sample_index"]) + int(event["n_samples"])))
        for event in events
    )
    merged: list[tuple[int, int]] = []
    for start, stop in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
        else:
            merged.append((start, stop))
    return merged


def test_a_tail_is_not_replayed_across_a_gps_gap() -> None:
    """A carried tail belongs to the next *data*, not merely the next call.

    `FrameWriter.write_segments` permits gaps, so a caller can place the next segment a
    thousand seconds later — and the interval between them holds no data for a waveform to
    run through. Replaying the tail anyway wrote strain into the new segment that no row of
    its catalogue accounted for, while the row that produced those samples was timestamped
    from before the gap: the strain and the truth then disagreed about when the glitch was.
    """
    sampling_frequency = 256.0
    duration = 2.0
    n_samples = int(duration * sampling_frequency)
    model = ScatteredLightGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=1.0,
        peak_frequency=20.0,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model])

    apply_segment_gps_start(simulator, O3_EPOCH)
    simulator.generate(duration, sampling_frequency, ["H1"], seed=1)
    first = simulator.segment_glitch_events
    # Vacuity guards: this seed must actually leave a waveform running past the boundary.
    assert any(event["sample_index"] + event["n_samples"] > n_samples for event in first), (
        "test is vacuous: no waveform overran the first segment"
    )
    assert any(simulator._pending_tails.values()), "test is vacuous: no tail was queued"

    # A genuine gap, not the contiguous continuation.
    apply_segment_gps_start(simulator, O3_EPOCH + 1000.0)
    strain = simulator.generate(duration, sampling_frequency, ["H1"], seed=None)["H1"]
    second = simulator.segment_glitch_events

    # Nothing before the segment's own first glitch: a replayed tail lands at sample zero,
    # which is exactly what must not be there. Asserted as strain equal to zero rather than as
    # span equality, because a scattered-light waveform's first sample *is* zero by
    # construction (its phase starts at zero), so the non-zero run begins one sample after the
    # row's `sample_index` and strict equality would fail on a correct catalogue.
    assert second, "test is vacuous: the second segment injected nothing"
    first_row = min(event["sample_index"] for event in second)
    assert first_row > 0, "test is vacuous: a row starts at sample zero, so a tail there would hide"
    np.testing.assert_array_equal(strain[:first_row], np.zeros(first_row))

    # And every non-zero sample the segment does hold falls inside one of its own rows.
    covered = np.zeros(n_samples, dtype=bool)
    for start, stop in _spans_clipped_to(second, n_samples):
        covered[start:stop] = True
    assert not np.any((strain != 0.0) & ~covered)

    for event in second:
        assert O3_EPOCH + 1000.0 <= event["gps_start_time"] < O3_EPOCH + 1000.0 + duration


def test_a_contiguous_segment_still_receives_its_tail() -> None:
    """The gap rule must not cost a contiguous run its carry-over.

    Assigning the epoch explicitly — which a writer does for every segment, gap or not — is
    contiguous when it matches the continuation, so the tail is kept and the streamed series
    still matches a single call of the same total duration.
    """
    sampling_frequency = 256.0
    chunk = 2.0
    n_chunks = 3
    model = ScatteredLightGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=1.0,
        peak_frequency=20.0,
    )

    assigned = InjectGlitches(_zero_base(["H1"]), [model])
    chunks = []
    for index in range(n_chunks):
        # Explicitly assigned each time, at the contiguous epoch.
        apply_segment_gps_start(assigned, O3_EPOCH + (index * chunk))
        chunks.append(assigned.generate(chunk, sampling_frequency, ["H1"], seed=1 if index == 0 else None)["H1"])
    streamed = np.concatenate(chunks)

    single = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH).generate(
        chunk * n_chunks, sampling_frequency, ["H1"], seed=1
    )["H1"]

    boundary = int(chunk * sampling_frequency)
    crossed = any(streamed[k * boundary - 1] != 0.0 and streamed[k * boundary] != 0.0 for k in range(1, n_chunks))
    assert crossed, "test is vacuous: no glitch straddled a boundary"
    np.testing.assert_array_equal(streamed, single)


def test_fractional_segments_keep_their_tails_despite_float_rounding() -> None:
    """A one-ULP epoch difference is the same instant, and must not read as a gap.

    A writer computes each epoch as `start + index * duration` while the injector advances by
    adding `duration` once per segment. For a duration binary floating point cannot hold
    exactly the two rounding paths diverge by an ULP -- ~2.4e-7 s at GPS magnitudes, four
    orders of magnitude below a sample period -- and an exact comparison read that as a gap
    and discarded valid tails. Measured before the fix: 38 samples of waveform lost from this
    run, on the segments where the rounding diverged.

    `0.1` is the point: `2.0` is exact, which is why every whole-second test passed.
    """
    sampling_frequency = 250.0
    duration = 0.1  # 25 samples exactly, and not representable in binary
    n_segments = 8
    model = ScatteredLightGlitch(
        rate=30.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=0.25,
        peak_frequency=20.0,
    )

    # Guard the guard: the epochs must actually diverge, or the test proves nothing.
    accumulated = O3_EPOCH
    diverged = False
    for index in range(1, n_segments):
        if O3_EPOCH + (index * duration) != accumulated + duration:
            diverged = True
        accumulated = O3_EPOCH + (index * duration)
    assert diverged, "test is vacuous: the two epoch computations never diverged"

    streamed_simulator = InjectGlitches(_zero_base(["H1"]), [model])
    parts = []
    for index in range(n_segments):
        apply_segment_gps_start(streamed_simulator, O3_EPOCH + (index * duration))
        parts.append(
            streamed_simulator.generate(duration, sampling_frequency, ["H1"], seed=7 if index == 0 else None)["H1"]
        )
    streamed = np.concatenate(parts)

    single = InjectGlitches(_zero_base(["H1"]), [model], gps_start=O3_EPOCH).generate(
        duration * n_segments, sampling_frequency, ["H1"], seed=7
    )["H1"]

    assert np.count_nonzero(single) > 0, "test is vacuous: no glitch was injected"
    np.testing.assert_array_equal(streamed, single)


def test_a_one_sample_offset_is_still_treated_as_a_gap() -> None:
    """The tolerance must not swallow a real discontinuity.

    Half a sample period is the threshold because below it two epochs name the same sample.
    A whole sample of offset is data that does not line up, so the tail does not continue
    there -- and a tolerance loose enough to miss that would have replaced one defect with
    its opposite.
    """
    sampling_frequency = 256.0
    duration = 2.0
    model = ScatteredLightGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=1.0,
        peak_frequency=20.0,
    )
    simulator = InjectGlitches(_zero_base(["H1"]), [model])

    apply_segment_gps_start(simulator, O3_EPOCH)
    simulator.generate(duration, sampling_frequency, ["H1"], seed=1)
    assert any(simulator._pending_tails.values()), "test is vacuous: no tail was queued"

    # One sample later than the contiguous continuation.
    apply_segment_gps_start(simulator, O3_EPOCH + duration + (1.0 / sampling_frequency))
    strain = simulator.generate(duration, sampling_frequency, ["H1"], seed=None)["H1"]
    events = simulator.segment_glitch_events

    assert events, "test is vacuous: the second segment injected nothing"
    first_row = min(event["sample_index"] for event in events)
    assert first_row > 0, "test is vacuous: a row at sample zero would hide a replayed tail"
    np.testing.assert_array_equal(strain[:first_row], np.zeros(first_row))
