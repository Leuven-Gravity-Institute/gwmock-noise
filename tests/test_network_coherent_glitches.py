"""Tests for glitch models whose events are shared across the network.

``InjectGlitches`` ran one Poisson process and one random stream per (model,
interferometer) pair, so no two interferometers could carry the same transient. That is the
right default and the measured behaviour of separated sites, but it cannot express the case
a co-located array is built to worry about: a common environmental transient appearing in
every interferometer at once, which is exactly what a coincidence veto and a null stream
assume they will not see.

These tests pin the three properties the new branch has to have, because a plausible
mutation of any of them passes an "it injects something" check:

1. **Coherence.** A network event is one waveform at one time in every participating
   interferometer, not several draws that happen to be close.
2. **Independence of the interferometer list.** What one interferometer receives -- its
   events, their waveforms, and whether it took each one -- must not change when another
   interferometer joins or leaves the run. Paired-geometry comparisons rest on this, and
   nothing about the output would reveal its absence.
3. **The existing guarantees still hold.** Streaming continuity, tail carry-over, gap
   handling and the truth catalogue behave for a network model as they do for an
   independent one.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.glitches import (
    BlipGlitch,
    LogNormalAmplitudeDistribution,
    NetworkCoherence,
    PowerLawSNRDistribution,
    ScatteredLightGlitch,
)
from gwmock_noise.simulators.glitches import (
    GLITCH_CATALOGUE_COLUMNS,
    InjectGlitches,
    _ZeroNoiseSimulator,
)

SAMPLING_FREQUENCY = 256.0


def _zero_base(detectors: list[str], duration: float = 8.0) -> _ZeroNoiseSimulator:
    """Return a zero-valued base simulator, so the strain is the glitch stream itself."""
    return _ZeroNoiseSimulator(
        detectors=detectors,
        duration=duration,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=None,
    )


def _coherent_model(
    *,
    rate: float = 2.0,
    participation_probability: float = 1.0,
    amplitude_ratio_std: float = 0.0,
    detectors: list[str] | None = None,
) -> BlipGlitch:
    """Return an uncolored blip model whose events are shared across the network."""
    return BlipGlitch(
        rate=rate,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.02,
        detectors=detectors,
        network=NetworkCoherence(
            participation_probability=participation_probability,
            amplitude_ratio_std=amplitude_ratio_std,
        ),
    )


def _write_flat_psd(path: Path, *, value: float = 1.0) -> Path:
    """Write a flat PSD table covering the band the tests use."""
    frequencies = np.linspace(0.0, SAMPLING_FREQUENCY, 129)
    np.savetxt(path, np.column_stack((frequencies, np.full_like(frequencies, value))))
    return path


def _generate(
    models: list[BlipGlitch | ScatteredLightGlitch],
    detectors: list[str],
    *,
    seed: int = 4321,
    duration: float = 8.0,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    """Run one segment and return the glitch strain and its truth catalogue."""
    injector = InjectGlitches(_zero_base(detectors, duration), models)
    strain = injector.generate(duration, SAMPLING_FREQUENCY, detectors, seed=seed)
    return strain, injector.glitch_events


# ---------------------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------------------


def test_a_network_event_is_one_waveform_in_every_participating_interferometer() -> None:
    """Participating interferometers get the same samples at the same index, not near-misses."""
    strain, events = _generate([_coherent_model()], ["E1", "E2", "E3"])

    assert events, "the fixture must fire at least one event for this test to say anything."
    assert np.array_equal(strain["E1"], strain["E2"])
    assert np.array_equal(strain["E1"], strain["E3"])

    by_network_event: dict[str, list[dict[str, object]]] = {}
    for record in events:
        by_network_event.setdefault(str(record["network_event_id"]), []).append(record)
    for rows in by_network_event.values():
        assert {row["detector"] for row in rows} == {"E1", "E2", "E3"}
        assert len({row["gps_start_time"] for row in rows}) == 1
        assert len({row["n_samples"] for row in rows}) == 1


def test_an_independent_model_puts_a_different_transient_in_each_interferometer() -> None:
    """The contrast case, so the coherence test above is not passing for a trivial reason."""
    independent = BlipGlitch(
        rate=2.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.02,
    )
    strain, events = _generate([independent], ["E1", "E2", "E3"])

    assert events
    assert not np.array_equal(strain["E1"], strain["E2"])
    assert all(record["network_event_id"] is None for record in events)
    assert all(record["amplitude_ratio"] is None for record in events)


def test_participation_of_one_reaches_every_interferometer_and_zero_reaches_none() -> None:
    """Both ends of the participation range, which a flipped comparison gets backwards.

    Mutating ``random() >= p`` to ``random() < p`` inverts the decision, and inverted
    behaviour at *one* end can look like a Poisson fluctuation. At both ends it cannot: a
    probability of one has to reach everything and a probability of zero has to reach
    nothing, exactly, however many events fire.
    """
    _, always = _generate([_coherent_model(participation_probability=1.0)], ["E1", "E2"])
    assert always
    assert {record["detector"] for record in always} == {"E1", "E2"}

    strain, never = _generate([_coherent_model(participation_probability=0.0)], ["E1", "E2"])
    assert never == []
    assert not np.any(strain["E1"])
    assert not np.any(strain["E2"])


def test_partial_participation_varies_between_events_and_between_interferometers() -> None:
    """Each (event, interferometer) decides for itself.

    Keying the participation stream by the interferometer alone -- dropping the event
    ordinal from the spawn key -- would make every event of a given interferometer decide
    identically, so that interferometer would take all of them or none. The counts below
    are strictly between those two, which that mutation cannot produce.
    """
    detectors = ["E1", "E2", "E3"]
    _, events = _generate(
        [_coherent_model(rate=8.0, participation_probability=0.5)],
        detectors,
        duration=64.0,
    )
    network_events = len({record["network_event_id"] for record in events})
    assert network_events >= 20, "the fixture must fire enough events for the counts to mean something."
    for detector in detectors:
        taken = sum(1 for record in events if record["detector"] == detector)
        assert 0 < taken < network_events


def test_the_shared_waveform_is_scaled_and_recorded_per_interferometer(tmp_path: Path) -> None:
    """A per-interferometer amplitude ratio scales the strain and the reported SNR alike.

    ``realized_snr`` describes what an interferometer received, and SNR is linear in
    amplitude, so a row whose ratio is not one must not report the shared waveform's figure.
    Dropping the scaling, or applying it the wrong way round, is invisible at ratio one --
    which is the registered default -- so it is pinned here at a ratio that is not.
    """
    psd_file = _write_flat_psd(tmp_path / "psd.txt")
    model = ScatteredLightGlitch(
        rate=0.25,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=0.5,
        peak_frequency=24.0,
        psd_file=psd_file,
        snr=PowerLawSNRDistribution(minimum=10.0, alpha=1.5, maximum=100.0),
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=100.0,
        network=NetworkCoherence(participation_probability=1.0, amplitude_ratio_std=0.4),
    )
    strain, events = _generate([model], ["E1", "E2"], seed=3)

    for record in events:
        ratio = record["amplitude_ratio"]
        assert ratio is not None
        assert record["realized_snr"] == pytest.approx(record["target_snr"] * ratio, rel=1e-9)

    first = sorted((row for row in events if row["detector"] == "E1"), key=lambda row: row["sample_index"])
    second = sorted((row for row in events if row["detector"] == "E2"), key=lambda row: row["sample_index"])
    assert len(first) == len(second) >= 2, "the fixture must fire at least two shared events."
    windows = [(int(row["sample_index"]), int(row["sample_index"]) + int(row["n_samples"])) for row in first]
    assert all(left[1] <= right[0] for left, right in pairwise(windows)), (
        "this fixture's events must not overlap, or the per-window comparison below compares sums."
    )
    for one, other, (start, stop) in zip(first, second, windows, strict=True):
        scale = other["amplitude_ratio"] / one["amplitude_ratio"]
        assert scale != 1.0, "a non-zero ratio spread must actually differ between channels."
        assert np.allclose(strain["E2"][start:stop], scale * strain["E1"][start:stop], rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------------------
# Independence of the interferometer list
# ---------------------------------------------------------------------------------------


def test_removing_an_interferometer_leaves_the_others_bit_identical() -> None:
    """The property paired-geometry comparisons rest on.

    Drawing participation from a stream shared by the network, or advancing one
    long-lived stream per interferometer once per event, would both make a channel's
    content depend on the run's interferometer list -- and nothing in the output would say
    so. A geometry comparison would then be measuring the generator.
    """
    triangle, triangle_events = _generate([_coherent_model(participation_probability=0.6)], ["E1", "E2", "E3"])
    pair, pair_events = _generate([_coherent_model(participation_probability=0.6)], ["E1", "E2"])

    assert np.array_equal(triangle["E1"], pair["E1"])
    assert np.array_equal(triangle["E2"], pair["E2"])

    def rows(events: list[dict[str, object]], detector: str) -> list[tuple[object, object]]:
        return [(row["network_event_id"], row["gps_start_time"]) for row in events if row["detector"] == detector]

    assert rows(triangle_events, "E1") == rows(pair_events, "E1")
    assert rows(triangle_events, "E2") == rows(pair_events, "E2")


def test_the_network_event_ordinal_counts_every_event_including_unclaimed_ones() -> None:
    """Ordinals index the network's events, not one interferometer's share of them.

    Counting only the events some interferometer took would make the ordinal -- and
    therefore the participation stream keyed by it -- depend on the run's interferometer
    list, which is the same defect the previous test guards from the other side. The
    visible consequence is that a channel's ordinals have gaps, and they must.
    """
    _, events = _generate(
        [_coherent_model(rate=8.0, participation_probability=0.5)],
        ["E1"],
        duration=64.0,
    )
    ordinals = [int(str(record["network_event_id"]).split("-")[1]) for record in events]
    assert ordinals == sorted(ordinals)
    assert ordinals[-1] + 1 > len(ordinals), "a channel taking half the events must skip ordinals."


def test_a_scoped_network_model_only_reaches_the_interferometers_it_names() -> None:
    """The interferometer selector composes with network coherence."""
    models = [
        _coherent_model(detectors=["E1", "E2"]),
        BlipGlitch(
            rate=0.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
            detectors=["E3"],
        ),
    ]
    strain, events = _generate(models, ["E1", "E2", "E3"])

    assert events
    assert {record["detector"] for record in events} == {"E1", "E2"}
    assert not np.any(strain["E3"])
    assert np.array_equal(strain["E1"], strain["E2"])


# ---------------------------------------------------------------------------------------
# The existing guarantees, for the new branch
# ---------------------------------------------------------------------------------------


def test_a_network_model_streams_to_the_same_samples_as_one_call() -> None:
    """Chunking must not move or truncate a shared waveform."""
    detectors = ["E1", "E2"]
    whole, whole_events = _generate([_coherent_model(rate=4.0)], detectors, duration=16.0)

    streamed = InjectGlitches(_zero_base(detectors, 4.0), [_coherent_model(rate=4.0)])
    chunks: dict[str, list[np.ndarray]] = {detector: [] for detector in detectors}
    seed: int | None = 4321
    for _ in range(4):
        result = streamed.generate(4.0, SAMPLING_FREQUENCY, detectors, seed=seed)
        for detector in detectors:
            chunks[detector].append(result[detector])
        seed = None

    for detector in detectors:
        assert np.array_equal(np.concatenate(chunks[detector]), whole[detector])
    assert [record["event_id"] for record in streamed.glitch_events] == [record["event_id"] for record in whole_events]


def test_a_network_model_reports_every_scoped_interferometer_in_its_counts() -> None:
    """Including one that took nothing, which is the informative case.

    The per-model counts used to be built from the per-(model, interferometer) Poisson
    clocks. A network-coherent model has no such clocks -- it has one for the network -- so
    a metadata block built the old way would silently omit every interferometer of every
    coherent model.
    """
    detectors = ["E1", "E2", "E3"]
    injector = InjectGlitches(_zero_base(detectors), [_coherent_model(participation_probability=0.0)])
    injector.generate(8.0, SAMPLING_FREQUENCY, detectors, seed=99)

    (entry,) = injector.metadata["glitches"]["counts"]
    assert entry["count"] == 0
    assert entry["count_by_detector"] == dict.fromkeys(detectors, 0)


def test_network_metadata_round_trips_the_coherence_specification() -> None:
    """A run replays its coherence from its own metadata."""
    from gwmock_noise.glitches.models import normalize_glitch_models

    model = _coherent_model(participation_probability=0.25, amplitude_ratio_std=0.3)
    payload = model.serialize()
    assert payload["network"] == {"participation_probability": 0.25, "amplitude_ratio_std": 0.3}

    (rebuilt,) = normalize_glitch_models([payload])
    assert rebuilt.serialize() == payload
    assert rebuilt.network == NetworkCoherence(participation_probability=0.25, amplitude_ratio_std=0.3)


def test_the_catalogue_documents_its_new_columns() -> None:
    """A file describes its own columns, so a new column has to arrive with its description."""
    _, events = _generate([_coherent_model()], ["E1", "E2"])
    assert events
    assert set(events[0]) == set(GLITCH_CATALOGUE_COLUMNS)


@pytest.mark.parametrize(
    ("participation", "spread"),
    [(-0.01, 0.0), (1.01, 0.0), (0.5, -0.1)],
)
def test_coherence_refuses_a_parameter_outside_its_range(participation: float, spread: float) -> None:
    """Both ends of the probability, and a negative spread."""
    with pytest.raises(ValueError, match="network"):
        NetworkCoherence(participation_probability=participation, amplitude_ratio_std=spread)


@pytest.mark.parametrize("participation", [0.0, 1.0])
def test_coherence_accepts_both_ends_of_the_probability_range(participation: float) -> None:
    """The bounds are inclusive; a strict comparison would reject the registered default."""
    assert NetworkCoherence(participation_probability=participation).participation_probability == participation


def test_a_network_specification_must_be_a_mapping_or_a_coherence() -> None:
    """A configuration that writes something else is refused where it is written."""
    with pytest.raises(TypeError, match="network must be a mapping"):
        BlipGlitch(
            rate=1.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
            network="yes",  # type: ignore[arg-type]
        )
