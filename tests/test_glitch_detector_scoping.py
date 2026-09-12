"""Tests for scoping a glitch model to a subset of interferometers."""

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.glitches import BlipGlitch, LogNormalAmplitudeDistribution
from gwmock_noise.glitches._coloring import load_psd_table
from gwmock_noise.glitches.models import normalize_glitch_models, validate_glitch_detector_coverage
from gwmock_noise.simulators import DefaultNoiseSimulator, InjectGlitches
from gwmock_noise.simulators.glitches import _ZeroNoiseSimulator

#: The ET case from the report: one configuration naming both designs resolves to
#: interferometers of two different arm lengths, which do not share a noise curve.
TRIANGLE = "ET1_SARD"
TWO_L = "ET2_2L_ALIGNED_EMR"
TRIANGLE_PSD = "ET_10_full_cryo_psd"
TWO_L_PSD = "ET_15_full_cryo_psd"

TARGET_SNR = 20.0
#: The unscoped baseline the regression story rests on, measured from the written
#: strain: one model with no selector colors the whole network against the 10 km
#: curve, so a glitch calibrated to SNR 20 realizes 20.0 against that curve and
#: ~24.2 against the 15 km curve the 2L interferometer actually has.
UNSCOPED_BASELINE_SNR = (20.0, 24.2)
SAMPLING_FREQUENCY = 4096.0
DURATION = 8.0
AMPLITUDE = {"distribution": "lognormal", "mean": 1.0, "std": 0.0}


def _optimal_snr(waveform: np.ndarray, psd_file: str | Path, *, low_frequency_cutoff: float = 2.0) -> float:
    """Recompute the optimal SNR of a strain slice against a PSD table.

    Deliberately re-derived from the written strain rather than read from the
    catalogue: the question is what the frames carry, and the catalogue reports
    what the model believed it injected.
    """
    frequencies, values = load_psd_table(psd_file)
    grid = np.fft.rfftfreq(waveform.size, d=1.0 / SAMPLING_FREQUENCY)
    psd = np.interp(grid, frequencies, values, left=0.0, right=0.0)
    band = (grid >= low_frequency_cutoff) & (grid < SAMPLING_FREQUENCY / 2.0) & (psd > 0.0)
    waveform_fd = np.fft.rfft(waveform) / SAMPLING_FREQUENCY
    delta_frequency = SAMPLING_FREQUENCY / waveform.size
    return float(np.sqrt(4.0 * delta_frequency * np.sum(np.abs(waveform_fd[band]) ** 2 / psd[band])))


def _psd_table_snr_ratio(
    coloring_psd: str | Path,
    evaluated_psd: str | Path,
    *,
    n_samples: int,
    low_frequency_cutoff: float = 2.0,
) -> float:
    """Predict an SNR ratio from the two PSD tables alone.

    A blip is white noise under a Gaussian envelope, so a waveform colored against
    one curve has spectrum ``|W(f)|^2 S_coloring(f)`` with ``E|W(f)|^2`` flat across
    the band. The ratio of the SNRs it realizes against two curves is then
    ``sqrt(mean(S_coloring / S_evaluated))`` over the band. Nothing of the coloring
    code enters it, which is the point: it says the pinned ~24.2 is what these two
    noise curves imply, not merely what this implementation currently emits.
    """
    grid = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLING_FREQUENCY)
    coloring = np.interp(grid, *load_psd_table(coloring_psd), left=0.0, right=0.0)
    evaluated = np.interp(grid, *load_psd_table(evaluated_psd), left=0.0, right=0.0)
    band = (grid >= low_frequency_cutoff) & (grid < SAMPLING_FREQUENCY / 2.0) & (coloring > 0.0) & (evaluated > 0.0)
    return float(np.sqrt(np.mean(coloring[band] / evaluated[band])))


def _blip(
    *,
    rate: float,
    psd_file: str | None = None,
    snr: float | None = None,
    detectors: Any = None,
) -> dict[str, Any]:
    """Build one blip-model config entry."""
    entry: dict[str, Any] = {"kind": "blip", "rate": rate, "width": 0.01, "amplitude_distribution": dict(AMPLITUDE)}
    if psd_file is not None:
        entry["psd_file"] = psd_file
    if snr is not None:
        entry["snr"] = snr
    if detectors is not None:
        entry["detectors"] = detectors
    return entry


def _run(tmp_path: Path, *, detectors: list[str], models: list[dict[str, Any]], seed: int = 1234) -> Path:
    """Run a glitches-only configuration and return the output directory."""
    out_dir = tmp_path / "output"
    config = NoiseConfig(
        detectors=detectors,
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        output=OutputConfig(directory=out_dir, prefix="scoped"),
        seed=seed,
        components=[{"simulator": "glitches", "models": models}],
    )
    DefaultNoiseSimulator().run(config)
    return out_dir


def _catalogue(out_dir: Path, detector: str) -> list[dict[str, Any]]:
    """Read the truth catalogue from one detector's metadata sidecar."""
    metadata = json.loads((out_dir / f"scoped_{detector}.json").read_text())
    return metadata["glitches"]["catalogue"]["events"]


def _isolated_events(events: list[dict[str, Any]], *, n_samples: int) -> list[dict[str, Any]]:
    """Return events whose samples are wholly in the data and overlap no other event.

    An SNR measured on a slice holding two overlapping glitches is the pair's, not
    the model's, so the check is made only where the strain carries one waveform.
    """
    isolated = []
    for event in events:
        start = event["sample_index"]
        stop = start + event["n_samples"]
        if stop > n_samples:
            continue
        neighbours = [
            other
            for other in events
            if other is not event
            and other["detector"] == event["detector"]
            and other["sample_index"] < stop
            and other["sample_index"] + other["n_samples"] > start
        ]
        if not neighbours:
            isolated.append(event)
    return isolated


def test_scoped_models_color_each_interferometer_against_its_own_psd(tmp_path: Path) -> None:
    """One configuration colors each interferometer against the PSD it was assigned.

    The regression: with no selector both models applied to both interferometers,
    so the 10 km triangle curve colored the 15 km interferometer too -- a 23-44%
    SNR error the run reported nothing about. Measured the way the report measured
    it, by recomputing SNR from the written strain against both candidate PSDs.
    """
    out_dir = _run(
        tmp_path,
        detectors=[TRIANGLE, TWO_L],
        models=[
            _blip(rate=1.0, psd_file=TRIANGLE_PSD, snr=TARGET_SNR, detectors=[TRIANGLE]),
            _blip(rate=1.0, psd_file=TWO_L_PSD, snr=TARGET_SNR, detectors=[TWO_L]),
        ],
    )

    strain = {detector: np.load(out_dir / f"scoped_{detector}.npy") for detector in (TRIANGLE, TWO_L)}
    n_samples = round(DURATION * SAMPLING_FREQUENCY)
    assigned = {TRIANGLE: TRIANGLE_PSD, TWO_L: TWO_L_PSD}
    rejected = {TRIANGLE: TWO_L_PSD, TWO_L: TRIANGLE_PSD}

    checked = dict.fromkeys(assigned, 0)
    for event in _isolated_events(_catalogue(out_dir, TRIANGLE), n_samples=n_samples):
        detector = event["detector"]
        start = event["sample_index"]
        waveform = strain[detector][start : start + event["n_samples"]]
        assert _optimal_snr(waveform, assigned[detector]) == pytest.approx(TARGET_SNR, rel=1e-6)
        # The other design's curve: what the strain would have been calibrated against
        # had the model applied to every interferometer.
        assert abs(_optimal_snr(waveform, rejected[detector]) - TARGET_SNR) / TARGET_SNR > 0.1
        checked[detector] += 1

    assert all(count > 0 for count in checked.values()), checked


def test_per_interferometer_rates_fire_only_where_they_are_scoped(tmp_path: Path) -> None:
    """A scoped model's Poisson process runs in its own interferometers only."""
    out_dir = _run(
        tmp_path,
        detectors=[TRIANGLE, TWO_L],
        models=[
            _blip(rate=4.0, detectors=[TRIANGLE]),
            _blip(rate=0.25, detectors=[TWO_L]),
        ],
    )
    counts = json.loads((out_dir / f"scoped_{TRIANGLE}.json").read_text())["glitches"]["counts"]

    assert set(counts[0]["count_by_detector"]) == {TRIANGLE}
    assert set(counts[1]["count_by_detector"]) == {TWO_L}
    assert counts[0]["count_by_detector"][TRIANGLE] > counts[1]["count_by_detector"][TWO_L]


def test_unscoped_model_still_applies_to_every_interferometer(tmp_path: Path) -> None:
    """A model with no selector means exactly what it means today: all of them."""
    out_dir = _run(tmp_path, detectors=[TRIANGLE, TWO_L], models=[_blip(rate=1.0)])
    events = _catalogue(out_dir, TRIANGLE)

    assert {event["detector"] for event in events} == {TRIANGLE, TWO_L}
    for detector in (TRIANGLE, TWO_L):
        assert np.any(np.load(out_dir / f"scoped_{detector}.npy") != 0.0)


def test_unscoped_model_colors_the_whole_network_against_the_one_configured_curve(tmp_path: Path) -> None:
    """The unscoped baseline, pinned: SNR 20.0 against the configured curve, ~24.2 against the other.

    A model with no selector means all of them, so one ``psd_file`` colors
    interferometers of both arm lengths -- and the 2L interferometer's strain is
    calibrated against the 10 km curve rather than the 15 km curve it has. The
    scoped case above is read against this pair, so it is measured here too, from
    the written strain against both candidate PSDs: without it a later change that
    silently re-colored the unscoped path would leave the suite green.
    """
    configured_snr, other_curve_snr = UNSCOPED_BASELINE_SNR
    out_dir = _run(
        tmp_path,
        detectors=[TRIANGLE, TWO_L],
        models=[_blip(rate=1.0, psd_file=TRIANGLE_PSD, snr=configured_snr)],
    )

    strain = {detector: np.load(out_dir / f"scoped_{detector}.npy") for detector in (TRIANGLE, TWO_L)}
    n_samples = round(DURATION * SAMPLING_FREQUENCY)
    isolated = _isolated_events(_catalogue(out_dir, TRIANGLE), n_samples=n_samples)

    realized: dict[str, list[float]] = {TRIANGLE: [], TWO_L: []}
    for event in isolated:
        detector = event["detector"]
        start = event["sample_index"]
        waveform = strain[detector][start : start + event["n_samples"]]
        # The one configured curve reaches every interferometer, so every glitch
        # carries exactly the SNR it was calibrated to against that curve.
        assert _optimal_snr(waveform, TRIANGLE_PSD) == pytest.approx(configured_snr, rel=1e-6)
        against_other = _optimal_snr(waveform, TWO_L_PSD)
        # Against the 15 km curve -- the one the 2L interferometer actually has --
        # the same strain reads high, because it was never calibrated against it.
        assert against_other == pytest.approx(other_curve_snr, rel=0.05)
        realized[detector].append(against_other)

    assert all(realized.values()), realized

    # A blip's length is fixed by its width, so every event shares one FFT grid.
    widths = {event["n_samples"] for event in isolated}
    assert len(widths) == 1
    # Anchored: the mean ratio the strain shows is the ratio the two PSD tables give
    # on their own, so ~24.2 is a property of these noise curves and not only of the
    # code that wrote the strain.
    measured = [snr for values in realized.values() for snr in values]
    anchor = _psd_table_snr_ratio(TRIANGLE_PSD, TWO_L_PSD, n_samples=widths.pop())
    assert float(np.mean(measured)) / configured_snr == pytest.approx(anchor, rel=0.01)


def test_unscoped_model_matches_a_selector_naming_every_interferometer() -> None:
    """Naming every interferometer explicitly is the same run, sample for sample."""
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.0)
    detectors = [TRIANGLE, TWO_L]

    def strain(selector: list[str] | None) -> dict[str, np.ndarray]:
        model = BlipGlitch(rate=1.0, amplitude_distribution=amplitude, width=0.01, detectors=selector)
        injector = InjectGlitches(
            _ZeroNoiseSimulator(detectors=detectors, duration=DURATION, sampling_frequency=256.0, seed=7),
            [model],
        )
        return injector.generate(DURATION, 256.0, detectors, seed=7)

    unscoped = strain(None)
    explicit = strain(detectors)
    for detector in detectors:
        np.testing.assert_array_equal(unscoped[detector], explicit[detector])


def test_incomplete_coverage_refuses_and_names_the_uncovered_interferometers(tmp_path: Path) -> None:
    """A run no model accounts for in full refuses rather than producing silence."""
    with pytest.raises(ValueError, match="no glitch model applies to") as excinfo:
        _run(
            tmp_path,
            detectors=[TRIANGLE, TWO_L, "ET3_SARD"],
            models=[_blip(rate=1.0, psd_file=TRIANGLE_PSD, snr=TARGET_SNR, detectors=[TRIANGLE])],
        )

    message = str(excinfo.value)
    assert TWO_L in message
    assert "ET3_SARD" in message
    # Named precisely: the covered interferometer is not swept into the complaint.
    assert TRIANGLE not in message


def test_conflicting_coloring_for_one_interferometer_refuses(tmp_path: Path) -> None:
    """Two coloring PSDs claiming one interferometer is a contradiction, not a choice."""
    with pytest.raises(ValueError, match="more than one PSD") as excinfo:
        _run(
            tmp_path,
            detectors=[TRIANGLE],
            models=[
                _blip(rate=1.0, psd_file=TRIANGLE_PSD, snr=TARGET_SNR),
                _blip(rate=1.0, psd_file=TWO_L_PSD, snr=TARGET_SNR, detectors=[TRIANGLE]),
            ],
        )

    message = str(excinfo.value)
    assert TRIANGLE in message
    assert TRIANGLE_PSD in message
    assert TWO_L_PSD in message


def test_a_preset_name_and_its_bundled_path_are_one_curve() -> None:
    """Naming one curve two ways over one interferometer is not a coloring conflict.

    ``ET_15_full_cryo_psd`` and the file it resolves to are the same noise curve,
    so two models spelling it differently do not disagree about what instrument the
    interferometer is. Compared as written they would: a configuration that names
    the preset in one model and an installed path in another -- the spelling a
    reader reaches for when they have the file in front of them -- would be refused
    for a contradiction it does not contain.
    """
    bundled = Path(str(importlib.resources.files("gwmock_noise.data.psd").joinpath(f"{TWO_L_PSD}.txt")))
    assert bundled.is_file()

    models = normalize_glitch_models(
        [
            _blip(rate=1.0, psd_file=TWO_L_PSD, snr=TARGET_SNR, detectors=[TWO_L]),
            _blip(rate=1.0, psd_file=str(bundled), snr=TARGET_SNR, detectors=[TWO_L]),
        ]
    )

    assert models[0].coloring_reference() == models[1].coloring_reference()
    # Raises if the two spellings are read as two curves claiming one interferometer.
    validate_glitch_detector_coverage(models, [TWO_L])


def test_selector_naming_an_absent_interferometer_refuses(tmp_path: Path) -> None:
    """A model scoped to an interferometer the run does not have never fires."""
    with pytest.raises(ValueError, match="this run does not have") as excinfo:
        _run(
            tmp_path,
            detectors=[TRIANGLE],
            models=[_blip(rate=1.0, detectors=[TRIANGLE, "V1"])],
        )

    assert "V1" in str(excinfo.value)


def test_zero_rate_model_covers_a_glitch_free_interferometer(tmp_path: Path) -> None:
    """An interferometer stated to carry no glitches is covered, not uncovered."""
    out_dir = _run(
        tmp_path,
        detectors=[TRIANGLE, TWO_L],
        models=[
            _blip(rate=1.0, psd_file=TRIANGLE_PSD, snr=TARGET_SNR, detectors=[TRIANGLE]),
            _blip(rate=0.0, detectors=[TWO_L]),
        ],
    )

    assert np.any(np.load(out_dir / f"scoped_{TRIANGLE}.npy") != 0.0)
    assert not np.any(np.load(out_dir / f"scoped_{TWO_L}.npy") != 0.0)


def test_selector_accepts_a_single_interferometer_name() -> None:
    """A bare string is one interferometer, not a sequence of characters."""
    model = BlipGlitch(
        rate=1.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        detectors=TRIANGLE,
    )

    assert model.applies_to(TRIANGLE)
    assert not model.applies_to(TWO_L)
    assert model.serialize()["detectors"] == [TRIANGLE]


def test_serialize_records_the_selector() -> None:
    """The sidecar says which interferometers a model applied to, or that it applied to all."""
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.0)

    assert BlipGlitch(rate=1.0, amplitude_distribution=amplitude).serialize()["detectors"] is None
    scoped = BlipGlitch(rate=1.0, amplitude_distribution=amplitude, detectors=[TWO_L, TRIANGLE])
    assert scoped.serialize()["detectors"] == [TWO_L, TRIANGLE]


@pytest.mark.parametrize(
    ("selector", "match"),
    [
        ([], "at least one"),
        ([TRIANGLE, TRIANGLE], "duplicate"),
        ([TRIANGLE, ""], "non-empty"),
        ([TRIANGLE, 3], "strings"),
    ],
)
def test_invalid_selectors_are_rejected(selector: Any, match: str) -> None:
    """A selector that names nothing usable is rejected at construction."""
    with pytest.raises((TypeError, ValueError), match=match):
        BlipGlitch(
            rate=1.0,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
            detectors=selector,
        )


def test_the_refusal_reaches_the_injector_built_without_a_config() -> None:
    """A caller that never goes through the config boundary is refused too.

    ``InjectGlitches`` is public API and takes its models directly, so a rule
    enforced only where a configuration is read would let the same wrong run
    through the front door the config never saw.
    """
    detectors = [TRIANGLE, TWO_L]
    injector = InjectGlitches(
        _ZeroNoiseSimulator(detectors=detectors, duration=DURATION, sampling_frequency=256.0, seed=7),
        [
            BlipGlitch(
                rate=1.0,
                amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
                detectors=[TRIANGLE],
            )
        ],
    )

    with pytest.raises(ValueError, match=TWO_L):
        injector.generate(DURATION, 256.0, detectors, seed=7)


def test_an_interferometer_added_mid_stream_is_checked_against_the_models() -> None:
    """Coverage is judged against the interferometers each segment asks for.

    A streaming caller may widen the network between chunks, and the models were
    fixed when the injector was built -- so the question has to be re-asked where
    the answer can change, not once at construction.
    """
    injector = InjectGlitches(
        _ZeroNoiseSimulator(detectors=[TRIANGLE], duration=1.0, sampling_frequency=256.0, seed=7),
        [
            BlipGlitch(
                rate=1.0,
                amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
                detectors=[TRIANGLE],
            )
        ],
    )
    injector.generate(1.0, 256.0, [TRIANGLE], seed=7)

    with pytest.raises(ValueError, match=TWO_L):
        injector.generate(1.0, 256.0, [TRIANGLE, TWO_L])


def test_every_model_kind_accepts_the_selector(tmp_path: Path) -> None:
    """The selector belongs to a glitch model, not to one kind of them.

    The reported failure was a DeepExtractor entry: ``detectors`` was handed
    straight to the constructor, which had no such parameter, so the configuration
    could not even be written.
    """
    psd_file = tmp_path / "psd.txt"
    frequencies = np.linspace(0.0, 2048.0, 129)
    np.savetxt(psd_file, np.column_stack((frequencies, np.full_like(frequencies, 1.0e-46))))

    models = normalize_glitch_models(
        [
            {"kind": "blip", "rate": 0.1, "detectors": TRIANGLE, "amplitude_distribution": dict(AMPLITUDE)},
            {"kind": "scattered_light", "rate": 0.1, "detectors": [TWO_L], "amplitude_distribution": dict(AMPLITUDE)},
            {
                "kind": "deepextractor",
                "rate": 0.1,
                "psd_file": str(psd_file),
                "snr": 8.0,
                "detectors": [TRIANGLE, TWO_L],
                "amplitude_distribution": dict(AMPLITUDE),
            },
        ]
    )

    assert [model.detectors for model in models] == [(TRIANGLE,), (TWO_L,), (TRIANGLE, TWO_L)]
