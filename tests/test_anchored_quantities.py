"""The ARMA line defaults must keep agreeing with the measurement behind them.

``DEFAULT_LINE_WIDTH_HZ`` and ``DEFAULT_LINE_PROMINENCE`` are measured from the
lines the bundled PSD presets tabulate rather than chosen. The tests here
re-run those measurements through the same functions the shipped harness
prints from -- :func:`pooled_line_widths` and :func:`prominence_gap` -- and
check the constants still follow from them. Sharing the functions is the point:
a test, the harness printout and the reference page then cannot disagree about
which statistic a number came from. Adding, replacing or re-sampling a bundled
preset cannot silently leave either default behind.
``docs/dev/anchored_quantities.md`` records what the numbers were when they
were set.
"""

from __future__ import annotations

import numpy as np
import pytest
from measure_anchored_quantities import (
    detection_population,
    pooled_line_widths,
    prominence_gap,
    selected_candidates,
    strongest_false_local_maximum,
)

from gwmock_noise.simulators import ARMANoiseSimulator
from gwmock_noise.simulators.arma import DEFAULT_LINE_PROMINENCE, DEFAULT_LINE_WIDTH_HZ

SAMPLING_FREQUENCY = 4096.0


def test_default_line_width_is_the_measured_median() -> None:
    """The default width is the median width of the lines the presets tabulate."""
    measured = pooled_line_widths()
    assert measured.size > 50, "the bundled presets should supply a line population to measure"
    assert pytest.approx(float(np.median(measured)), abs=0.01) == DEFAULT_LINE_WIDTH_HZ


def test_default_prominence_lies_in_the_gap_between_the_measured_populations() -> None:
    """No non-line candidate reaches the default, and a tabulated line still does.

    The population is the candidates that reach a preset's top ``max_lines``,
    because those are the only ones the detector can place. Scoring them splits
    them in two: the strongest that is not a tabulated line, and the weakest
    tabulated line standing above it. The default has to sit between the two,
    which is the property that makes it an anchored number rather than a chosen
    one.
    """
    strongest_false, weakest_true = prominence_gap()
    assert np.isfinite(weakest_true), "a threshold clear of every false candidate must still keep a tabulated line"
    assert strongest_false < DEFAULT_LINE_PROMINENCE <= weakest_true


def test_the_unselected_statistic_is_recorded_as_a_different_one() -> None:
    """The strongest non-line local maximum is not a candidate the detector places.

    Scoring every local maximum rather than the selected ones gives a much
    larger floor, which the default does not clear. That is not a contradiction
    -- the feature never reaches its preset's top ``max_lines``, so no threshold
    can place it -- but the two statistics must not be confused for each other,
    which is exactly the mistake this test exists to catch.
    """
    population = detection_population()
    strongest_false, _ = prominence_gap(population)
    unselected, preset, frequency = strongest_false_local_maximum(population)

    assert unselected > strongest_false
    assert unselected > DEFAULT_LINE_PROMINENCE
    case = next(entry for entry in population if entry.psd_file == preset)
    assert all(abs(candidate - frequency) > case.match_window for candidate in selected_candidates(case))


def test_default_detection_places_only_the_tabulated_et_d_line() -> None:
    """On ET-D the detector places the one line the curve tabulates, and no ripple.

    The ET-D design curve carries a single tabulated line near 420 Hz. Below
    the default threshold the detector also places poles on the ripple of the
    steep 10-25 Hz rise, and those poles drive the fit residual to about 1e8.
    """
    simulator = ARMANoiseSimulator(
        psd_file="ET_D_psd",
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        ar_order=192,
        ma_order=32,
        low_frequency_cutoff=5.0,
        detect_lines=True,
    )
    placed = simulator.metadata["autoregressive_moving_average"]["placed_lines"]
    assert [round(line["frequency"]) for line in placed] == [420]
    assert placed[0]["width"] == DEFAULT_LINE_WIDTH_HZ
    assert simulator.metadata["autoregressive_moving_average"]["fit_residual"]["worst_relative_error"] < 1.0
