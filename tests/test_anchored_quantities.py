"""The ARMA line defaults must keep agreeing with the measurement behind them.

``DEFAULT_LINE_WIDTH_HZ`` and ``DEFAULT_LINE_PROMINENCE`` are measured from the
lines the bundled PSD presets tabulate rather than chosen. The tests here
re-run those measurements with the shipped harness and check the constants
still follow from them, so adding, replacing or re-sampling a bundled preset
cannot silently leave either default behind. ``docs/dev/anchored_quantities.md``
records what the numbers were when they were set.
"""

from __future__ import annotations

import numpy as np
import pytest
from measure_anchored_quantities import (
    PRESETS,
    design_grid,
    full_width_half_maximum,
    tabulated_lines,
)

from gwmock_noise.simulators import ARMANoiseSimulator
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.arma import (
    DEFAULT_LINE_PROMINENCE,
    DEFAULT_LINE_WIDTH_HZ,
    DEFAULT_MAX_LINES,
    _detect_line_frequencies,
)

SAMPLING_FREQUENCY = 4096.0


def _population() -> list[tuple[np.ndarray, np.ndarray, list[float], float]]:
    """Return each preset's in-band target, its tabulated lines and a match window.

    Returns:
        One entry per bundled preset: the design-grid frequencies and values the
        detector sees, the tabulated line frequencies, and how far a detection
        may sit from a tabulated line and still be the same line -- two steps of
        the design grid plus one of the table the line was read from.
    """
    population = []
    for psd_file, low_frequency in PRESETS.items():
        frequencies, values = design_grid(psd_file, low_frequency)
        table_frequencies, _ = load_spectral_series(psd_file, kind="PSD")
        window = 2.0 * float(frequencies[1] - frequencies[0]) + float(np.median(np.diff(table_frequencies)))
        population.append((frequencies, values, tabulated_lines(psd_file, low_frequency), window))
    return population


def test_default_line_width_is_the_measured_median() -> None:
    """The default width is the median width of the lines the presets tabulate."""
    widths = []
    for frequencies, values, lines, _ in _population():
        widths.extend(full_width_half_maximum(frequencies, values, line) for line in lines)
    measured = np.array([width for width in widths if np.isfinite(width)])
    assert measured.size > 50, "the bundled presets should supply a line population to measure"
    assert pytest.approx(float(np.median(measured)), abs=0.01) == DEFAULT_LINE_WIDTH_HZ


def test_default_prominence_lies_in_the_gap_between_the_measured_populations() -> None:
    """No non-line candidate reaches the default, and a tabulated line still does.

    Scoring every candidate the detector would consider splits them in two: the
    strongest that is not a tabulated line, and the weakest tabulated line that
    stands above it. The default has to sit between the two, which is the
    property that makes it an anchored number rather than a chosen one.
    """
    strongest_false = 0.0
    true_ratios = []
    for frequencies, values, lines, window in _population():
        median = float(np.median(values))
        for candidate in _detect_line_frequencies(frequencies, values, max_lines=DEFAULT_MAX_LINES, prominence=1.0):
            ratio = float(values[int(np.argmin(np.abs(frequencies - candidate)))] / median)
            if any(abs(candidate - line) <= window for line in lines):
                true_ratios.append(ratio)
            else:
                strongest_false = max(strongest_false, ratio)

    above_the_false_floor = [ratio for ratio in true_ratios if ratio > strongest_false]
    assert above_the_false_floor, "a threshold clear of every false candidate must still keep a tabulated line"
    assert strongest_false < DEFAULT_LINE_PROMINENCE <= min(above_the_false_floor)


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
