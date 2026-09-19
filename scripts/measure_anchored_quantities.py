"""Re-measure the quantities the AR/ARMA fit contract is anchored to.

Four numbers in the AR and ARMA simulators are set from measurements over the
bundled PSD presets rather than chosen by hand:

1. the per-band PSD-residual tolerances the fit tests assert;
2. :data:`gwmock_noise.simulators.arma.DEFAULT_LINE_WIDTH_HZ`;
3. :data:`gwmock_noise.simulators.arma.DEFAULT_LINE_PROMINENCE`;
4. the lowest-band residual of the ET-D fit, which is reported rather than set.

This script prints every number quoted in ``docs/dev/anchored_quantities.md``.
It is a measurement harness, not a test: it prints tables and changes nothing.
Run it after touching the fit code, the bundled presets or the band definition,
and update the reference page with whatever it prints.

The functions the tests share with it -- :func:`detection_population`,
:func:`pooled_line_widths` and :func:`prominence_gap` -- are the single
definition of each anchored statistic, so a number on the page, a number a
test asserts and a number this script prints cannot drift apart.

Usage::

    uv run python scripts/measure_anchored_quantities.py           # everything
    uv run python scripts/measure_anchored_quantities.py tolerance line-width
"""

from __future__ import annotations

import argparse
from itertools import pairwise
from typing import NamedTuple

import numpy as np
from scipy.signal import welch

from gwmock_noise.simulators import ARMANoiseSimulator, ARNoiseSimulator
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.arma import (
    DEFAULT_LINE_PROMINENCE,
    DEFAULT_LINE_WIDTH_HZ,
    DEFAULT_MAX_LINES,
    _detect_line_frequencies,
)

#: Sampling frequency every measurement below is made at.
SAMPLING_FREQUENCY = 4096.0

#: Upper edge of the reported band; the lower edge is the preset's cutoff.
HIGH_FREQUENCY = 2000.0

#: Number of geometric bands, matching ``DEFAULT_FIT_BANDS``.
N_BANDS = 8

#: Every bundled preset with the low-frequency cutoff its detector is used at.
PRESETS: dict[str, float] = {
    "aLIGO_O3_actual_H1_psd": 20.0,
    "aLIGO_O3_actual_L1_psd": 20.0,
    "aLIGO_O4_high_projected_psd": 20.0,
    "aLIGO_O4_low_projected_psd": 20.0,
    "ET_D_psd": 5.0,
    "ET_10_HF_psd": 5.0,
    "ET_10_full_cryo_psd": 5.0,
    "ET_15_HF_psd": 5.0,
    "ET_15_full_cryo_psd": 5.0,
    "ET_20_HF_psd": 5.0,
    "ET_20_full_cryo_psd": 5.0,
}

#: The two presets the fit tolerance tests assert on.
TOLERANCE_CASES = (("aLIGO_O4_high_projected_psd", 20.0), ("ET_D_psd", 5.0))

#: Seed groups used to sample the generated-path scatter.
N_SEED_GROUPS = 20

#: Segments per seed group, matching the fit tests.
N_SEGMENTS = 6

#: Seconds per generated segment, matching the fit tests.
SEGMENT_SECONDS = 32.0

#: Local contrast above which a tabulated sample is labelled a spectral line.
TRUTH_CONTRAST = 10.0

#: Half-width of the local baseline window, as a fraction of the frequency.
TRUTH_WINDOW = 0.20

#: Minimum half-width of that window in samples, so its median is defined.
TRUTH_MIN_SAMPLES = 15

#: How far from its nominal frequency a line's peak sample is looked for.
PEAK_SEARCH_SAMPLES = 3

#: Lower edge of the fitted band for the Einstein Telescope presets, in hertz.
ET_LOW_FREQUENCY = 5.0

#: The two ET-D low-frequency ripple features whose score moves with the band.
ET_RIPPLE_HZ = (15.0, 21.2)

#: Preset the pole-placement cost is measured on: a measured curve with lines.
POLE_COST_PSD = "aLIGO_O3_actual_H1_psd"

#: Its strongest tabulated line, used for the single-pole cost measurement.
POLE_COST_LINE_HZ = 998.04

#: The prominence the line defaults were raised from, kept for the before-and-
#: after table; it is not a default any more.
SUPERSEDED_LINE_PROMINENCE = 4.0

#: The line width the defaults were raised from, kept for the same table.
SUPERSEDED_LINE_WIDTH_HZ = 0.5

#: Headroom factor applied to the deterministic part of a fit tolerance.
TOLERANCE_HEADROOM = 1.25

#: Standard deviations of realization scatter covered by a fit tolerance.
TOLERANCE_SIGMA = 5.0


def band_errors(
    frequencies: np.ndarray,
    target: np.ndarray,
    estimate: np.ndarray,
    low: float,
    high: float = HIGH_FREQUENCY,
) -> list[float]:
    """Return the band-integrated relative error in each geometric band.

    Args:
        frequencies: Frequencies of ``target`` and ``estimate``.
        target: Target PSD values.
        estimate: Model or estimated PSD values on the same frequencies.
        low: Lower edge of the reported band in hertz.
        high: Upper edge of the reported band in hertz.

    Returns:
        One relative error per band that contains at least one sample.
    """
    errors = []
    for band_low, band_high in pairwise(np.geomspace(low, high, N_BANDS + 1)):
        selected = (frequencies >= band_low) & (frequencies < band_high)
        if not np.any(selected):
            continue
        reference = float(np.sum(target[selected]))
        if reference <= 0.0:
            continue
        errors.append(abs(float(np.sum(estimate[selected])) - reference) / reference)
    return errors


def build(model: str, psd_file: str, low_frequency: float, **kwargs: object) -> ARNoiseSimulator | ARMANoiseSimulator:
    """Return an AR or ARMA simulator at the orders the fit tests use.

    Args:
        model: Either ``"AR"`` or ``"ARMA"``.
        psd_file: Bundled preset name.
        low_frequency: Lower edge of the fitted band in hertz.
        **kwargs: Extra keyword arguments forwarded to the simulator.

    Returns:
        The fitted simulator.
    """
    common = {
        "psd_file": psd_file,
        "detectors": ["H1"],
        "sampling_frequency": SAMPLING_FREQUENCY,
        "low_frequency_cutoff": low_frequency,
    }
    if model == "AR":
        return ARNoiseSimulator(order=256, **common, **kwargs)  # type: ignore[arg-type]
    return ARMANoiseSimulator(ar_order=192, ma_order=32, **common, **kwargs)  # type: ignore[arg-type]


def design_grid(psd_file: str, low_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the in-band target the ARMA line detector actually sees.

    Args:
        psd_file: Bundled preset name.
        low_frequency: Lower edge of the fitted band in hertz.

    Returns:
        The in-band design-grid frequencies and the interpolated target values.
    """
    table_frequencies, table_values = load_spectral_series(psd_file, kind="PSD")
    grid_size = max(2 * (table_frequencies.size - 1), 8 * 192)
    grid = np.fft.rfftfreq(grid_size, d=1.0 / SAMPLING_FREQUENCY)
    mask = (grid >= low_frequency) & (grid <= SAMPLING_FREQUENCY / 2.0)
    return grid[mask], np.interp(grid[mask], table_frequencies, table_values, left=0.0, right=0.0)


def local_window(frequencies: np.ndarray, index: int) -> slice:
    """Return the index window the local baseline at ``index`` is taken over.

    The window is fractional in frequency so that it means the same thing on
    the aLIGO presets' uniform grids and on the ET presets' logarithmic grid,
    and it is widened to :data:`TRUTH_MIN_SAMPLES` on each side so a median
    over it is still defined near the low-frequency edge.

    Args:
        frequencies: Strictly increasing frequencies.
        index: Index the window is centred on.

    Returns:
        The slice of samples forming the local baseline.
    """
    low = min(int(np.searchsorted(frequencies, frequencies[index] * (1.0 - TRUTH_WINDOW))), index - TRUTH_MIN_SAMPLES)
    high = max(
        int(np.searchsorted(frequencies, frequencies[index] * (1.0 + TRUTH_WINDOW))), index + TRUTH_MIN_SAMPLES + 1
    )
    return slice(max(0, low), min(frequencies.size, high))


def tabulated_lines(psd_file: str, low_frequency: float) -> list[float]:
    """Return the frequencies of the spectral lines a preset tabulates.

    A tabulated sample counts as a line when it is a local maximum and stands
    at least :data:`TRUTH_CONTRAST` times above the median of its
    :func:`local_window`. The criterion is local, so unlike the peak-to-median
    ratio the detector uses it is unaffected by how far the preset's baseline
    varies across the band.

    This is the reference the detector is scored against, so it is worth being
    explicit that it is an operational definition, not a published line list.
    Narrow, isolated lines -- the violin modes, the mains harmonic, the
    calibration line -- are labelled the same way by any reasonable window.
    Broad low-frequency structure is not: a few of the 20-25 Hz features in
    the measured O3 curves move in and out of the list as the window width
    changes, and the numbers below inherit that ambiguity.

    Args:
        psd_file: Bundled preset name.
        low_frequency: Lower edge of the fitted band in hertz.

    Returns:
        The tabulated line frequencies in hertz.
    """
    table_frequencies, table_values = load_spectral_series(psd_file, kind="PSD")
    mask = (table_frequencies >= low_frequency) & (table_frequencies <= SAMPLING_FREQUENCY / 2.0)
    frequencies, values = table_frequencies[mask], table_values[mask]
    lines = []
    for index in range(1, values.size - 1):
        if not (values[index] >= values[index - 1] and values[index] > values[index + 1]):
            continue
        baseline = float(np.median(values[local_window(frequencies, index)]))
        if baseline > 0.0 and values[index] / baseline >= TRUTH_CONTRAST:
            lines.append(float(frequencies[index]))
    return lines


def full_width_half_maximum(frequencies: np.ndarray, values: np.ndarray, line_frequency: float) -> float:
    """Return a line's full width at half its excess over the local baseline.

    The peak is looked for within :data:`PEAK_SEARCH_SAMPLES` of the nominal
    frequency so a neighbouring, stronger line in the same window cannot be
    measured instead; the baseline is the median of the wider local window.

    Args:
        frequencies: Uniformly spaced frequencies.
        values: PSD values on those frequencies.
        line_frequency: Nominal line frequency in hertz.

    Returns:
        The width in hertz, or ``nan`` when the half-power points are not
        bracketed inside the local window.
    """
    nominal = int(np.argmin(np.abs(frequencies - line_frequency)))
    search = slice(max(0, nominal - PEAK_SEARCH_SAMPLES), min(values.size, nominal + PEAK_SEARCH_SAMPLES + 1))
    peak_index = search.start + int(np.argmax(values[search]))
    window = local_window(frequencies, peak_index)
    low, high = max(window.start, 1), min(window.stop, values.size - 1)
    baseline = float(np.median(values[low:high]))
    excess = values[peak_index] - baseline
    if excess <= 0.0:
        return float("nan")
    half = baseline + 0.5 * excess
    left_index, right_index = peak_index, peak_index
    while left_index > low and values[left_index] > half:
        left_index -= 1
    while right_index < high - 1 and values[right_index] > half:
        right_index += 1
    if values[left_index] >= half or values[right_index] >= half:
        return float("nan")
    spacing = float(frequencies[1] - frequencies[0])
    left = (
        frequencies[left_index] + (half - values[left_index]) / (values[left_index + 1] - values[left_index]) * spacing
    )
    right = (
        frequencies[right_index]
        - (half - values[right_index]) / (values[right_index - 1] - values[right_index]) * spacing
    )
    return float(right - left)


class DetectionCase(NamedTuple):
    """One preset's in-band target, its tabulated lines and a match window.

    Attributes:
        psd_file: Bundled preset name.
        frequencies: Design-grid frequencies the detector sees.
        values: Interpolated target values on those frequencies.
        lines: Tabulated line frequencies from :func:`tabulated_lines`.
        match_window: How far a detection may sit from a tabulated line and
            still be that line -- two steps of the design grid plus one of the
            table the line was read from.
    """

    psd_file: str
    frequencies: np.ndarray
    values: np.ndarray
    lines: list[float]
    match_window: float

    def is_tabulated(self, frequency: float) -> bool:
        """Return whether a detected frequency matches a tabulated line.

        Args:
            frequency: Detected line frequency in hertz.

        Returns:
            True when a tabulated line lies within the match window.
        """
        return any(abs(frequency - line) <= self.match_window for line in self.lines)

    def peak_to_median(self, frequency: float) -> float:
        """Return the detector's own score for a frequency on this preset.

        Args:
            frequency: Frequency in hertz.

        Returns:
            The value at the nearest design-grid sample over the band median.
        """
        index = int(np.argmin(np.abs(self.frequencies - frequency)))
        return float(self.values[index] / float(np.median(self.values)))


def detection_population() -> list[DetectionCase]:
    """Return the line-detection population, one entry per bundled preset.

    Returns:
        The eleven presets as :class:`DetectionCase` entries.
    """
    population = []
    for psd_file, low_frequency in PRESETS.items():
        frequencies, values = design_grid(psd_file, low_frequency)
        table_frequencies, _ = load_spectral_series(psd_file, kind="PSD")
        window = 2.0 * float(frequencies[1] - frequencies[0]) + float(np.median(np.diff(table_frequencies)))
        population.append(
            DetectionCase(psd_file, frequencies, values, tabulated_lines(psd_file, low_frequency), window)
        )
    return population


def selected_candidates(case: DetectionCase) -> list[float]:
    """Return the candidates that reach a preset's top ``DEFAULT_MAX_LINES``.

    This is the population the threshold actually decides over. The detector
    keeps the strongest ``max_lines`` candidates *after* applying the
    threshold, and on one preset the band median is a constant, so ordering by
    value and ordering by peak-to-median ratio are the same ordering: raising
    the threshold can only drop members of this set, never admit a new one.
    Scoring the threshold over every local maximum instead answers a different
    question -- see :func:`report_prominence`.

    Args:
        case: One preset's detection population.

    Returns:
        The selected candidate frequencies, in increasing order.
    """
    return _detect_line_frequencies(case.frequencies, case.values, max_lines=DEFAULT_MAX_LINES, prominence=1.0)


def prominence_gap(population: list[DetectionCase] | None = None) -> tuple[float, float]:
    """Return the gap the prominence threshold has to sit inside.

    Args:
        population: The detection population; measured afresh when omitted.

    Returns:
        The largest peak-to-median ratio of a selected candidate that is not a
        tabulated line, and the smallest ratio of a selected tabulated line
        standing above it. Any threshold in between keeps every tabulated line
        that a false-candidate-free threshold can keep, and places no others.
    """
    population = population if population is not None else detection_population()
    strongest_false = 0.0
    true_ratios: list[float] = []
    for case in population:
        for candidate in selected_candidates(case):
            ratio = case.peak_to_median(candidate)
            if case.is_tabulated(candidate):
                true_ratios.append(ratio)
            else:
                strongest_false = max(strongest_false, ratio)
    above = [ratio for ratio in true_ratios if ratio > strongest_false]
    return strongest_false, min(above) if above else float("nan")


def strongest_false_local_maximum(population: list[DetectionCase] | None = None) -> tuple[float, str, float]:
    """Return the strongest non-line *local maximum*, selected or not.

    This is deliberately a different statistic from :func:`prominence_gap`: it
    scores every local maximum rather than only the ones the detector would
    place, so it says what a threshold would have to clear if ``max_lines``
    were unbounded.

    Args:
        population: The detection population; measured afresh when omitted.

    Returns:
        Its peak-to-median ratio, the preset it belongs to, and its frequency.
    """
    population = population if population is not None else detection_population()
    worst, worst_preset, worst_frequency = 0.0, "", float("nan")
    for case in population:
        interior = (case.values[1:-1] >= case.values[:-2]) & (case.values[1:-1] > case.values[2:])
        indices = np.flatnonzero(interior) + 1
        median = float(np.median(case.values))
        for index in indices:
            frequency = float(case.frequencies[index])
            ratio = float(case.values[index] / median)
            if ratio > worst and not case.is_tabulated(frequency):
                worst, worst_preset, worst_frequency = ratio, case.psd_file, frequency
    return worst, worst_preset, worst_frequency


def pooled_line_widths(population: list[DetectionCase] | None = None) -> np.ndarray:
    """Return the measurable widths of every tabulated line, pooled.

    Args:
        population: The detection population; measured afresh when omitted.

    Returns:
        The finite full widths at half maximum, in hertz.
    """
    population = population if population is not None else detection_population()
    widths = [
        full_width_half_maximum(case.frequencies, case.values, line) for case in population for line in case.lines
    ]
    return np.array([width for width in widths if np.isfinite(width)])


def worst_band(simulator: ARNoiseSimulator | ARMANoiseSimulator, low_frequency: float) -> float:
    """Return a fitted simulator's worst per-band relative error.

    Args:
        simulator: A fitted AR or ARMA simulator.
        low_frequency: Lower edge of the reported band in hertz.

    Returns:
        The largest of the per-band relative errors.
    """
    frequencies, target = simulator.target_psd_curve
    return max(band_errors(frequencies, target, simulator.model_psd_curve, low_frequency))


def report_tolerances() -> None:
    """Measure the per-band residual populations the fit tolerances bound."""
    print("== 1. per-band fit-residual populations ==")
    print("rule: tolerance = ceil_2dp(1.25 * mean + 5 * standard deviation)")
    for model in ("AR", "ARMA"):
        for psd_file, low_frequency in TOLERANCE_CASES:
            simulator = build(model, psd_file, low_frequency)
            frequencies, target = simulator.target_psd_curve
            fit = band_errors(frequencies, target, simulator.model_psd_curve, low_frequency)
            table_frequencies, table_values = load_spectral_series(psd_file, kind="PSD")
            worst = []
            for group in range(N_SEED_GROUPS):
                strains = [
                    simulator.generate(SEGMENT_SECONDS, SAMPLING_FREQUENCY, ["H1"], seed=1000 * group + index)["H1"]
                    for index in range(N_SEGMENTS)
                ]
                estimate_frequencies, estimate = welch(
                    np.concatenate(strains), fs=SAMPLING_FREQUENCY, nperseg=16384, noverlap=8192
                )
                on_grid = np.interp(estimate_frequencies, table_frequencies, table_values, left=0.0, right=0.0)
                worst.append(max(band_errors(estimate_frequencies, on_grid, estimate, low_frequency)))
            sampled = np.array(worst)
            print(f"  {model:4s} {psd_file:30s} cutoff {low_frequency:4.1f} Hz")
            print("    fit curve : per band " + " ".join(f"{error:.4f}" for error in fit))
            print(f"    fit curve : worst {max(fit):.4f} -> tolerance {TOLERANCE_HEADROOM * max(fit):.4f}")
            print(
                f"    generated : mean {sampled.mean():.4f} sd {sampled.std(ddof=1):.4f} "
                f"max {sampled.max():.4f} over {N_SEED_GROUPS} seed groups -> tolerance "
                f"{TOLERANCE_HEADROOM * sampled.mean() + TOLERANCE_SIGMA * sampled.std(ddof=1):.4f}"
            )

    print("  does the fitted-curve bound discriminate? AR at orders below the tested 256:")
    for psd_file, low_frequency in TOLERANCE_CASES:
        row = []
        for order in (192, 224, 256):
            simulator = ARNoiseSimulator(
                psd_file=psd_file,
                detectors=["H1"],
                sampling_frequency=SAMPLING_FREQUENCY,
                order=order,
                low_frequency_cutoff=low_frequency,
            )
            row.append(f"order {order}: {worst_band(simulator, low_frequency):.4f}")
        print(f"    {psd_file:30s} " + "  ".join(row))

    print("  worst fitted band across all eleven presets, both models, no lines placed:")
    spread = [
        worst_band(build(model, psd_file, low_frequency), low_frequency)
        for model in ("AR", "ARMA")
        for psd_file, low_frequency in PRESETS.items()
    ]
    print(f"    min {min(spread):.4f}  max {max(spread):.4f}  (the tolerances are scoped to the two cases above)")


def report_line_width() -> None:
    """Measure the width of the lines the bundled presets tabulate."""
    print("== 2. tabulated line widths ==")
    population = detection_population()
    tabulated = 0
    for case in population:
        widths = [full_width_half_maximum(case.frequencies, case.values, line) for line in case.lines]
        widths = [width for width in widths if np.isfinite(width)]
        tabulated += len(case.lines)
        median = float(np.median(widths)) if widths else float("nan")
        print(
            f"  {case.psd_file:30s} grid {case.frequencies[1] - case.frequencies[0]:.4f} Hz  "
            f"lines {len(case.lines):3d}  measurable {len(widths):3d}  median {median:.4f} Hz"
        )
    sampled = pooled_line_widths(population)
    print(f"  {tabulated} tabulated lines, {sampled.size} of them with a measurable width; pooled:")
    for quantile in (5, 25, 50, 75, 90, 95):
        print(f"    p{quantile:02d} = {np.percentile(sampled, quantile):.4f} Hz")

    print(f"  what a placed pole costs (O3 Hanford, ARMA {192}/{32}, worst band residual):")
    baseline = ARMANoiseSimulator(
        psd_file=POLE_COST_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        ar_order=192,
        ma_order=32,
        low_frequency_cutoff=20.0,
    )
    print(f"    no placed lines                      {worst_band(baseline, 20.0):.5g}")
    for width in (0.5, DEFAULT_LINE_WIDTH_HZ, 100.0):
        simulator = ARMANoiseSimulator(
            psd_file=POLE_COST_PSD,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            ar_order=192,
            ma_order=32,
            low_frequency_cutoff=20.0,
            line_frequencies=[POLE_COST_LINE_HZ],
            line_widths=[width],
        )
        print(f"    one line at {POLE_COST_LINE_HZ} Hz, width {width:6.2f} Hz  {worst_band(simulator, 20.0):.5g}")
    print("    the lines the detector selects, at the default width, versus the moving-average order:")
    for ma_order in (32, 128, 512, 1024):
        simulator = ARMANoiseSimulator(
            psd_file=POLE_COST_PSD,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            ar_order=192,
            ma_order=ma_order,
            low_frequency_cutoff=20.0,
            detect_lines=True,
        )
        print(f"      ma_order {ma_order:5d}: {worst_band(simulator, 20.0):.5g}")


def report_prominence() -> None:
    """Measure the detector's precision and recall against the tabulated lines."""
    print("== 3. line-detection performance versus the prominence threshold ==")
    print(f"  every number in this section is scored over the top {DEFAULT_MAX_LINES} candidates per preset,")
    print("  which is the population the threshold decides: the detector places no more than that.")
    population = detection_population()

    print(f"  {'threshold':>9s} {'TP':>4s} {'FP':>4s} {'FN':>4s} {'precision':>10s} {'recall':>8s} {'F1':>7s}")
    for threshold in (2.0, 4.0, 6.0, 8.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 30.0, 50.0, 100.0):
        true_positive = false_positive = false_negative = 0
        for case in population:
            detected = _detect_line_frequencies(
                case.frequencies, case.values, max_lines=DEFAULT_MAX_LINES, prominence=threshold
            )
            matched = sum(case.is_tabulated(line) for line in detected)
            true_positive += matched
            false_positive += len(detected) - matched
            false_negative += max(0, min(len(case.lines), DEFAULT_MAX_LINES) - matched)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else float("nan")
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else float("nan")
        harmonic = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(
            f"  {threshold:9.1f} {true_positive:4d} {false_positive:4d} {false_negative:4d} "
            f"{precision:10.3f} {recall:8.3f} {harmonic:7.3f}"
        )

    strongest_false, weakest_true = prominence_gap(population)
    print("  the gap the threshold sits in, over the same selected candidates:")
    print(f"    strongest selected candidate that is NOT a tabulated line : {strongest_false:.4f}")
    print(f"    weakest selected tabulated line standing above it         : {weakest_true:.4f}")
    print(f"    DEFAULT_LINE_PROMINENCE                                   : {DEFAULT_LINE_PROMINENCE:.4f}")
    print(f"    inside the gap: {strongest_false < DEFAULT_LINE_PROMINENCE <= weakest_true}")

    unselected, preset, frequency = strongest_false_local_maximum(population)
    print("  for contrast, a DIFFERENT statistic -- every local maximum, not only the selected ones:")
    print(f"    strongest non-line local maximum: {unselected:.2f} at {frequency:.4f} Hz in {preset}")
    print(f"    it is not among that preset's top {DEFAULT_MAX_LINES}, so no threshold change can place it,")
    print("    and the default does not clear it. The selected-candidate gap above is the anchor.")

    print("  what each preset places, and what it misses, at the anchored threshold:")
    for case in population:
        detected = _detect_line_frequencies(
            case.frequencies, case.values, max_lines=DEFAULT_MAX_LINES, prominence=DEFAULT_LINE_PROMINENCE
        )
        missed = [
            (line, case.peak_to_median(line))
            for line in case.lines
            if not any(abs(line - placed) <= case.match_window for placed in detected)
        ]
        placed = ", ".join(f"{line:.2f}" for line in detected) or "none"
        print(f"    {case.psd_file:30s} places [{placed}]")
        if missed:
            worst_missed = sorted(missed, key=lambda item: -item[1])[:3]
            print(
                "      strongest tabulated lines left to the smooth fit: "
                + ", ".join(f"{line:.2f} Hz (peak-to-median {ratio:.1f})" for line, ratio in worst_missed)
            )

    report_detection_cost()
    report_band_dependence()


def report_detection_cost() -> None:
    """Show what the threshold change buys, in fit residual, on every preset."""
    print("  what the threshold change buys, worst band residual with detect_lines=True:")
    print(
        f"    {'preset':30s} {'no lines':>10s} "
        f"{f'{SUPERSEDED_LINE_PROMINENCE}/{SUPERSEDED_LINE_WIDTH_HZ}':>14s} "
        f"{f'{DEFAULT_LINE_PROMINENCE}/{DEFAULT_LINE_WIDTH_HZ}':>14s}"
    )
    for psd_file, low_frequency in PRESETS.items():
        baseline = build("ARMA", psd_file, low_frequency)
        row = [worst_band(baseline, low_frequency)]
        for threshold, width in (
            (SUPERSEDED_LINE_PROMINENCE, SUPERSEDED_LINE_WIDTH_HZ),
            (DEFAULT_LINE_PROMINENCE, DEFAULT_LINE_WIDTH_HZ),
        ):
            detected = build("ARMA", psd_file, low_frequency, detect_lines=True, line_prominence=threshold)
            placed = [line["frequency"] for line in detected.metadata["autoregressive_moving_average"]["placed_lines"]]
            at_width = (
                build("ARMA", psd_file, low_frequency, line_frequencies=placed, line_widths=[width] * len(placed))
                if placed
                else baseline
            )
            row.append(worst_band(at_width, low_frequency))
        print(f"    {psd_file:30s} {row[0]:10.4f} {row[1]:14.4g} {row[2]:14.4g}")


def report_band_dependence() -> None:
    """Score the same ET-D features over two bands, to show the statistic moves."""
    print("  the statistic is band-dependent, so the threshold is anchored at the full band only:")
    table_frequencies, table_values = load_spectral_series("ET_D_psd", kind="PSD")
    for label, sampling_frequency, high_frequency in (
        ("full band to Nyquist", SAMPLING_FREQUENCY, SAMPLING_FREQUENCY / 2.0),
        ("5-400 Hz at 1024 Hz", 1024.0, 400.0),
    ):
        grid_size = max(2 * (table_frequencies.size - 1), 8 * 192)
        grid = np.fft.rfftfreq(grid_size, d=1.0 / sampling_frequency)
        mask = (grid >= ET_LOW_FREQUENCY) & (grid <= high_frequency)
        frequencies, values = grid[mask], np.interp(grid[mask], table_frequencies, table_values, left=0.0, right=0.0)
        median = float(np.median(values))
        scores = [float(values[int(np.argmin(np.abs(frequencies - target)))] / median) for target in ET_RIPPLE_HZ]
        print(f"    ET-D ripple near 15 and 21 Hz, {label:22s}: {scores[0]:8.2f} {scores[1]:8.2f}")


def report_low_band() -> None:
    """Separate method, table and band definition in the ET-D lowest band."""
    print("== 4. the ET-D lowest-band residual ==")
    edges = np.geomspace(5.0, HIGH_FREQUENCY, N_BANDS + 1)
    low, high = float(edges[0]), float(edges[1])
    print(f"  lowest band = [{low:.3f}, {high:.3f}) Hz")

    table_frequencies, table_values = load_spectral_series("ET_D_psd", kind="PSD")
    dense = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, 24000)
    dense_values = np.interp(dense, table_frequencies, table_values, left=0.0, right=0.0)

    print("  a. model order, at a fixed dense target grid (is it the method?)")
    for order in (64, 128, 192, 256, 384, 512, 768):
        simulator = ARMANoiseSimulator(
            target_psd=dense_values,
            target_frequencies=dense,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            ar_order=order,
            ma_order=32,
            low_frequency_cutoff=low,
        )
        frequencies, target = simulator.target_psd_curve
        errors = band_errors(frequencies, target, simulator.model_psd_curve, low)
        print(f"    ar_order {order:5d}: lowest band {errors[0]:.4f}  whole band worst {max(errors):.4f}")

    print("  b. target grid spacing, at a fixed order (is it the tabulated curve?)")
    near_the_edge = int(np.searchsorted(table_frequencies, low))
    print(
        f"    the ET-D table is sampled at {np.diff(table_frequencies)[near_the_edge]:.4f} Hz near {low:.0f} Hz, "
        f"against the fit's own design grid"
    )
    for points in (3000, 6000, 12000, 24000):
        grid = np.linspace(0.0, SAMPLING_FREQUENCY / 2.0, points)
        simulator = ARMANoiseSimulator(
            target_psd=np.interp(grid, table_frequencies, table_values, left=0.0, right=0.0),
            target_frequencies=grid,
            detectors=["H1"],
            sampling_frequency=SAMPLING_FREQUENCY,
            ar_order=192,
            ma_order=32,
            low_frequency_cutoff=low,
        )
        frequencies, target = simulator.target_psd_curve
        errors = band_errors(frequencies, target, simulator.model_psd_curve, low)
        print(f"    target points {points:6d} ({grid[1] - grid[0]:.4f} Hz): lowest band {errors[0]:.4f}")

    print("  c. band definition, at a fixed fit (is it the band edges?)")
    simulator = build("AR", "ET_D_psd", low)
    frequencies, target = simulator.target_psd_curve
    model = simulator.model_psd_curve
    for count in (4, 8, 16, 32):
        errors = []
        for band_low, band_high in pairwise(np.geomspace(low, HIGH_FREQUENCY, count + 1)):
            selected = (frequencies >= band_low) & (frequencies < band_high)
            if not np.any(selected):
                continue
            reference = float(np.sum(target[selected]))
            if reference <= 0.0:
                continue
            errors.append(abs(float(np.sum(model[selected])) - reference) / reference)
        print(f"    n_bands {count:3d}: lowest band {errors[0]:.4f}  worst band {max(errors):.4f}")

    selected = (frequencies >= low) & (frequencies < high)
    in_band = (frequencies >= low) & (frequencies <= HIGH_FREQUENCY)
    print(
        f"  the lowest band carries {float(np.sum(target[selected])) / float(np.sum(target[in_band])):.4f} "
        f"of the in-band target power; the model puts "
        f"{float(np.sum(model[selected])) / float(np.sum(target[selected])):.4f} of the target power in it"
    )

    native = (table_frequencies >= low) & (table_frequencies < high)
    design_mean = float(np.mean(target[selected]))
    native_mean = float(
        np.trapezoid(table_values[native], table_frequencies[native])
        / (table_frequencies[native][-1] - table_frequencies[native][0])
    )
    print(
        f"  band-mean target on the design grid {design_mean:.4e} against {native_mean:.4e} on the native table: "
        f"ratio {design_mean / native_mean:.4f}"
    )


SECTIONS = {
    "tolerance": report_tolerances,
    "line-width": report_line_width,
    "prominence": report_prominence,
    "low-band": report_low_band,
}


def main() -> None:
    """Run the requested measurement sections."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sections", nargs="*", choices=[*SECTIONS, []], help="Sections to run; default is all of them.")
    arguments = parser.parse_args()
    for name in arguments.sections or SECTIONS:
        SECTIONS[name]()
        print()


if __name__ == "__main__":
    main()
