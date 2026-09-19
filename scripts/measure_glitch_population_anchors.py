"""Re-measure the quantities the ``et-o3-anchored-v1`` glitch population is anchored to.

Every rate, amplitude-law parameter and frequency edge in
:mod:`gwmock_noise.populations.et_o3_anchored` is measured over two published data
products rather than chosen by hand:

1. the Gravity Spy machine-learning classifications of LIGO glitches from O1-O3b
   (Zenodo, doi:10.5281/zenodo.5649212) -- one row per Omicron trigger, carrying its
   SNR, duration, peak frequency and classified morphology; and
2. the Gravitational Wave Open Science Center's O3 observing segments, which supply the
   livetime the counts are divided by.

This script downloads both (about 560 MB, cached), measures everything, prints it, and
optionally rewrites the small summary the package bundles at
``src/gwmock_noise/data/glitch_population/o3_reference_summary.json``. That summary is
what ``docs/dev/glitch_population.md`` quotes and what ``tests/test_glitch_population.py``
asserts the registered constants against, so the page, this printout and the assertions
cannot drift apart.

It is a measurement harness, not a test: it needs the network, it takes a few minutes on a
cold cache, and it changes nothing unless asked to.

Usage::

    uv run python scripts/measure_glitch_population_anchors.py
    uv run python scripts/measure_glitch_population_anchors.py --write-summary
    uv run python scripts/measure_glitch_population_anchors.py --data-dir /scratch/gspy
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from scipy.optimize import brentq

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the bundled summary lives. Read by the tests, quoted by the anchor page.
SUMMARY_PATH = REPO_ROOT / "src" / "gwmock_noise" / "data" / "glitch_population" / "o3_reference_summary.json"

#: Zenodo record holding the Gravity Spy classifications.
ZENODO_RECORD = "5649212"

#: The four O3 classification files, keyed by ``(interferometer, run)``.
CLASSIFICATION_FILES = {
    ("H1", "O3a"): "H1_O3a.csv",
    ("H1", "O3b"): "H1_O3b.csv",
    ("L1", "O3a"): "L1_O3a.csv",
    ("L1", "O3b"): "L1_O3b.csv",
}

#: GWOSC data-release tag and GPS span of each O3 half, used to fetch observing segments.
GWOSC_RUNS = {
    "O3a": ("O3a_16KHZ_R1", 1238166018, 15811200),
    "O3b": ("O3b_16KHZ_R1", 1256655618, 12708000),
}

#: Gravity Spy classes the two registered glitch classes are measured from.
MEASURED_CLASSES = ("Blip", "Scattered_Light")

#: Trigger selection. ``SNR > 7.5`` is the Omicron threshold the Gravity Spy set is built
#: on, and 0.9 is the confidence the reference paper calls its fiducial threshold.
SNR_MINIMUM = 7.5
CONFIDENCE_MINIMUM = 0.9

#: Percentiles reported for every measured column.
PERCENTILES = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)

#: Thresholds the Hill index is reported at, so its threshold dependence -- the statement
#: that neither class is a pure power law from 7.5 upward -- is visible rather than hidden
#: behind the one value the population registers.
HILL_THRESHOLDS = (7.5, 10.0, 15.0, 20.0, 30.0)

#: The upper SNR threshold Cabero et al. imposed on their blip search, reported here so the
#: cost of adopting it as a truncation instead of the observed maximum is a number rather
#: than a guess. It is a search selection -- imposed to keep the sample morphologically
#: pure -- and not a statement that the class never exceeds it.
PUBLISHED_BLIP_SNR_CEILING = 150.0


class Triggers(NamedTuple):
    """The measured columns of one class in one interferometer."""

    snr: np.ndarray
    duration: np.ndarray
    peak_frequency: np.ndarray
    central_frequency: np.ndarray
    bandwidth: np.ndarray

    @classmethod
    def concatenate(cls, parts: list[Triggers]) -> Triggers:
        """Pool several interferometers' or runs' triggers into one sample."""
        return cls(*(np.concatenate([getattr(part, name) for part in parts]) for name in cls._fields))


def hill_index(values: np.ndarray, threshold: float) -> tuple[float, float, int]:
    """Estimate the survival exponent of a power-law tail above ``threshold``.

    The Hill estimator, ``n / sum(log(x_i / threshold))`` over the ``n`` samples at or above
    the threshold. It is the maximum-likelihood estimate of the exponent of
    ``S(x) = (x / threshold) ** -alpha``, which is the convention
    :class:`~gwmock_noise.glitches.snr.PowerLawSNRDistribution` takes, so the number goes
    into a configuration unconverted.

    Args:
        values: The sample.
        threshold: The tail threshold.

    Returns:
        The index, its asymptotic standard error ``alpha / sqrt(n)``, and ``n``.

    Raises:
        ValueError: If no sample reaches the threshold.
    """
    tail = values[values >= threshold]
    if tail.size == 0:
        raise ValueError(f"no samples at or above {threshold}.")
    alpha = float(tail.size / np.sum(np.log(tail / threshold)))
    return alpha, float(alpha / np.sqrt(tail.size)), int(tail.size)


def truncated_pareto_index(values: np.ndarray, minimum: float, maximum: float) -> float:
    """Maximum-likelihood survival exponent of a power law *truncated* at ``maximum``.

    **This, and not the Hill index, is what a registered amplitude law is set from**, and
    the difference is not academic. The Hill estimator is the maximum-likelihood exponent of
    an *untruncated* Pareto tail. A registered class truncates its draws at the largest SNR
    the reference sample contains, and a truncated law drawn with the Hill index produces a
    sample whose own Hill index is steeper than the one it was set from -- 1.3763 becomes
    1.4149 for the short class, a 2.8 % error in the quantity the anchor exists to
    reproduce. Estimating the exponent under the model that will actually be sampled removes
    that: a truncated law drawn with this index reproduces the measured Hill index, which
    :func:`expected_hill_of_truncated_draws` states in closed form and the tests assert.

    Solved from the score equation ``1/alpha - U exp(-alpha U) / (1 - exp(-alpha U)) =
    mean(log(x / minimum))`` with ``U = log(maximum / minimum)``, whose left-hand side is
    strictly decreasing in ``alpha``.

    Args:
        values: The sample.
        minimum: The threshold the law is defined above.
        maximum: The truncation, normally the sample maximum, which is its own
            maximum-likelihood estimate.

    Returns:
        The exponent.

    Raises:
        ValueError: If no sample lies in ``[minimum, maximum]``.
    """
    sample = np.asarray(values, dtype=float)
    sample = sample[(sample >= minimum) & (sample <= maximum)]
    if sample.size == 0:
        raise ValueError(f"no samples in [{minimum}, {maximum}].")
    mean_log = float(np.mean(np.log(sample / minimum)))
    span = float(np.log(maximum / minimum))

    def score(alpha: float) -> float:
        truncated = np.exp(-alpha * span)
        return (1.0 / alpha) - (span * truncated / (1.0 - truncated)) - mean_log

    return float(brentq(score, 1e-6, 1e3, xtol=1e-12))


def expected_hill_of_truncated_draws(minimum: float, alpha: float, maximum: float) -> float:
    """The Hill index a truncated power law's own draws have, in closed form.

    ``E[log(X / minimum)]`` under the truncated law, inverted. Reported beside the measured
    Hill index so that "the registered law reproduces the measured population" is a
    comparison of two numbers rather than an assertion.

    Args:
        minimum: The threshold.
        alpha: The registered survival exponent.
        maximum: The truncation.

    Returns:
        The Hill index the law's draws converge to.
    """
    span = np.log(maximum / minimum)
    truncated = np.exp(-alpha * span)
    mean_log = ((1.0 / alpha) - (span + 1.0 / alpha) * truncated) / (1.0 - truncated)
    return float(1.0 / mean_log)


def power_law_median(minimum: float, alpha: float, maximum: float | None = None) -> float:
    """Median of a power law with threshold ``minimum``, index ``alpha`` and a truncation.

    Quoted beside the measured median so the gap between them is on the page. A power law
    fitted to a tail is not a fit to the bulk, and the gap is the size of that statement.

    Args:
        minimum: The threshold.
        alpha: The survival exponent.
        maximum: The truncation, or ``None`` for an untruncated law.

    Returns:
        The median of the modelled distribution.
    """
    if maximum is None:
        return float(minimum * 2.0 ** (1.0 / alpha))
    truncated = (maximum / minimum) ** (-alpha)
    return float(minimum * (0.5 * (1.0 + truncated)) ** (-1.0 / alpha))


def round_outward(value: float, significant: int = 2, *, upward: bool) -> float:
    """Round a percentile outward to ``significant`` significant figures.

    The rule the registered frequency edges are set by, written once so the page, the
    printout and the assertion apply the same one.

    Args:
        value: The percentile to round.
        significant: How many significant figures to keep.
        upward: Whether to round away from zero (an upper edge) or toward it (a lower one).

    Returns:
        The rounded edge.
    """
    if value <= 0.0:
        raise ValueError("round_outward expects a positive value.")
    scale = 10.0 ** (significant - 1 - int(np.floor(np.log10(value))))
    rounded = np.ceil(value * scale) if upward else np.floor(value * scale)
    return float(rounded / scale)


def _download(url: str, destination: Path) -> Path:
    """Fetch ``url`` to ``destination`` unless it is already there."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}", file=sys.stderr)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url, timeout=1800) as response, partial.open("wb") as handle:  # noqa: S310
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.rename(destination)
    return destination


def observing_livetime(data_dir: Path) -> dict[str, float]:
    """Return each interferometer's total O3 observing time, in seconds.

    Summed over the GWOSC observing segments rather than taken as a duty cycle times a
    calendar span, because the count this divides is a count over *analysed* data. The
    printout reports the implied duty cycle too, which is the number the published detector
    characterisation quotes and therefore the cross-check on this one.

    Args:
        data_dir: Cache directory for the fetched segment lists.

    Returns:
        The pooled O3a+O3b observing time per interferometer.
    """
    livetime: dict[str, float] = {}
    for detector in ("H1", "L1"):
        total = 0.0
        for run, (tag, start, span) in GWOSC_RUNS.items():
            url = f"https://gwosc.org/timeline/segments/json/{tag}/{detector}_DATA/{start}/{span}/"
            path = _download(url, data_dir / f"segments_{detector}_{run}.json")
            segments = json.loads(path.read_text())["segments"]
            seconds = float(sum(stop - begin for begin, stop in segments))
            print(f"  {detector} {run}: {len(segments)} segments, {seconds:.0f} s, duty {seconds / span:.4f}")
            total += seconds
        livetime[detector] = total
    return livetime


def read_triggers(data_dir: Path) -> dict[tuple[str, str], Triggers]:
    """Read the selected triggers of every measured class, per interferometer.

    Args:
        data_dir: Cache directory for the classification files.

    Returns:
        The triggers keyed by ``(interferometer, Gravity Spy class)``, pooled over O3a and
        O3b.
    """
    collected: dict[tuple[str, str], list[list[float]]] = {
        (detector, name): [] for detector in ("H1", "L1") for name in MEASURED_CLASSES
    }
    for (detector, run), filename in sorted(CLASSIFICATION_FILES.items()):
        url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{filename}/content"
        path = _download(url, data_dir / filename)
        print(f"  reading {detector} {run} from {path.name}", file=sys.stderr)
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                label = row["ml_label"]
                if label not in MEASURED_CLASSES:
                    continue
                if float(row["ml_confidence"]) <= CONFIDENCE_MINIMUM or float(row["snr"]) <= SNR_MINIMUM:
                    continue
                collected[(detector, label)].append(
                    [
                        float(row["snr"]),
                        float(row["duration"]),
                        float(row["peak_frequency"]),
                        float(row["central_freq"]),
                        float(row["bandwidth"]),
                    ]
                )
    return {key: Triggers(*np.asarray(values, dtype=float).T) for key, values in collected.items()}


def _percentiles(values: np.ndarray) -> dict[str, float]:
    """Return the reported percentiles of one column, keyed by percentile."""
    return {
        f"p{percentile:g}": float(value)
        for percentile, value in zip(PERCENTILES, np.percentile(values, PERCENTILES), strict=True)
    }


def summarize(triggers: dict[tuple[str, str], Triggers], livetime: dict[str, float]) -> dict[str, Any]:
    """Build the bundled summary from the raw triggers and the observing time.

    Args:
        triggers: The selected triggers per ``(interferometer, class)``.
        livetime: Observing time per interferometer, in seconds.

    Returns:
        The summary, in the shape the bundled JSON has.
    """
    pooled_livetime = float(sum(livetime.values()))
    summary: dict[str, Any] = {
        "description": (
            "Measured reference statistics for the et-o3-anchored-v1 glitch population. "
            "Regenerated by scripts/measure_glitch_population_anchors.py; quoted by "
            "docs/dev/glitch_population.md; asserted against by tests/test_glitch_population.py."
        ),
        "sources": {
            "classifications": "doi:10.5281/zenodo.5649212 (Gravity Spy machine-learning classifications, O1-O3b)",
            "livetime": "Gravitational Wave Open Science Center O3 observing segments (H1_DATA, L1_DATA)",
        },
        "selection": {"snr_minimum": SNR_MINIMUM, "confidence_minimum": CONFIDENCE_MINIMUM},
        "livetime_seconds": {**{key: float(value) for key, value in livetime.items()}, "pooled": pooled_livetime},
        "classes": {},
    }
    for name in MEASURED_CLASSES:
        pooled = Triggers.concatenate([triggers[(detector, name)] for detector in ("H1", "L1")])
        alpha, stderr, _ = hill_index(pooled.snr, SNR_MINIMUM)
        maximum = float(pooled.snr.max())
        truncated = truncated_pareto_index(pooled.snr, SNR_MINIMUM, maximum)
        summary["classes"][name] = {
            "counts": {
                **{detector: int(triggers[(detector, name)].snr.size) for detector in ("H1", "L1")},
                "pooled": int(pooled.snr.size),
            },
            "rate_hz": {
                **{
                    detector: float(triggers[(detector, name)].snr.size / livetime[detector])
                    for detector in ("H1", "L1")
                },
                "pooled": float(pooled.snr.size / pooled_livetime),
            },
            "snr": {
                "hill_alpha": alpha,
                "hill_alpha_stderr": stderr,
                "truncated_pareto_alpha": truncated,
                "hill_of_truncated_draws": expected_hill_of_truncated_draws(SNR_MINIMUM, truncated, maximum),
                "hill_by_threshold": {
                    f"{threshold:g}": hill_index(pooled.snr, threshold)[0] for threshold in HILL_THRESHOLDS
                },
                "median": float(np.median(pooled.snr)),
                "maximum": maximum,
                "count_above_published_ceiling": int((pooled.snr > PUBLISHED_BLIP_SNR_CEILING).sum()),
                "percentiles": _percentiles(pooled.snr),
            },
            "duration_seconds": {
                "median": float(np.median(pooled.duration)),
                "percentiles": _percentiles(pooled.duration),
            },
            "peak_frequency_hz": {
                "median": float(np.median(pooled.peak_frequency)),
                "percentiles": _percentiles(pooled.peak_frequency),
            },
            "central_frequency_hz": {"median": float(np.median(pooled.central_frequency))},
            "bandwidth_hz": {"median": float(np.median(pooled.bandwidth))},
        }
    return summary


def report(summary: dict[str, Any]) -> None:
    """Print every number the anchor page quotes."""
    livetime = summary["livetime_seconds"]
    print("\n== Observing time (GWOSC O3 segments) ==")
    for key, value in livetime.items():
        print(f"  {key:>7s}: {value:12.0f} s = {value / 86400.0:8.2f} d")

    for name, block in summary["classes"].items():
        print(f"\n== {name} ==")
        counts, rates = block["counts"], block["rate_hz"]
        for key in ("H1", "L1", "pooled"):
            print(f"  {key:>7s}: n={counts[key]:7d} rate={rates[key]:.6e} Hz = {rates[key] * 3600.0:8.4f}/hr")
        print(f"  site-to-site rate ratio: {rates['H1'] / rates['L1']:.3f}")
        snr = block["snr"]
        print(f"  SNR  Hill(>{SNR_MINIMUM:g}) = {snr['hill_alpha']:.4f} +/- {snr['hill_alpha_stderr']:.4f}")
        print(
            f"  SNR  truncated-Pareto MLE (registered) = {snr['truncated_pareto_alpha']:.4f}; "
            f"its draws' Hill index = {snr['hill_of_truncated_draws']:.4f} against the measured "
            f"{snr['hill_alpha']:.4f}"
        )
        print(
            f"  SNR  the Hill index used as the exponent instead would give draws of Hill "
            f"{expected_hill_of_truncated_draws(SNR_MINIMUM, snr['hill_alpha'], snr['maximum']):.4f}"
        )
        print("  SNR  Hill by threshold: " + "  ".join(f"{k}->{v:.4f}" for k, v in snr["hill_by_threshold"].items()))
        print(
            f"  SNR  median measured {snr['median']:.3f} vs registered law "
            f"{power_law_median(SNR_MINIMUM, snr['truncated_pareto_alpha'], snr['maximum']):.3f}"
        )
        print(
            f"  SNR  maximum {snr['maximum']:.3f}; events above the published ceiling "
            f"{PUBLISHED_BLIP_SNR_CEILING:g}: {snr['count_above_published_ceiling']}"
        )
        print(f"  SNR  percentiles {snr['percentiles']}")
        print(
            f"  duration     median {block['duration_seconds']['median']:.4f} s  {block['duration_seconds']['percentiles']}"
        )
        peak = block["peak_frequency_hz"]
        print(f"  peak freq    median {peak['median']:.3f} Hz  {peak['percentiles']}")
        print(
            f"  peak freq    outward-rounded p1/p99 edges: "
            f"{round_outward(peak['percentiles']['p1'], upward=False):g} Hz to "
            f"{round_outward(peak['percentiles']['p99'], upward=True):g} Hz"
        )
        print(
            f"  Omicron central freq median {block['central_frequency_hz']['median']:.1f} Hz, "
            f"bandwidth median {block['bandwidth_hz']['median']:.1f} Hz"
        )


def main(argv: list[str] | None = None) -> int:
    """Measure the anchors, print them, and optionally rewrite the bundled summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache" / "gwmock-noise" / "glitch-anchors",
        help="Where the downloaded classification and segment files are cached.",
    )
    parser.add_argument(
        "--write-summary",
        action="store_true",
        help=f"Rewrite {SUMMARY_PATH.relative_to(REPO_ROOT)} with the measured summary.",
    )
    arguments = parser.parse_args(argv)

    print("== Fetching inputs ==", file=sys.stderr)
    livetime = observing_livetime(arguments.data_dir)
    triggers = read_triggers(arguments.data_dir)
    summary = summarize(triggers, livetime)
    report(summary)

    if arguments.write_summary:
        SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {SUMMARY_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
