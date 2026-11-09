"""The ``deepextractor-o3-v1`` glitch population.

Seven Gravity Spy classes, each drawn from DeepExtractor's reconstructions of real O3 LIGO glitches
of that class, at its measured O3 rate and with its measured O3 SNR tail. The waveforms are the
data; the population fixes how often each class occurs and how loud it is.

Every rate and tail is measured from the published Advanced LIGO O3 glitch classifications and the
published observing segments, under one selection:

- **every** trigger the classifier labelled with the class, whatever its confidence, above the
  Omicron threshold of SNR 7.5. A DeepExtractor class is defined by Gravity Spy's label, so its rate
  is the rate of that label. The cut ``et-o3-anchored-v1`` applies, confidence above 0.9, would
  lower the rates by 13 % (Scattered_Light) to 30 % (Koi_Fish);
- **rate**: the class's count over the pooled H1 and L1 observing time;
- **tail**: a power law in optimal SNR above 7.5, truncated at the class's largest observed SNR, with
  the maximum-likelihood exponent of the truncated law.

``scripts/measure_glitch_population_anchors.py --population deepextractor-o3-v1`` reproduces every
number, and ``docs/dev/deepextractor_o3_population.md`` records what each is and how well it describes the
data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gwmock_noise.glitches.deepextractor import DeepExtractorGlitch
from gwmock_noise.glitches.models import LogNormalAmplitudeDistribution
from gwmock_noise.glitches.snr import PowerLawSNRDistribution
from gwmock_noise.populations.population import GlitchClass, GlitchPopulation

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

#: Registered name of the population.
POPULATION_NAME = "deepextractor-o3-v1"

#: DeepExtractor dataset revision the waveforms are drawn from. Pinned, so a population built
#: with the defaults draws from the same reconstructions on every machine.
DATASET_REVISION = "144f56880c6e7aa8def31c537ca843b8c9e5bdda"

#: Default noise curve the target SNRs are defined against: the bundled 10 km Einstein Telescope
#: cryogenic curve. A run colours each interferometer against its own design's curve, so it
#: passes ``psd_file``; a bundled preset name keeps the digest portable, a path does not.
DEFAULT_PSD = "ET_10_full_cryo_psd"

#: Default lower edge of the band the glitches are coloured and calibrated over, in Hz.
DEFAULT_LOW_FREQUENCY_CUTOFF = 5.0

#: SNR threshold every tail is defined above: the selection threshold of the reference data set.
REFERENCE_SNR_THRESHOLD = 7.5

#: Per class: Poisson rate per interferometer in Hz, survival exponent of the truncated SNR tail,
#: and the truncation (the class's largest O3 SNR, rounded up to a hundredth). Measured over all
#: of the class's O3 triggers above SNR 7.5, pooled over 42,952,865 s of H1 and L1 observing time;
#: the bundled ``deepextractor_o3_summary.json`` holds the measurement.
CLASS_PARAMETERS: dict[str, tuple[float, float, float]] = {
    "Blip": (3.5930e-4, 1.2613, 621.23),
    "Fast_Scattering": (1.9866e-3, 3.5052, 9747.92),
    "Koi_Fish": (6.2070e-4, 0.2098, 11841.49),
    "Low_Frequency_Burst": (6.7120e-4, 1.3680, 2781.22),
    "Scattered_Light": (3.2009e-3, 1.4139, 7797.32),
    "Tomte": (1.2482e-3, 1.2492, 128.71),
    "Whistle": (4.1674e-4, 1.7077, 1397.99),
}

#: One sentence per class, part of the registered definition.
CLASS_DESCRIPTIONS: dict[str, str] = {
    "Blip": "Short broadband transients; DeepExtractor reconstructions of O3 Blip glitches.",
    "Fast_Scattering": "Fast-scattering arches; DeepExtractor reconstructions of O3 Fast_Scattering glitches.",
    "Koi_Fish": "Loud broadband transients; DeepExtractor reconstructions of O3 Koi_Fish glitches.",
    "Low_Frequency_Burst": ("Low-frequency bursts; DeepExtractor reconstructions of O3 Low_Frequency_Burst glitches."),
    "Scattered_Light": "Scattered-light arches; DeepExtractor reconstructions of O3 Scattered_Light glitches.",
    "Tomte": "Short low-frequency transients; DeepExtractor reconstructions of O3 Tomte glitches.",
    "Whistle": "Frequency-swept whistles; DeepExtractor reconstructions of O3 Whistle glitches.",
}

#: Published sources the registered quantities are anchored to.
REFERENCES = (
    (
        "Glanzer J et al. 2021, 'Gravity Spy Machine Learning Classifications of LIGO Glitches from "
        "Observing Runs O1, O2, O3a, and O3b', Zenodo, doi:10.5281/zenodo.5649212 -- the per-trigger "
        "classifications and SNRs the rates and tails are measured from."
    ),
    (
        "Glanzer J et al. 2023, 'Data quality up to the third observing run of Advanced LIGO: Gravity "
        "Spy glitch classifications', Class. Quantum Grav. 40, 065004, arXiv:2208.12849 -- the classes."
    ),
    (
        "Gravitational Wave Open Science Center, O3 observing segments (H1_DATA, L1_DATA) -- the "
        "42,952,865 s of pooled observing time the counts are divided by."
    ),
    (
        "DeepExtractor glitch reconstructions, tomdooney/deepextractor-glitch-reconstructions at revision "
        f"{DATASET_REVISION} -- the waveforms, 5,000 per class."
    ),
)

#: Quantities that could not be anchored, or that the measurement describes only in part.
UNANCHORED = (
    (
        "Transferring rates measured on Advanced LIGO to a different instrument, site and band. Each "
        "interferometer glitches like an average O3 LIGO interferometer."
    ),
    (
        "The pooled rate hides large differences between sites: Fast_Scattering and Tomte occur almost "
        "only at Livingston (Hanford/Livingston rate ratios 0.03 and 0.06), Low_Frequency_Burst mostly at "
        "Hanford (3.3)."
    ),
    (
        "One power law above SNR 7.5 does not describe every class's loud end. Above SNR 150 the law "
        "predicts 13 times the observed count for Blip, a fiftieth of it for Fast_Scattering, and 3 to 5 "
        "times it for Scattered_Light and Whistle. The truncation keeps draws within what O3 recorded."
    ),
    (
        "Identifying an Omicron SNR, measured against the O3 LIGO noise over the band Omicron searched, "
        "with an optimal SNR against the noise curve and band configured here."
    ),
    (
        "The seven classes are the Gravity Spy classes DeepExtractor reconstructs. The other glitch "
        "classes, 23 % of the O3 glitch rate, are absent."
    ),
)

#: What the population is for, part of the registered definition.
DESCRIPTION = (
    "Seven Gravity Spy glitch classes as DeepExtractor reconstructions of O3 LIGO glitches, each at its "
    "measured O3 rate per interferometer and with its measured O3 SNR tail above 7.5, truncated at its "
    "largest observed SNR. Every interferometer glitches independently."
)


def deepextractor_o3_population(
    *,
    psd_file: str | Path = DEFAULT_PSD,
    detectors: str | Sequence[str] | None = None,
    low_frequency_cutoff: float = DEFAULT_LOW_FREQUENCY_CUTOFF,
    revision: str = DATASET_REVISION,
) -> GlitchPopulation:
    """Build the ``deepextractor-o3-v1`` population.

    Called with no arguments this returns the registered population, whose digest a campaign pins.
    The arguments are a run's own settings: the noise curve and band each glitch is coloured and
    calibrated against, the interferometers it applies to, and the dataset revision. Changing any
    of them changes the digest. The measured rates and tails are not arguments.

    Args:
        psd_file: Noise curve the target SNRs are defined against.
        detectors: One interferometer name or a list of them, the classes' scope; ``None`` for every
            interferometer in the run.
        low_frequency_cutoff: Lower edge of the colouring and calibration band, in Hz.
        revision: DeepExtractor dataset revision.

    Returns:
        The population, one class per Gravity Spy class, each its own Poisson process per
        interferometer.
    """
    classes = []
    for name, (rate, alpha, maximum) in CLASS_PARAMETERS.items():
        model = DeepExtractorGlitch(
            rate=rate,
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
            psd_file=psd_file,
            snr=PowerLawSNRDistribution(minimum=REFERENCE_SNR_THRESHOLD, alpha=alpha, maximum=maximum),
            glitch_classes=[name],
            low_frequency_cutoff=low_frequency_cutoff,
            revision=revision,
            detectors=detectors,
        )
        classes.append(
            GlitchClass(
                name=name,
                morphology=f"deepextractor-{name.lower().replace('_', '-')}",
                description=CLASS_DESCRIPTIONS[name],
                model=model,
            )
        )
    return GlitchPopulation(
        name=POPULATION_NAME,
        description=DESCRIPTION,
        classes=tuple(classes),
        references=REFERENCES,
        unanchored=UNANCHORED,
    )
