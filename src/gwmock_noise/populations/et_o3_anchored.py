"""The ``et-o3-anchored-v1`` glitch population.

Two transient classes for a network whose interferometers share a site: a **short,
broadband, single-channel** class and a **longer, low-frequency, network-coherent** class.
Every number that could be measured is measured, from the published Advanced LIGO O3 glitch
classifications and the published observing segments; every number that could not is listed
in :data:`UNANCHORED` and in ``docs/dev/glitch_population.md``, which records what each
measurement was and how it was cross-checked.

Why these two classes and not others: the pair spans the two ways a transient population
hurts a coincidence- or coherence-based search. The short class is loud, broadband and
looks like a high-mass compact-binary merger, and it fires in one channel at a time. The
longer class is low-frequency, lasts seconds, and -- on co-located interferometers -- is
expected to fire in all of them at once, which is precisely the case a coincidence veto and
a null stream are built on the assumption of never seeing.

``docs/dev/glitch_population.md`` is the anchor record and
``scripts/measure_glitch_population_anchors.py`` reproduces every number on it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gwmock_noise.glitches.models import (
    BlipGlitch,
    LogNormalAmplitudeDistribution,
    NetworkCoherence,
    ScatteredLightGlitch,
)
from gwmock_noise.glitches.snr import PowerLawSNRDistribution
from gwmock_noise.populations.population import GlitchClass, GlitchPopulation

if TYPE_CHECKING:
    from pathlib import Path

#: Registered name. The version suffix is part of the name rather than a separate field: a
#: campaign pins a name and a digest, and a name that silently meant something different
#: last month is worse than no name.
POPULATION_NAME = "et-o3-anchored-v1"

#: Default noise curve the target SNRs are defined against. The bundled 10 km Einstein
#: Telescope cryogenic curve. An SNR is meaningless without one, and naming a *bundled
#: preset* rather than a path is what keeps the population's digest reproducible on another
#: machine.
DEFAULT_PSD = "ET_10_full_cryo_psd"

#: SNR threshold both classes' amplitude laws are defined above. It is the selection
#: threshold of the reference data set -- Gravity Spy classifies Omicron triggers with
#: SNR > 7.5 -- so it is where the measured sample starts, and a power law fitted above it
#: says nothing about what lies below.
REFERENCE_SNR_THRESHOLD = 7.5

# ---------------------------------------------------------------------------------------
# Short single-channel class. Measured over the Blip classifications.
# ---------------------------------------------------------------------------------------

#: Poisson rate per interferometer, in Hz. 12,196 Blip classifications with SNR > 7.5 and
#: classifier confidence > 0.9 over 42,952,865 s of pooled LIGO Hanford and Livingston O3
#: observing time: 1.022 per hour. The two sites differ by a factor 1.6 (1.267/hr at
#: Hanford, 0.784/hr at Livingston), which is the honest spread on this number.
SHORT_RATE_HZ = 2.8394e-4

#: Full width at half maximum of the Gaussian envelope, in seconds. Cabero et al. define a
#: blip as "a very short duration transient, O(10) ms, with a large frequency bandwidth,
#: O(100) Hz".
SHORT_WIDTH_SECONDS = 0.01

#: Lower edge of the class's frequency support, in Hz: the 1st percentile of the measured
#: Omicron peak frequency, rounded outward to two significant figures.
SHORT_LOW_FREQUENCY_HZ = 75.0

#: Upper edge of the class's frequency support, in Hz: the 99th percentile of the measured
#: Omicron peak frequency, rounded outward to two significant figures.
SHORT_HIGH_FREQUENCY_HZ = 780.0

#: Survival exponent of the target-SNR power law: the maximum-likelihood exponent of a
#: power law *truncated* at :data:`SHORT_SNR_MAXIMUM`, measured over the whole reference
#: sample above :data:`REFERENCE_SNR_THRESHOLD`. It is deliberately **not** the Hill index
#: of that sample, which is 1.3763 +/- 0.0125: the Hill estimator is the exponent of an
#: untruncated tail, and a truncated law drawn with it produces a sample whose own Hill
#: index is 1.4149 -- 2.8 % steeper than the population it was meant to reproduce. Drawn
#: with this exponent instead, the law's sample has a Hill index of 1.3763, the measured
#: value. See ``docs/dev/glitch_population.md``.
SHORT_SNR_ALPHA = 1.3333

#: Loud-tail truncation, in SNR. The largest Blip SNR in the reference data set. It is a
#: property of that *sample*, not a physical cap -- a longer observation would extend it --
#: and it is required rather than cosmetic, because an untruncated index below two has
#: infinite variance and a long run draws SNRs no instrument could produce.
SHORT_SNR_MAXIMUM = 344.38

# ---------------------------------------------------------------------------------------
# Longer network-coherent class. Measured over the Scattered Light classifications.
# ---------------------------------------------------------------------------------------

#: Poisson rate of the shared network process, in Hz. Equal to the measured per-channel
#: rate because every interferometer takes every event at the registered participation of
#: one: 119,162 Scattered Light classifications with SNR > 7.5 and confidence > 0.9 over the
#: same 42,952,865 s, i.e. 9.99 per hour. Hanford and Livingston differ by a factor 1.4
#: (11.59/hr and 8.43/hr).
COHERENT_RATE_HZ = 2.7743e-3

#: Length of one arch, in seconds: the median measured Omicron duration of the class.
COHERENT_DURATION_SECONDS = 1.75

#: Frequency the arch reaches at its apex, in Hz: the median measured Omicron peak
#: frequency of the class.
COHERENT_PEAK_FREQUENCY_HZ = 26.0

#: Lower edge of the class's frequency support, in Hz. Soni et al.: slow scattering "creates
#: 'scatter shelves' in the frequency band 10 Hz to 120 Hz".
COHERENT_LOW_FREQUENCY_HZ = 10.0

#: Upper edge of the class's frequency support, in Hz. The other end of the same shelf band.
COHERENT_HIGH_FREQUENCY_HZ = 120.0

#: Survival exponent of the target-SNR power law, on the same truncated-maximum-likelihood
#: footing as :data:`SHORT_SNR_ALPHA`. The measured Hill index is 1.4363 +/- 0.0042, and a
#: truncated law drawn with this exponent reproduces exactly that.
COHERENT_SNR_ALPHA = 1.4184

#: Loud-tail truncation, in SNR: the largest Scattered Light SNR in the reference data set,
#: with the same caveat as the short class's.
COHERENT_SNR_MAXIMUM = 601.24

#: Probability that any one interferometer receives a given network event. **Not anchored.**
#: One is the strongest coherence the model can express and the hardest case for a
#: coincidence veto or a null stream, and it is the only value that does not add a second
#: unmeasured number: at one, the network rate and the per-channel rate coincide, so the
#: class's rate stays exactly the measured one. Sweep it to weaken the assumption.
COHERENT_PARTICIPATION_PROBABILITY = 1.0

#: Spread of the per-interferometer amplitude ratio. **Not anchored.** Zero injects the
#: identical strain into every participating interferometer, which is what a common-mode
#: environmental coupling into co-located instruments would do if the coupling were equal;
#: it is not a measurement that it is equal.
COHERENT_AMPLITUDE_RATIO_STD = 0.0

#: Published sources the registered quantities are anchored to.
REFERENCES = (
    (
        "Glanzer J et al. 2023, 'Data quality up to the third observing run of Advanced LIGO: "
        "Gravity Spy glitch classifications', Class. Quantum Grav. 40, 065004, "
        "arXiv:2208.12849 -- the glitch classes, their morphologies and their relative "
        "prevalence in O3."
    ),
    (
        "Glanzer J et al. 2021, 'Gravity Spy Machine Learning Classifications of LIGO Glitches "
        "from Observing Runs O1, O2, O3a, and O3b', Zenodo, doi:10.5281/zenodo.5649212 -- the "
        "per-trigger classifications, SNRs, durations and peak frequencies the rates and "
        "amplitude laws here are measured from."
    ),
    (
        "Cabero M et al. 2019, 'Blip glitches in Advanced LIGO data', Class. Quantum Grav. 36, "
        "155010, arXiv:1901.05093 -- the short class's morphology and duration, its independent "
        "occurrence between separated interferometers, and an independent measurement of its "
        "rate."
    ),
    (
        "Soni S et al. 2021, 'Reducing scattered light in LIGO's third observing run', Class. "
        "Quantum Grav. 38, 025016, arXiv:2007.14876 -- the longer class's arch morphology and "
        "its 10-120 Hz frequency support."
    ),
    (
        "Davis D et al. 2021, 'LIGO detector characterization in the second and third observing "
        "runs', Class. Quantum Grav. 38, 135014, arXiv:2101.11673 -- the O3 duty cycles the "
        "observing time used here is cross-checked against."
    ),
    (
        "Gravitational Wave Open Science Center, O3 observing segments (H1_DATA, L1_DATA) -- "
        "the 42,952,865 s of pooled observing time the rates are divided by."
    ),
)

#: Quantities that could not be anchored to a published measurement. Each is a statement a
#: consumer of this population is making on its own authority, so each is carried with the
#: population rather than left in a commit message.
UNANCHORED = (
    (
        "That any glitch class is coherent across a network at all. The only measurement of the "
        "question says the opposite for separated sites: over O1 and O2 the LIGO blip "
        "population produced no coincidences within the +/-15 ms window an astrophysical signal "
        "can occupy. Applying a coherent class to co-located interferometers is an "
        "extrapolation from the shared-infrastructure argument, not from data."
    ),
    (
        "The participation probability of 1.0, and therefore the network event rate, which is "
        "the per-channel rate divided by it. No measurement of transient coincidence between "
        "co-located interferometers was found to set it."
    ),
    (
        "The zero spread of the per-interferometer amplitude ratio, i.e. that a common-mode "
        "transient couples equally into every interferometer of a site."
    ),
    (
        "The zero inter-interferometer arrival-time offset. Co-located instruments make it "
        "small for an environmental source, but no measurement bounds it."
    ),
    (
        "Transferring a rate measured on Advanced LIGO to a different instrument, site and "
        "band. The rates here are what those detectors did, not a forecast of what another one "
        "will do."
    ),
    (
        "Identifying a target SNR measured against the O3 Advanced LIGO noise curves over the "
        "band Omicron searched with an optimal SNR against the noise curve configured here over "
        "the band configured here. The two are different inner products; the sampler reproduces "
        "whatever distribution it is given and cannot tell whether that identification is the "
        "one intended."
    ),
    (
        "The homogeneous Poisson arrival process. The measured rates of both classes are "
        "strongly non-stationary -- the longer class tracks microseismic ground motion and the "
        "published hourly rate varies by more than an order of magnitude across a run -- so a "
        "constant rate reproduces the run average and not the clustering."
    ),
    (
        "The truncation of each class's loud tail at the largest SNR observed in the reference "
        "data set. That is the largest value a finite observation happened to contain, and it "
        "grows with observing time; it is not an instrumental ceiling."
    ),
)


def et_o3_anchored_population(
    *,
    psd_file: str | Path = DEFAULT_PSD,
    participation_probability: float = COHERENT_PARTICIPATION_PROBABILITY,
    amplitude_ratio_std: float = COHERENT_AMPLITUDE_RATIO_STD,
    coherent_rate_hz: float = COHERENT_RATE_HZ,
    short_rate_hz: float = SHORT_RATE_HZ,
) -> GlitchPopulation:
    """Build the ``et-o3-anchored-v1`` population.

    Called with no arguments this returns the registered population, and its digest is the
    one a campaign pins. The arguments exist so the assumptions that could not be anchored
    can be varied deliberately -- a sweep over the participation probability is the obvious
    one -- and any such variation produces a *different digest*, which is the point: a run
    cannot claim to have used the registered population while having weakened it.

    Note that ``short_rate_hz`` and ``coherent_rate_hz`` are the rates of two different
    kinds of process. The short class runs one Poisson process **per interferometer**, so
    its rate is what each of them sees. The longer class runs **one** process for the whole
    network, so its rate is the rate of shared events; each interferometer sees that times
    the participation probability.

    Args:
        psd_file: Noise curve the target SNRs are calibrated against. A bundled preset name
            keeps the digest portable; an absolute path does not.
        participation_probability: Probability that one interferometer receives a given
            event of the coherent class. Unanchored; see :data:`UNANCHORED`.
        amplitude_ratio_std: Spread of the per-interferometer amplitude ratio of the
            coherent class. Unanchored.
        coherent_rate_hz: Network event rate of the coherent class, in Hz.
        short_rate_hz: Per-interferometer event rate of the short class, in Hz.

    Returns:
        The population, with both classes scoped to every interferometer in the run.
    """
    short = BlipGlitch(
        rate=short_rate_hz,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=SHORT_WIDTH_SECONDS,
        psd_file=psd_file,
        snr=PowerLawSNRDistribution(
            minimum=REFERENCE_SNR_THRESHOLD,
            alpha=SHORT_SNR_ALPHA,
            maximum=SHORT_SNR_MAXIMUM,
        ),
        low_frequency_cutoff=SHORT_LOW_FREQUENCY_HZ,
        high_frequency_cutoff=SHORT_HIGH_FREQUENCY_HZ,
    )
    coherent = ScatteredLightGlitch(
        rate=coherent_rate_hz,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=COHERENT_DURATION_SECONDS,
        peak_frequency=COHERENT_PEAK_FREQUENCY_HZ,
        psd_file=psd_file,
        snr=PowerLawSNRDistribution(
            minimum=REFERENCE_SNR_THRESHOLD,
            alpha=COHERENT_SNR_ALPHA,
            maximum=COHERENT_SNR_MAXIMUM,
        ),
        low_frequency_cutoff=COHERENT_LOW_FREQUENCY_HZ,
        high_frequency_cutoff=COHERENT_HIGH_FREQUENCY_HZ,
        network=NetworkCoherence(
            participation_probability=participation_probability,
            amplitude_ratio_std=amplitude_ratio_std,
        ),
    )
    return GlitchPopulation(
        name=POPULATION_NAME,
        description=(
            "Two transient classes for a network of co-located interferometers, anchored to the "
            "Advanced LIGO O3 glitch phenomenology: a short, broadband, single-channel class "
            "drawn from the measured Blip population, and a longer, low-frequency, "
            "network-coherent class drawn from the measured Scattered Light population. Rates "
            "are per unit observing time; amplitudes are truncated power laws in optimal SNR "
            "against the configured noise curve. The coherence of the second class is the "
            "population's one deliberate extrapolation and is listed among its unanchored "
            "quantities."
        ),
        classes=(
            GlitchClass(
                name="short_single_channel",
                morphology="gaussian-windowed-broadband-burst",
                description=(
                    "A sub-second broadband burst with no time-frequency structure, occurring "
                    "independently in each interferometer. Drawn from the measured Advanced LIGO "
                    "O3 Blip population: O(10) ms long, O(100) Hz wide, and observed to produce "
                    "no coincidences between separated interferometers within the window an "
                    "astrophysical signal can occupy."
                ),
                model=short,
            ),
            GlitchClass(
                name="network_coherent",
                morphology="arch-chirp-with-gaussian-envelope",
                description=(
                    "A seconds-long low-frequency arch, shared by every interferometer that "
                    "takes it. Its morphology, duration, frequency support, rate and amplitude "
                    "law are drawn from the measured Advanced LIGO O3 Scattered Light "
                    "population; its coherence across the network is an extrapolation from the "
                    "shared site and infrastructure of a co-located array, not a measurement."
                ),
                model=coherent,
            ),
        ),
        references=REFERENCES,
        unanchored=UNANCHORED,
    )
