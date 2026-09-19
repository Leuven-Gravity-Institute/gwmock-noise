# The `et-o3-anchored-v1` glitch population

A registered glitch population fixes what a campaign injects: two transient
classes, their morphologies, their rates, their amplitude laws, how they are
shared across a network, and how their arrival times are drawn. Every arm of the
campaign resolves the same name and pins the same digest, so "the arms used the
same glitch model" is something the run stamps can be checked for rather than
something the campaign asserts.

This page is the record of **where each registered number comes from**. Numbers
that are measurements say what was measured, over what data, and what they were
cross-checked against. Numbers that are not measurements are named as such, here
and in the population's own `unanchored` field, which travels with it.

Everything below is reproduced by

```bash
uv run python scripts/measure_glitch_population_anchors.py
```

which downloads about 560 MB on a cold cache and prints every number quoted on
this page. `--write-summary` also rewrites
`src/gwmock_noise/data/glitch_population/o3_reference_summary.json`, the small
summary the package bundles; `tests/test_glitch_population.py` asserts the
registered constants against that summary, through the harness's own estimator
and rounding rule, so the page, the printout and the assertions cannot drift
apart.

## The two classes, and why these two

| | `short_single_channel` | `network_coherent` |
| --- | --- | --- |
| Morphology | Gaussian-windowed broadband burst | Arch-shaped chirp with a Gaussian envelope |
| Model | `BlipGlitch` | `ScatteredLightGlitch` |
| Duration | 0.01 s (envelope FWHM) | 1.75 s |
| Frequency support | 75–780 Hz | 10–120 Hz, apex at 26 Hz |
| Amplitude law | truncated power law in optimal SNR, `α = 1.3333` above 7.5, cut at 344.38 | truncated power law, `α = 1.4184` above 7.5, cut at 601.24 |
| Rate | 2.8394e-4 Hz **per interferometer** (1.02/hr) | 2.7743e-3 Hz for the **network** (9.99/hr) |
| Detector participation | n/a — each interferometer glitches on its own | 1.0, i.e. every interferometer takes every event |
| Arrival process | homogeneous Poisson, independent per interferometer | homogeneous Poisson, one process for the network |

They are two classes rather than ten because they span the two ways a transient
population hurts a coincidence- or coherence-based search, and the pair is what
the consuming campaign needs. The short class is loud, broadband, and resembles
a high-mass compact-binary merger; it fires in one channel at a time, so it
enters a search's background only through accidental coincidence. The longer
class is low-frequency, lasts seconds, and — on interferometers sharing a
site — is expected to fire in all of them at once, which is exactly the case a
coincidence veto and a null stream are built on the assumption of not seeing.

**The two rates are rates of different processes**, and conflating them is the
easiest way to be wrong by the number of interferometers. The short class runs
one Poisson process per interferometer, so its rate is what each of them sees.
The longer class runs one process for the whole network, so its rate is the rate
of shared events; each interferometer sees that times the participation
probability. At the registered participation of 1.0 the two coincide, which is
one reason that value was registered.

## Source data

Two published products, and nothing else.

**The classifications.** Glanzer et al., _Gravity Spy Machine Learning
Classifications of LIGO Glitches from Observing Runs O1, O2, O3a, and O3b_,
Zenodo, [doi:10.5281/zenodo.5649212](https://doi.org/10.5281/zenodo.5649212)
(v1.0.0, 2021-11-07 — the only version). One row per Omicron trigger, carrying
its SNR, duration, peak frequency, central frequency, bandwidth and classified
morphology. The four O3 files are read and the selection is `SNR > 7.5` — the
threshold the set is built on — and machine-learning confidence `> 0.9`, the
fiducial threshold the companion paper uses.

**The livetime.** The Gravitational Wave Open Science Center's O3 observing
segments (`H1_DATA`, `L1_DATA`), summed. This is *observing* time, which is what
a count of triggers must be divided by; using calendar time instead would
understate every rate on this page by about a third.

| | O3a | O3b | O3 total | Implied duty cycle |
| --- | --- | --- | --- | --- |
| H1 | 11,218,675 s (545 segments) | 9,967,195 s (414 segments) | **21,185,870 s** = 245.21 d | 0.710 / 0.784 |
| L1 | 11,956,179 s (535 segments) | 9,810,816 s (352 segments) | **21,766,995 s** = 251.93 d | 0.756 / 0.772 |
| Pooled | | | **42,952,865 s** = 497.14 d | |

**Anchor on the livetime.** The duty cycles implied by these sums are
independently published: Davis et al., _LIGO detector characterization in the
second and third observing runs_, Class. Quantum Grav. **38**, 135014 (2021)
([arXiv:2101.11673](https://arxiv.org/abs/2101.11673)), Table 1, quotes 71 % and
76 % for H1 and L1 in O3a and 79 % for both in O3b, against 71.0 %, 75.6 %,
78.4 % and 77.2 % here. Three of the four agree to better than a percentage
point; the fourth (L1 in O3b) is 1.8 points low, which is the size of the
difference between the published analysis-ready definition and the GWOSC
`_DATA` flag, and it moves the L1 rates below by under 2 %.

**Where the classifications differ from the paper's own table.** Glanzer et al.
2023, _Data quality up to the third observing run of Advanced LIGO: Gravity Spy
glitch classifications_, Class. Quantum Grav. **40**, 065004
([arXiv:2208.12849](https://arxiv.org/abs/2208.12849)), Table 1 gives per-class
O3 counts from a later classification pass than the data release above. The
*fractions* agree to a fraction of a percentage point — Scattered Light 46.1 %
here against "about 47 %" for Hanford, Fast Scattering 27.6 % against
"approximately 27 %" and Tomte 19.2 % against "approximately 19 %" for
Livingston — so the two describe the same population. The *counts* differ by up
to 20 % (Blip at Hanford: 7,457 here, 6,020 there; Scattered Light at Hanford:
68,175 here, 57,118 there), and the rates below inherit that as a systematic.
The measurement is made on the data release rather than on the table because the
release is what can be re-derived.

## 1. The rates

Counts over pooled observing time, per class:

| Class | H1 | L1 | Pooled | Pooled rate |
| --- | --- | --- | --- | --- |
| Blip | 7,457 (1.267/hr) | 4,739 (0.784/hr) | 12,196 | **2.8394e-4 Hz** (1.022/hr) |
| Scattered Light | 68,175 (11.585/hr) | 50,987 (8.433/hr) | 119,162 | **2.7743e-3 Hz** (9.987/hr) |

**The site-to-site spread is the honest uncertainty**, and it is larger than
anything else on this page: a factor **1.62** for Blip and **1.37** for
Scattered Light between two instruments of the same design, in the same run,
analysed by the same pipeline. A single pooled number is a run average over two
sites, not a prediction for a third.

**Anchor on the short class's rate.** Cabero et al., _Blip glitches in Advanced
LIGO data_, Class. Quantum Grav. **36**, 155010 (2019)
([arXiv:1901.05093](https://arxiv.org/abs/1901.05093)) measured the same class
with a completely different method — a reduced-template-bank PyCBC search with
manual morphological vetting, rather than a classifier on spectrograms — over a
different pair of runs. They report "approximately two blip glitches per hour of
data": a median of 39/day at Hanford and 31/day at Livingston in O1, rising to
47/day and 48/day in O2. The measurement here is 30.4/day and 18.8/day in O3.
The two agree in order of magnitude and in the statement that the two sites are
within a factor of about two of each other; the O3 figure is lower by a factor
1.3–2.6, across a change of pipeline, a change of run and a documented
improvement in the instruments. **This is a cross-check, not a calibration**: no
number here was adjusted to match it.

**Anchor on the longer class's rate.** Glanzer et al. 2023 report Scattered
Light as the most common O3 glitch class at Hanford at "about 47 %" of all
classified glitches, and about 23 % at Livingston; the measurement here gives
46.1 % and 22.7 %. Their Table 1 counts, over the same livetime, give
2.70e-3 Hz and 2.17e-3 Hz against 3.22e-3 Hz and 2.34e-3 Hz here — the count
difference described above.

## 2. The amplitude laws

Both classes' loudness is registered **once**, as a truncated power law in
optimal SNR against the configured noise curve. The lognormal amplitude
multiplier every glitch model also carries is left at mean 1.0 and standard
deviation 0.0, deliberately: a second scatter on top of the SNR distribution
would encode the same physical quantity in two places, and the realized SNR
would then be neither the drawn target nor anything measured.

The threshold is 7.5 for both, because that is where the reference sample
starts. A power law fitted above a threshold describes the tail above it and
says nothing about the bulk below; the model reproduces the tail and continues
the same law down to the threshold.

### Tail indices

| Class | Registered `α` | Hill index of the reference sample | The law's own draws' Hill index |
| --- | --- | --- | --- |
| Blip | **1.3333** | 1.3763 ± 0.0125 (n = 12,196) | 1.3763 |
| Scattered Light | **1.4184** | 1.4363 ± 0.0042 (n = 119,162) | 1.4363 |

**The registered exponent is not the Hill index, and the reason is worth reading.**
The Hill estimator is the maximum-likelihood exponent of an *untruncated* Pareto
tail, and it is the natural thing to reach for — the package's own
`PowerLawSNRDistribution` takes the survival exponent in exactly the convention
Hill reports it. But the registered law is truncated, at the largest SNR the
reference sample contains, and a truncated law drawn with the Hill index of a
sample does **not** reproduce that sample: cutting the far tail off raises the
Hill index of what comes out. Drawn with `α = 1.3763` and cut at 344.38, the
short class's own draws have a Hill index of **1.4149** — 2.8 % steeper than the
population the anchor exists to reproduce, with nothing in the configuration to
say so.

The exponent is therefore estimated under the model that will actually be
sampled: the maximum-likelihood exponent of a power law truncated at the sample
maximum, which is itself the maximum-likelihood estimate of the truncation. That
gives 1.3333 and 1.4184, and the third column above is the closed-form Hill
index of draws from those laws — equal to the measurement, which is the property
the choice was made for. `tests/test_glitch_population.py` asserts it, and
asserts that the naive choice would not have satisfied it.

The correction is 3.1 % for the short class and 1.2 % for the longer one, both
in the direction of a *heavier* registered tail than the Hill index alone would
have given. It is small next to the factor-1.6 site-to-site rate spread, and it
is not small next to the 0.9 % statistical error on the short class's own index.

**Neither class is a pure power law from 7.5 upward, and the page says so rather
than hiding it behind one number.** Re-estimating the Hill index at higher
thresholds:

| Threshold | 7.5 | 10 | 15 | 20 | 30 |
| --- | --- | --- | --- | --- | --- |
| Blip | 1.3763 | 1.4800 | 1.7216 | 2.0329 | 2.5595 |
| Scattered Light | 1.4363 | 1.4720 | 1.3916 | 1.5074 | 1.9097 |

The Blip index steepens monotonically — the class has a lighter tail than a
single power law from 7.5 would give, so the registered model **over-produces**
its loudest events relative to the measurement. Scattered Light is flat to about
SNR 20 and steepens beyond. A campaign whose conclusion turns on the far tail of
the short class should read the registered index as an upper bound on how heavy
that tail is; this is much the larger of the two effects on this page, and
unlike the truncation correction it cannot be removed without replacing the
functional form.

The same statement from the other side: the registered law's median is 12.56 for
Blip against a measured median of 13.41 (6 % low), and 12.21 for Scattered Light
against a measured 11.97 (2 % high). The laws are fitted to the tail, and land
within 10 % of the bulk as a by-product, not by construction.

### The loud-tail truncation

| Class | Truncation | Events above it | Fraction of the sample above SNR 150 |
| --- | --- | --- | --- |
| Blip | **344.38** | 0 by construction | 8 of 12,196 (0.066 %) |
| Scattered Light | **601.24** | 0 by construction | 433 of 119,162 (0.363 %) |

Each is the largest SNR the reference sample contains.

**A truncation is required, not cosmetic.** Both indices are below 2, so the
untruncated distributions have infinite variance, and the Blip index is close
enough to 1 that the mean is barely finite. Left untruncated, a long enough run
draws an SNR no instrument could produce, and a background estimate built on it
is dominated by an event that cannot happen.

**What the truncation is, and is not.** It is a property of a *finite
observation*: the largest of ~1.2e4 and ~1.2e5 draws from a heavy tail, which
grows with observing time roughly as `T^(1/α)`. It is not an instrumental
ceiling, and a campaign running far longer than 497 days of pooled observing
time is extrapolating past where this number was measured.

**An alternative published truncation exists for the short class, and it was not
adopted.** Cabero et al. imposed `ρ ≤ 150` on their blip search. That is a
*search selection* — imposed "to avoid loud noise events that are clearly not
blips but contain similar morphology near loud, long-duration noise
transients" — and not a statement that the class never exceeds it; the reference
sample contains 8 events above it, classified as Blip with confidence above 0.9.
Adopting 150 would discard 0.066 % of the class by number and a larger fraction
of its contribution to a loud-tail background. The table above carries the
number so the choice can be revisited with the cost in hand.

## 3. The frequency supports

**The short class: 75–780 Hz**, the 1st and 99th percentiles of the measured
Omicron peak frequency, rounded outward to two significant figures (75.26 and
779.06). The model is a Gaussian-windowed white-noise burst whose spectrum is
flat across its configured band, so the band *is* the support.

That band is 705 Hz wide and centred on 427 Hz. The Omicron columns describing
the same quantity, which are not what the edges were set from, give a median
bandwidth of 913 Hz and a median central frequency of 536 Hz. The model is
therefore about 25 % narrower and 20 % lower than the measured tiles, which is
the size of the mismatch between a flat-in-band burst and a heavy-tailed
distribution of tile shapes.

Cabero et al. describe the class as having "a large frequency bandwidth,
O(100) Hz", and their example time–frequency representation spans 10 Hz to
500 Hz. Both are consistent with the band above; neither is precise enough to
have set it.

**The longer class: 10–120 Hz**, taken from Soni et al., _Reducing scattered
light in LIGO's third observing run_, Class. Quantum Grav. **38**, 025016 (2021)
([arXiv:2007.14876](https://arxiv.org/abs/2007.14876)), which states that slow
scattering "creates 'scatter shelves' in the frequency band 10 Hz to 120 Hz in
h(t) spectra" and that scattered light "impacts the gravitational strain in
10 Hz to 120 Hz frequency band".

This is the one frequency edge on the page that is published rather than
measured, and the reason is that the Omicron columns cannot supply it for this
class. The *peak* frequency can — it is the apex of the arch, measured median
25.97 Hz, registered as **26 Hz**, with 1st and 99th percentiles at 18.50 and
40.28 Hz — but the central frequency and bandwidth columns give medians of
2021 Hz and 4008 Hz for a class whose apex sits at 26 Hz. Those columns describe
the whitened tiling rather than the arch, and rounding them would produce a band
with no relation to the morphology. The published shelf band comfortably
contains the measured apex distribution, which is the consistency check between
the two kinds of anchor, and `tests/test_glitch_population.py` asserts it.

## 4. The duration

**The longer class: 1.75 s**, the median measured Omicron duration (quartiles
1.44 s and 2.50 s; 99th percentile 10.8 s).

Two published cross-checks, and they disagree with each other by more than
either disagrees with this:

- Glanzer et al. 2023 describe Scattered Light as "longer-duration (∼2.0–2.5 s)
  arches" — 14 % to 43 % above the median here.
- Soni et al. Table 1 gives a **median duration of 3.2 s** for slow scattering,
  83 % above. Their selection is different: Livingston only, O3a only, SNR above
  10 rather than 7.5, and restricted to the slow population by arch frequency
  (below 0.2 Hz), which is 40 % of all scattering at that site. Restricting to a
  louder, longer sub-population raises the median, which is the direction the
  difference goes.

**The short class: 0.01 s**, taken from Cabero et al.'s definition of the class
as "a very short duration transient, O(10) ms". It is *not* the measured Omicron
duration, whose median is 0.125 s, and the reason is that the Omicron duration
is a tile length set by the search's `Q` at the trigger's frequency, not the
transient's own width — for this class the 1st and 5th percentiles are 7.8 ms
and 11.7 ms, which is where the intrinsic scale shows through. The registered
width is the FWHM of the model's Gaussian envelope, which is a statement about
the morphology, so the morphological source is the right one.

## 5. What is not anchored

Eight things. Each is carried in the population's `unanchored` field so that a
consumer reading a pinned population reads them with it, and each is a statement
the consumer is making on its own authority.

1. **That any glitch class is coherent across a network at all.** This is the
   population's central extrapolation. The one direct measurement of the
   question says the opposite for separated sites: Cabero et al. Table 1 finds
   **zero** coincidences between Hanford and Livingston blips within the ±15 ms
   window an astrophysical signal can occupy, over both O1 and O2, with the
   coincidences at wider windows matching the accidental expectation exactly
   (19.7 expected, 24 found at ±10 s in O1; 83.8 expected, 65 found in O2). The
   argument for coherence in a co-located array is that its interferometers
   share a site, a vacuum system and a seismic environment, and that the
   scattering mechanism is environmental — Soni et al. observe "similar harmonic
   series of arches at both LHO and LLO", which says the mechanism is common to
   both sites but says nothing about whether two instruments at one site glitch
   together. **No measurement of transient coincidence between co-located
   interferometers was found.**
2. **The participation probability of 1.0**, and therefore the network event
   rate, which is the per-channel rate divided by it. One is the strongest
   coherence the model can express, the hardest case for a coincidence veto or a
   null stream, and the only value that does not introduce a second unmeasured
   number — at one, the network rate and the measured per-channel rate coincide.
   It is a choice, not a measurement, and a campaign should sweep it.
3. **The zero spread of the per-interferometer amplitude ratio** — that a
   common-mode transient couples equally into every interferometer of a site.
4. **The zero inter-interferometer arrival-time offset.** Co-location makes it
   small for an environmental source; nothing measured here bounds it.
5. **Transferring a rate measured on Advanced LIGO to a different instrument,
   site and band.** The rates are what those two detectors did, not a forecast
   of what another one will do — and the factor-1.6 spread between two
   instruments of the same design, in the same run, is the lower bound on how
   wrong that transfer can be.
6. **Identifying the target SNR across noise curves and bands.** The measured
   SNRs are Omicron's, defined against the O3 Advanced LIGO spectra over the
   band Omicron searched; the model calibrates an optimal SNR against the
   configured `psd_file` over the configured band. These are different inner
   products. The sampler reproduces whatever distribution it is given and cannot
   tell whether that identification is the one intended; read the result as "a
   population as loud, relative to this instrument's own noise, as the O3
   population was relative to O3 noise".
7. **The homogeneous Poisson arrival process.** Both measured rates are strongly
   non-stationary. Soni et al. show slow scattering tracking microseismic ground
   motion, particularly on named days of high ground motion; Glanzer et al. show
   the hourly rate of Scattered Light varying by more than an order of magnitude
   across O3 and dropping after reaction-chain tracking was introduced, and Fast
   Scattering varying with the day of the week. A constant rate reproduces the
   run average and not the clustering, and a search's background is sensitive to
   clustering.
8. **The truncation of each class's loud tail at the largest observed SNR**, as
   discussed in §2.

## 6. Decomposing an expected count

`GlitchPopulation.expected_counts` returns the factors, not only the product, so
a total that comes out wrong is traceable to the factor that is wrong:

```text
expected = rate_hz × livetime_seconds × participation × selection
```

For a 2048 s segment of a three-interferometer network, with a recording
threshold at SNR 10:

| Class | Interferometer | Rate / Hz | Livetime / s | Participation | Selection | Expected | Network events |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `short_single_channel` | each of E1, E2, E3 | 2.8394e-4 | 2048 | 1.0 | 0.6714 | 0.3904 | 0.3904 |
| `network_coherent` | each of E1, E2, E3 | 2.7743e-3 | 2048 | 1.0 | 0.6609 | 3.7551 | 3.7551 |

The selection factor is the survival function of the class's own truncated power
law at the threshold, read off in closed form rather than estimated from a
realization.

Two reminders the table is laid out to make hard to miss. The livetime is
*observing* time; a campaign that writes 2048 s segments at less than 100 % duty
must use the analysed total, not the span. And the two rates are rates of
different processes, so the network row's `rate_hz` is not a per-channel rate
and multiplying it by the number of interferometers does not give a network
total.

## 7. A stamped example

```python
from gwmock_noise import get_glitch_population

population = get_glitch_population("et-o3-anchored-v1")
realization = population.realize(
    detectors=["E1", "E2", "E3"],
    duration=4096.0,
    sampling_frequency=4096.0,
    seed=20260919,
)
```

`realization.stamp` is what a campaign manifest pins:

```text
population              et-o3-anchored-v1
schema_version          1.0.0
digest                  sha256:58fb23aef4b4dd0d0038ebdae09da51bdaf638821aca0df97ead74dcbc1dc3b0
```

and it produces, over this hour and a bit of three-interferometer data:

| | E1 | E2 | E3 |
| --- | --- | --- | --- |
| `short_single_channel` | 1 | 0 | 1 |
| `network_coherent` | 8 | 8 | 8 |

— 8 distinct network events, every one of them in all three interferometers, and
two independent short glitches. Against the decomposition of §6 scaled to
4096 s: 1.16 short events per channel expected against 2 seen across three
channels, and 11.4 network events expected against 8. Realized SNRs run from
7.63 to 35.66 and peak strains from 4.2e-23 to 8.2e-23.

**An order-of-magnitude anchor on those strains, which nothing else here
supplies.** For a transient of optimal SNR `ρ` over a band where the PSD is
roughly flat at `S`, the one-sided inner product gives `∫h² dt = ρ² S / 2`. The
bundled 10 km cryogenic curve has an amplitude spectral density of
7.5e-25 /√Hz at 26 Hz, so the loudest event here — a `ρ = 35.7` arch of 1.75 s
with an envelope duty of about a half — should have an RMS strain near 2.0e-23
and a peak a factor of a few above it, around 6e-23. The realization's largest
peak is 8.2e-23. The agreement is the check that the SNR calibration, the PSD
units and the strain scale are consistent with each other; it is a
back-of-envelope, and it would not catch an error smaller than a factor of two.

`realization.strain` is the glitch contribution with no noise under it, which is
what paired arms add so that their glitch content is bit-identical by
construction. `realization.events` is the per-event truth catalogue; rows of a
network-coherent event share a `network_event_id`, which is how the coincident
rows are recognised.

The digest is `sha256` over the canonical serialization — sorted keys, no
insignificant whitespace — of every registered quantity **and** of every prose
field, so a population whose numbers are untouched but whose meaning has been
rewritten digests differently. The digest travels between machines only because
the noise curve is named (`ET_10_full_cryo_psd`) rather than pathed; a population
built against an absolute path carries that path into its digest and nobody else
can reproduce it.
