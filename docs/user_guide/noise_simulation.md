# Advanced: Noise simulation

For CLI and Python snippets see [Minimal usage](minimal_usage.md).

This page details every configuration option, simulator variant, and output
format in `gwmock-noise`.

## Quick example (CLI, TOML)

Create a configuration file, for example:

```toml
# examples/noise_config_example.toml
detectors = ["H1", "L1"]
duration = 4.0
sampling_frequency = 4096.0

[[components]]
simulator = "white"

[[components]]
simulator = "spectral_lines"
lines = [{ frequency = 60.0, amplitude = 1.0e-3 }]

[output]
directory = "./output"
prefix = "noise"

seed = 42
```

Then run:

```bash
gwmock-noise simulate examples/noise_config_example.toml
```

This will create one NumPy strain artifact plus one JSON metadata sidecar per
detector in the configured output directory (for example `output/noise_H1.npy`
and `output/noise_H1.json`). The JSON file describes the produced artifact; the
strain samples live in the `.npy` file and `SimulationResult.output_paths`
points to that real data artifact.

## Configuration

Noise simulations are configured with a Pydantic model
`gwmock_noise.NoiseConfig`. When using the CLI, the configuration is loaded from
TOML, YAML, or JSON into the same model.

Supported top-level fields:

| Field                | Type                   | Description                                                                                           |
| -------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `detectors`          | `list[str]`            | Names of detectors to simulate (for example `H1`, `L1`)                                               |
| `duration`           | `float`                | Duration of the realization in seconds (`> 0`)                                                        |
| `sampling_frequency` | `float`                | Sampling frequency in Hz (`> 0`)                                                                      |
| `components`         | `list[str \| mapping]` | Ordered simulator components; each entry is a simulator name or mapping                               |
| `output.directory`   | `path`                 | Output directory for generated files                                                                  |
| `output.prefix`      | `str`                  | Prefix for output file names; may not contain `/`, `\`, or `:` (default: `noise`)                     |
| `output.format`      | `str`                  | Artifact format written by `run(config)`: `npy` (default), `gwf`, or `hdf5`                           |
| `output.gps_start`   | `float`                | GPS start time used for timestamped formats such as `gwf` and `hdf5`                                  |
| `output.channel`     | `str`                  | Channel name for `gwf` and `hdf5` output, assembled as `{detector}:{channel}` (default: `MOCK_NOISE`) |
| `output.channels`    | `dict[str, str]`       | Per-detector full channel names (e.g. `{"H1": "H1:STRAIN_NOISE"}`); overrides `channel` when set      |
| `seed`               | `int` or `null`        | Optional random seed for reproducibility                                                              |

### Output formats

`npy` writes one bare array per detector, plus the JSON sidecar every format
writes. Neither carries the epoch: the sidecar records the duration and the
sampling frequency but not `gps_start`, so a reader can recover the sample
spacing and not the absolute time. `gwf` writes frame files, for pipelines that
read frames. `hdf5` writes one file per detector carrying the samples together
with the epoch, the sample interval, the channel and the unit, so a reader does
not need to be told the grid separately; GWpy reads these files directly.

Detector, channel and prefix names may not contain `/` or `\`, nor any character
Windows reserves in a file name: `< > " | ? *`, and anything below `0x20`, which
includes newline and tab. Detector and prefix names may not contain `:` either.
A channel may carry one colon, and only one: a resolved channel is `IFO:name` by
convention, and that prefix is dropped when the channel enters a frame name. A
detector or channel may not be empty.

Two different reasons sit behind that list. `/` is a group separator inside an
HDF5 file, so a channel carrying one writes the data into a nested group instead
of the dataset the reader looks for. The rest cannot appear in a file name on at
least one supported platform -- and they are refused **everywhere**, not only on
Windows, so that the same configuration stays valid wherever it is run.

An empty prefix is accepted, but note that it does not remove the separator: the
artifacts are named `_H1.npy` and `_H1.json`, not `H1.npy`. Two detectors may
also not compose the same artifact name -- `H1` and `h1` differ as strings and
name one file on macOS and Windows -- and a detector may not be repeated. Those
two are checked by the simulator and by the frame writer as well as by the
config, since a configuration can be built in ways that skip validation.

Channel names are checked for the formats that use the channel: `npy` writes a
bare array and never reads it, so a channel is not restricted there. **Detector
and prefix names are checked for every format**, because both become part of a
file name whatever the format is -- and of the JSON sidecar's name too.

`gps_start` and `duration` must be **whole seconds** for `gwf` and `hdf5`, whose
artifact names carry both -- `H-H1_MOCK_NOISE_1187008512-4096.gwf`, following
the observatory convention. They previously accepted sub-second values and
encoded them as `100p25`, which gave two times that round alike -- `1.0` and
`1.0000001`, say -- the same name: the second run silently overwrote the first.
`npy` is unaffected, since its name carries no time at all, so a fractional
duration there collides with nothing.

The simulator checks the same rule again before it generates anything, for every
output format -- a config can be constructed in ways that skip validation, and a
check made while writing would leave the artifacts already written behind. That
second check covered HDF5 alone at first, which left the bypass open for `npy`
and `gwf`: a detector named `H1/A` wrote `noise_H1/A.npy` when that directory
happened to exist, reporting success for a path below the output directory the
run was given.

HDF5 artifacts are named for the detector -- `H-H1_1000000000-4.hdf5` -- rather
than for the channel as frames are. The channel is stored inside the file. Two
reasons: a channel can contain characters that are not valid in a file name on
every platform this runs on, and escaping them made two distinct channels
collide onto one name, silently losing a detector's data.

For integration with the upstream `gwmock` package, the same structure can be
nested under a `noise` key inside a larger configuration file. In that case the
CLI still works; it automatically looks for a `noise` section if present.

## Component composition

`NoiseConfig.components` is the extension point for built-in simulations. Each
entry is either a string shorthand such as `"white"` or a mapping with a
`simulator` name plus simulator-specific options.

Components are evaluated in order and combined additively, so users can build a
simulation from whichever parts they need without editing the top-level schema.
For example, colored background noise, spectral lines, and glitches can live in
one config:

```toml
detectors = ["H1", "L1"]
duration = 8.0
sampling_frequency = 4096.0
seed = 42

[[components]]
simulator = "colored"
psd_file = "ET_D_psd"

[[components]]
simulator = "spectral_lines"
lines = [{ frequency = 60.0, amplitude = 1.0e-3 }]

[[components]]
simulator = "glitches"
models = [
  { kind = "blip", rate = 0.25, width = 0.01, amplitude_distribution = { distribution = "lognormal", mean = 0.5, std = 0.0 } }
]
```

Every glitch model runs an independent Poisson process per detector, so event
times and waveforms are uncorrelated between detectors and `rate` is the event
rate seen by each individual detector. The metadata sidecar reports, for each
model, the total number of injected events (`count`) plus a per-detector
breakdown (`count_by_detector`).

A glitch whose waveform runs past the end of a streamed chunk has its remainder
carried into the next chunk and replayed in event order, so the injected glitch
series is identical, sample for sample, to a single generate call of the same
total duration; a tail is dropped only when it overflows the final chunk, where
the data window ends.

### Scoping a model to some interferometers

A model applies to every interferometer in the run unless it says otherwise.
`detectors` says otherwise: one name or a list of them, and the model then
injects only there. That is what lets a single configuration describe a network
whose instruments differ — a 10 km triangle and a 15 km 2L do not share a noise
curve, so they cannot share a `psd_file` — and what lets rates differ per
interferometer, which is what the instruments actually do: in O3,
`Fast_Scattering` fired about 29 times more often in L1 than in H1.

```toml
detectors = ["ET1_SARD", "ET2_SARD", "ET3_SARD", "ET1_2L_ALIGNED_SARD", "ET2_2L_ALIGNED_EMR"]

[[components]]
simulator = "glitches"
models = [
  { kind = "blip", rate = 0.2, width = 0.01, psd_file = "ET_10_full_cryo_psd", snr = 20.0, detectors = ["ET1_SARD", "ET2_SARD", "ET3_SARD"], amplitude_distribution = { distribution = "lognormal", mean = 1.0, std = 0.0 } },
  { kind = "blip", rate = 0.2, width = 0.01, psd_file = "ET_15_full_cryo_psd", snr = 20.0, detectors = ["ET1_2L_ALIGNED_SARD", "ET2_2L_ALIGNED_EMR"], amplitude_distribution = { distribution = "lognormal", mean = 1.0, std = 0.0 } },
]
```

The sidecar records each model's `detectors`, `null` meaning all of them, so a
run says which interferometers a model applied to rather than leaving it to be
inferred from the configuration that produced it.

**A configuration that does not say what every interferometer gets is refused,
not run.** Before selectors existed a single `psd_file` colored the whole
network, so the configuration above — written with one model — applied the 10 km
curve to the 15 km instruments as well, and the run succeeded silently: SNRs
23–44% away from the 20 that was asked for, varying with glitch morphology, with
nothing in the output or the logs to say so. Three cases now raise a
`ValueError` naming the interferometers involved:

- **An interferometer no model claims.** Its strain would be written without
  glitches while the rest of the network carries them. Where that is the intent,
  say it in the configuration: give it a model with `rate = 0.0`.
- **Two coloring PSDs claiming one interferometer.** An interferometer has one
  noise floor; two models coloring it against different curves disagree about
  what instrument it is. Models with no `psd_file` impose no floor and are not
  part of this.
- **A selector naming an interferometer the run does not have** — a typo or a
  leftover from another network, whose model would never fire.

A model written without `detectors` still applies to every interferometer, so an
existing configuration keeps its meaning exactly, and a run whose models are all
unscoped covers the network by construction.

### The glitch truth catalogue

Counts say how many glitches went in; the truth catalogue says which ones, and
that is what a detection-efficiency curve, a classifier's training labels or a
veto study needs. Every injected event is recorded as it fires, under
`glitches.catalogue` in the metadata sidecar and on the simulator itself as
`InjectGlitches.glitch_events` (the whole run) and
`InjectGlitches.segment_glitch_events` (just the chunk generated last):

```json
{
    "event_id": "H1-0-3",
    "detector": "H1",
    "model_index": 0,
    "kind": "deepextractor",
    "glitch_class": "Koi_Fish",
    "gps_start_time": 1256655661.5,
    "gps_peak_time": 1256655662.47,
    "duration_seconds": 2.0,
    "n_samples": 8192,
    "segment_index": 10,
    "sample_index": 1638,
    "target_snr": 8.0,
    "realized_snr": 8.0,
    "amplitude": 1.0
}
```

**`gps_start_time` is where the waveform starts, not where it peaks.** The
Poisson process draws the time of the waveform's _first sample_, so for a 2 s
DeepExtractor reconstruction the visible transient sits about a second later;
`gps_peak_time` is the largest-|strain| sample of the same waveform. Cutting an
analysis window around the wrong one of the two misses the glitch. The sidecar
carries the same statement in `glitches.catalogue.time_convention`, and a
one-line description of every column in `glitches.catalogue.columns`.

`target_snr` is the optimal SNR the draw was calibrated to, before the amplitude
multiplier; `realized_snr` is what the injected samples actually carry against
the PSD they were colored with. They differ by `amplitude`, and both are `null`
for a model with no PSD, which has no SNR to report.

A row is written where a glitch _starts_, so a waveform straddling a chunk
boundary appears once, at its true time, in the chunk holding its first sample —
the carried-over tail adds samples to the next chunk but no second row. A
streamed run and a single generate call of the same total duration therefore
produce the same catalogue, and it replays exactly for a fixed (version, config,
seed) just as the strain does.

Times are GPS: `output.gps_start` is the epoch of the first sample, and each
generated segment advances it by its own duration. A caller driving
`InjectGlitches` directly can pass `gps_start=` or assign it per segment; it
defaults to `0.0`, which makes the catalogue's times seconds from the start of
the run.

### Parametric glitch models

Two built-in models are described analytically rather than drawn from data:

- **`blip`** — a short, broadband burst: white-noise carrier under a Gaussian
  envelope whose full width at half maximum is `width` seconds. It approximates
  the common "blip" transient (a brief, roughly symmetric broadband tick) and,
  uncolored, has a flat spectrum. Parameter: `width`.
- **`scattered_light`** — an arch-shaped chirp modelling light scattered off a
  slowly moving surface: a Gaussian-enveloped sinusoid whose instantaneous
  frequency arches up and back down over the event as
  `peak_frequency * |sin(pi t / duration)| ** arch_exponent`. This reproduces
  the stacked "arches" seen in scattered-light glitches. Parameters: `duration`,
  `peak_frequency`, `arch_exponent`, `phase`.

Both draw their overall amplitude from `amplitude_distribution` and are, by
default, defined purely by these parameters (no detector noise floor enters).
See the next section to shape either against a target PSD.

### Optional PSD coloring

By default `blip` is a spectrally flat (white-noise) burst and `scattered_light`
is a deterministic arch chirp. Both accept an optional `psd_file` that shapes
the waveform's spectrum by `sqrt(PSD)` inside the analysis band
(`low_frequency_cutoff`, `high_frequency_cutoff`), so the glitch sits in a
realistic noise floor. When a target `snr` is also given, the waveform is
rescaled so its optimal SNR against that PSD equals `snr` (the
`amplitude_distribution` then scales on top); `snr` requires `psd_file`. Omit
`psd_file` for the original uncolored behavior.

```toml
[[components]]
simulator = "glitches"
models = [
  { kind = "blip", rate = 0.1, width = 0.01, psd_file = "ET_D_psd", snr = 12.0, amplitude_distribution = { distribution = "lognormal", mean = 1.0, std = 0.0 } },
]
```

### Sampled SNR distributions

A number for `snr` calibrates every event of a model — or, for a per-class
mapping, every event of a class — to exactly the same loudness. Measured glitch
populations are not like that: they are heavy-tailed, and for the heaviest
classes the tail is where most of the class's effect on a search or a classifier
sits. The `amplitude_distribution` multiplier cannot stand in for it, because a
log-normal has every moment finite while a power law with an index below one has
no finite mean at all.

So `snr` also accepts a **distribution**, and the target is then drawn per event
from the same random stream as the rest of the model — which keeps a run
reproducible for a fixed (version, config, seed) exactly as the fixed-SNR path
is. Two shapes are supported, and both work for `blip`, `scattered_light` and
`deepextractor`:

```toml
# Power law above a threshold, truncated at `maximum` (see below for the
# survival function in each case).
snr = { distribution = "power_law", minimum = 10.0, alpha = 1.34, maximum = 621.2 }

# Draw with replacement from observed SNRs, given inline ...
snr = { distribution = "empirical", samples = [11.4, 42.0, 92.2, 621.2] }
# ... or from a file: one SNR per line, or an HDF5 file with an `snr` dataset
# (the schema `gwmock-noise build-blip-glitch-table` writes).
snr = { distribution = "empirical", file = "observed_snrs.txt" }
```

`alpha` is the exponent of the **survival** function, which is the convention a
Hill or maximum-likelihood tail index is quoted in — one less than the density's
exponent. A tail index measured above some threshold therefore goes in as it was
measured, with `minimum` set to the threshold it was measured above.

The survival function itself depends on whether the tail is capped. With
`maximum` unset it is the plain power law,

```text
S(s) = (s / minimum) ** -alpha,          s >= minimum
```

and setting `maximum` renormalizes that onto `[minimum, maximum]`,

```text
(S(s) - S(maximum)) / (1 - S(maximum)),  minimum <= s <= maximum
```

which reaches zero at the cap instead of carrying probability past it. The
truncated form is what the sampler inverts whenever `maximum` is set, so a
capped configuration is not the uncapped one with its tail discarded — the
probability the cap removes is spread back over the range below it.

Two things are worth being deliberate about:

- **`minimum` is a threshold, not a fit to the whole population.** A measured
  index describes the tail above the threshold and says nothing about the bulk
  below it, so the model reproduces the tail and replaces the bulk with the same
  power law continued down to `minimum`.
- **`maximum` is optional but matters for a heavy tail.** Left unset the power
  law is unbounded, and with `alpha` below 1 a long enough run will eventually
  draw an SNR no detector could produce — at `alpha = 0.4` and `minimum = 10`,
  one draw in a hundred lands above SNR 10⁶. The largest SNR observed for the
  class is the natural cap. Below `alpha` of about 0.052 the cap stops being
  optional: the uncapped draw then runs off the top of the double-precision
  range, so such a configuration is refused when the model is built rather than
  left to fail partway through a run. Every index in the measured range is far
  above that and is unaffected.

For `deepextractor` the two forms compose per class, and can be mixed freely —
one class sampled, another pinned:

```toml
[[components]]
simulator = "glitches"
models = [
  { kind = "deepextractor", rate = { Blip = 3.5e-4, Koi_Fish = 6.2e-4 }, psd_file = "noise_psd.txt", glitch_classes = ["Blip", "Koi_Fish"], amplitude_distribution = { distribution = "lognormal", mean = 1.0, std = 0.0 }, snr = { Blip = { distribution = "power_law", minimum = 10.0, alpha = 1.34, maximum = 621.2 }, Koi_Fish = { distribution = "power_law", minimum = 10.0, alpha = 0.40, maximum = 11841.5 } } }
]
```

Per-event targets are recorded in the glitch truth catalogue, so `target_snr`
reports the value the event was actually drawn with rather than the shape it
came from, and the sampled population can be read back off a finished run.

**What the target means is yours to decide.** Tail indices and SNR tables
usually come from a trigger generator — Omicron, say — whose SNR is defined
against _that pipeline's_ PSD over the band it searched, while this model
calibrates against `psd_file` from `low_frequency_cutoff` upward. Feeding one
into the other equates two SNRs defined against different noise curves over
different bands, which is a statement about the population you are asking for,
not about the sampler: the sampler reproduces whatever distribution it is given
and cannot tell whether that identification is the one you meant. If the two
noise curves differ materially, rescale the measured SNRs before configuring
them, or read the resulting population as "the same loudness distribution,
expressed in this detector's band".

## Gengli blip glitches

`gwmock-noise[gengli]` adds a file-backed `GengliBlipGlitch` model that plugs
into a `glitches` component. The expected population file is an HDF5 file with
an `snr` dataset; the built-in CLI can generate that file from a GravitySpy CSV
export:

```bash
gwmock-noise build-blip-glitch-table --gravity-spy-csv gravity_spy.csv --out glitches.h5
```

Programmatic configuration uses the same `NoiseConfig` surface as the built-in
parametric glitches:

```python
from pathlib import Path

from gwmock_noise import (
    GengliBlipGlitch,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
)

config = NoiseConfig(
    detectors=["L1"],
    duration=8.0,
    sampling_frequency=4096.0,
    components=[
        {"simulator": "colored", "psd_file": Path("noise_psd.txt")},
        {
            "simulator": "glitches",
            "models": [
                GengliBlipGlitch.from_population_file(
                    "glitches.h5",
                    rate=0.25,
                    psd_file=Path("noise_psd.txt"),
                    amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
                )
            ],
        },
    ],
)
```

The model samples an SNR from the population table for each injected event,
generates one whitened gengli blip, and colors it against the configured PSD
before additive injection through `InjectGlitches`.

## DeepExtractor glitches

`gwmock-noise[deepextractor]` adds a `DeepExtractorGlitch` model that injects
real O3 glitch reconstructions from the
[DeepExtractor dataset](https://huggingface.co/datasets/tomdooney/deepextractor-glitch-reconstructions)
(CC BY 4.0). The dataset holds 35,000 whitened, amplitude-normalized 2-second
waveforms at 4096 Hz covering seven Gravity Spy classes (`Blip`,
`Fast_Scattering`, `Koi_Fish`, `Low_Frequency_Burst`, `Scattered_Light`,
`Tomte`, `Whistle`); the ~2.3 GB samples file is downloaded lazily on first use
and cached by `huggingface_hub`.

Each injected event draws a reconstruction from the configured classes,
resamples it to the simulation rate, colors it against `psd_file`, and rescales
it so its optimal SNR `sqrt(4 df sum(|h(f)|^2 / S(f)))` against that PSD equals
the configured target. `snr` accepts a single number for all classes, a
per-class mapping, or a sampled distribution per class (see
[Sampled SNR distributions](#sampled-snr-distributions), which is what a
heavy-tailed class needs). `rate` likewise accepts either a single number — the
total Poisson rate shared by all configured classes, drawn uniformly — or a
per-class mapping, in which case each class occurs at its own rate (the total
rate is their sum):

```toml
[[components]]
simulator = "glitches"
models = [
  { kind = "deepextractor", rate = { Blip = 0.04, Koi_Fish = 0.01 }, psd_file = "noise_psd.txt", snr = { Blip = 12.0, Koi_Fish = 8.0 }, glitch_classes = ["Blip", "Koi_Fish"], amplitude_distribution = { distribution = "lognormal", mean = 1.0, std = 0.0 } }
]
```

With the default `amplitude_distribution` mean of 1.0 and std of 0.0 the target
SNR is met exactly; a non-zero std adds multiplicative SNR scatter. Events are
placed by the same Poisson `rate` process as the other glitch models, run
independently per detector: each detector receives its own event times and
waveform draws, and `rate` is the event rate seen by each detector. Note that
resampling below 4096 Hz uses linear interpolation without an anti-aliasing
filter, which aliases high-frequency content (the SNR calibration itself is
unaffected).

The dataset is cached by `huggingface_hub` after the first download, so later
runs reuse the cached files. Each run contacts the Hub first to validate the
cached files' ETags and fetch anything missing; if the Hub is unreachable a
warning notes that the ETag check was skipped and the cached files are used
instead, and only a genuinely missing cache raises `LocalEntryNotFoundError`.
Set `local_files_only = true` to skip the network unconditionally and read
straight from the cache.

Set `revision` to pin the download to a specific dataset version — a git branch,
tag, or commit SHA passed straight to `hf_hub_download`. Leave it unset to track
the repository default. Either way the first download resolves to a concrete
commit SHA, and that SHA is what the run metadata records (not the branch name
you asked for). Replaying a run from its metadata therefore fetches the exact
commit that produced it, so glitch generation stays bit-reproducible for a fixed
(version, config, seed) even as the upstream dataset moves.

## Schumann-resonance correlated noise

The `schumann` simulator generates strain noise from the global magnetic field
of the Earth–ionosphere cavity. Its Schumann resonances (~7.8, 14, 20, … Hz) are
represented as a sum of Lorentzian peaks (`SchumannParams`:
`mode_frequencies_hz`, `quality_factors`, `amplitudes`), a per-detector
magnetic-to-strain coupling (`coupling_files`) converts the field to strain, and
the shared magnetic origin makes the noise **correlated between detectors**.
That correlated magnetic noise from Schumann resonances is coherent across
globally separated detectors — and can limit stochastic-background searches —
was established observationally by Thrane, Christensen & Schofield, _Correlated
magnetic noise in global networks of gravitational-wave detectors_, Phys. Rev. D
**87**, 123009 (2013) ([arXiv:1303.2613](https://arxiv.org/abs/1303.2613)); see
also Coughlin et al., Class. Quantum Grav. **33**, 224003 (2016) on measurement
and subtraction.

**Why detector positions matter.** Because the resonant field fills the whole
cavity, two detectors see a correlated field whose coherence depends on their
angular separation on the globe, not on their local orientation. The simulator
uses an idealized **isotropic cavity-mode** approximation: the `n`-th resonance
is dominated by spherical-harmonic degree `n` (the cavity modes satisfy
`f_n ≈ (c / 2π R_earth) √(n(n+1))`), whose zonal correlation between two surface
points separated by great-circle angle `θ` is the Legendre polynomial

```text
coherence(f) = P_n(cos θ),   n ≈ round(2π f R_earth / c).
```

The simulator therefore requires each detector's geographic `positions`
(latitude, longitude): nearby sites stay strongly correlated, while widely
separated sites decorrelate (and the coherence changes sign) as `n` grows with
frequency. This `P_n(cos θ)` form is the simulator's own modelling
approximation, not a result taken from the references above.

## Programmatic usage

You can also construct configurations and run the simulator directly from
Python:

```python
from pathlib import Path

from gwmock_noise import DefaultNoiseSimulator, NoiseConfig, OutputConfig

config = NoiseConfig(
    detectors=["H1", "L1"],
    duration=4.0,
    sampling_frequency=4096.0,
    output=OutputConfig(directory=Path("output"), prefix="noise"),
    seed=42,
)

simulator = DefaultNoiseSimulator()
result = simulator.run(config)

for detector, path in result.output_paths.items():
    print(detector, "->", path)
```

Colored-noise components accept `psd_file` values as local paths, HTTP(S) URLs,
and bundled preset names. The Einstein Telescope presets are `ET_D_psd`,
`ET_10_HF_psd`, `ET_10_full_cryo_psd`, `ET_15_HF_psd`, `ET_15_full_cryo_psd`,
`ET_20_HF_psd`, and `ET_20_full_cryo_psd`; the Advanced LIGO presets are
`aLIGO_O3_actual_H1_psd`, `aLIGO_O3_actual_L1_psd`,
`aLIGO_O4_high_projected_psd`, and `aLIGO_O4_low_projected_psd`. The same names
work for `psd_file` on the glitch models, which matters when coloring a
LIGO-derived glitch (DeepExtractor reconstructions and gengli blips are both
LIGO glitches) against the instrument that produced it rather than against an ET
curve.

The `O3_actual` curves are measured O3 spectra and still carry instrumental
lines; the `O4_projected` curves are pre-run sensitivity forecasts, not
measurements. `src/gwmock_noise/data/psd/PROVENANCE.md` records the source
document, the ASD-to-PSD conversion, and a BNS-range cross-check for every
Advanced LIGO curve.

The upstream `gwmock` package is expected to import and compose
`gwmock_noise.NoiseConfig` into its own configuration model and to drive a noise
simulator that implements the `gwmock_noise.BaseNoiseSimulator` interface.

## Frequency resolution and the synthesis window

The colored, correlated, and Schumann simulators synthesize noise in
`window_duration`-second blocks and stitch them together. The frequency
resolution of the generated noise is therefore **approximately**
`Δf ≈ 1 / window_duration` (default `4.0 s` → `0.25 Hz`), largely independent of
`sampling_frequency`. The block length is rounded to a whole number of samples
(`round(window_duration × sampling_frequency)`), so the realized `Δf` can differ
slightly from `1 / window_duration` — most noticeably for short windows or low
sampling rates. Input PSD structure finer than `Δf` cannot be reproduced, so
increase `window_duration` to resolve narrow or fast-varying features:

```python
from gwmock_noise import CorrelatedNoiseSimulator

simulator = CorrelatedNoiseSimulator(
    psd_files={"D1": "d1_psd.txt"},
    detectors=["D1"],
    sampling_frequency=16384.0,
    low_frequency_cutoff=2.0,
    window_duration=16.0,  # Δf = 0.0625 Hz, resolves few-Hz PSD structure
)
```

When the window is too coarse for the input spectrum (or for the requested
`low_frequency_cutoff`), the simulator emits a `WARNING` through the
`gwmock-noise` logger suggesting a larger `window_duration`. A larger window
improves resolution at the cost of more samples per synthesis block.

## Spectral covariance utilities

`gwmock_noise.spectral` exposes the lower-level PSD/CSD operations used by the
correlated-noise simulator. These helpers are signal-agnostic, so
`gwmock-signal` can use them when building multi-detector SGWB data products
without depending on simulator internals.

The convention is one-sided spectra in units of strain squared per Hz. For each
positive real-FFT bin with spacing `df`, a spectral covariance matrix `S(f)` is
converted to complex coefficient covariance `S(f) / (2 df)`. The inverse real
FFT then applies the simulator normalization `df * n`, where `n` is the chunk
length. With this convention, a one-sided periodogram of long generated strain
segments recovers the input PSD/CSD away from taper and edge effects.

The public workflow is:

1. Load and interpolate detector PSDs with `load_and_interpolate_psd(...)`.
2. Load and interpolate pairwise complex CSDs with
   `load_and_interpolate_csd(...)`.
3. Assemble per-frequency Hermitian matrices with
   `assemble_hermitian_spectral_matrices(...)`.
4. Build regularized coefficient-space Cholesky factors with
   `cholesky_factors_from_spectral_matrices(...)`, or use
   `build_spectral_covariance_from_files(...)` to perform the whole file-backed
   path.
5. Draw real detector chunks with `simulate_spectral_covariance_chunk(...)`.

When `output.format = "gwf"`, `run(config)` reuses the built-in GWpy/GWF output
stack to write frame files instead of NumPy artifacts. The metadata sidecar is
still written, and `SimulationResult.output_paths` points to the generated GWF
files.

For stateful continuation across chunk boundaries, use the public streaming
contract instead of reseeding separate runs:

```python
import numpy as np

from gwmock_noise import ColoredNoiseSimulator, open_stream

simulator = ColoredNoiseSimulator(
    psd_file="example_psd.txt",
    detectors=["H1", "L1"],
    sampling_frequency=4096.0,
)
stream = open_stream(
    simulator,
    chunk_duration=4.0,
    sampling_frequency=4096.0,
    detectors=["H1", "L1"],
    seed=42,
)

first_three_chunks = [next(stream) for _ in range(3)]
strain_h1 = np.concatenate([chunk["H1"] for chunk in first_three_chunks])
```

`open_stream(...)` is the supported public continuation surface for
`NoiseSimulator` implementations. Shipped colored and correlated simulators keep
their overlap-add state inside the iterator, so concatenating sequential chunks
reproduces the same realization as one seeded single-shot `generate(...)` call.

## Overlap-save FIR colouring

`OverlapSaveFirSimulator` is a bounded-state alternative to the overlap-add
simulators. It designs a single causal colouring filter from the target PSD and
applies it to white noise with **overlap-save block convolution**, so the only
continuation state is the filter memory: a running stream holds
`filter_length - 1` input samples per detector no matter how long it runs.

The filter is the inverse transform of the target's square root, truncated to
`filter_length` samples around zero lag and tapered with a Hann design window.
With `minimum_phase=True` (the default) a cepstral minimum-phase factorisation
concentrates the filter energy at the front, so the truncation loses less of the
target.

`filter_length` (`L_f`) is the accuracy/state trade-off knob. It must be a power
of two between `2**4` and `2**16` samples: a longer filter follows the target
spectrum more closely and costs more state.

```python
from gwmock_noise import OverlapSaveFirSimulator

simulator = OverlapSaveFirSimulator(
    psd_file="example_psd.txt",
    filter_length=512,
    detectors=["H1"],
    sampling_frequency=4096.0,
)
strain = simulator.generate(4.0, 4096.0, ["H1"], seed=42)
print(simulator.state_nbytes, simulator.resume_metadata_nbytes)
```

The simulator also accepts an in-memory one-sided target as `target_psd` (with
an optional `target_frequencies` grid) instead of a file, so an analytic target
can bypass the tabulated-curve interpolation.

## AR (Levinson-Durbin) and ARMA / state-space simulators

`ARNoiseSimulator` fits an all-pole model to the autocovariance implied by the
tabulated PSD with the **Levinson-Durbin recursion**. Stability is guaranteed by
construction: for a positive-definite autocovariance every reflection
coefficient has magnitude below one, so every pole lies strictly inside the unit
circle. The recursion enforces the pre-registered limits while it runs and
raises `FitError` if a coefficient reaches the unit circle or the prediction
error loses its sign, so a fit that cannot be trusted fails instead of
degrading. The metadata records the conditioning diagnostics
(`max_reflection_coefficient`, `min_prediction_error`,
`toeplitz_condition_number`), the state size in bytes, and the relative PSD
residual per geometric band (`fit_residual`).

```python
from gwmock_noise import ARNoiseSimulator

simulator = ARNoiseSimulator(
    psd_file="aLIGO_O4_high_projected_psd",
    order=256,
    detectors=["H1"],
    sampling_frequency=4096.0,
    low_frequency_cutoff=20.0,
)
strain = simulator.generate(4.0, 4096.0, ["H1"], seed=42)
print(simulator.metadata["autoregressive_noise"]["conditioning"])
print(simulator.metadata["autoregressive_noise"]["fit_residual"])
```

`ARMANoiseSimulator` adds a moving-average numerator on top, giving a
bounded-state ARMA / state-space filter. Spectral lines are handled by **pole
placement**: each requested (or detected) line contributes a conjugate pole pair
whose radius is set by the line width. The target is pre-whitened by those
poles, the smooth remainder is fitted with the same stable Levinson recursion,
and the numerator is obtained by **log-spectrum matching** (fitting the log
spectrum with a cosine series and factoring it with the cepstral method). The
metadata reports both orders, the state size, the placed lines and the per-band
residual.

```python
from gwmock_noise import ARMANoiseSimulator

simulator = ARMANoiseSimulator(
    psd_file="ET_D_psd",
    ar_order=192,
    ma_order=32,
    detectors=["H1"],
    sampling_frequency=4096.0,
    low_frequency_cutoff=5.0,
    detect_lines=True,
)
strain = simulator.generate(4.0, 4096.0, ["H1"], seed=42)
print(simulator.metadata["autoregressive_moving_average"]["placed_lines"])
```

`ARMANoiseSimulator` also accepts an in-memory `target_psd` (with an optional
`target_frequencies` grid) instead of a file. Both simulators expose
`state_nbytes` and support `export_state()` / `import_state()` exactly like the
other bounded-state simulators; the reported `state_size` is the delay-line
length the filter actually carries, including the two taps each placed conjugate
pole pair adds.

## Multichannel generation from a PSD/CSD matrix

`MultichannelNoiseSimulator` generates correlated multichannel noise from a
tabulated PSD/CSD matrix. It fits a causal minimum-phase matrix spectral factor
with the **Whittle block Levinson-Durbin recursion** -- the multivariate form of
the AR recursion above -- and drives the resulting vector autoregression

```text
x_t = -A_1 x_{t-1} - ... - A_p x_{t-p} + V^{1/2} w_t
```

with white innovations. Stability is guaranteed by construction: the recursion
keeps the prediction-error covariance positive definite, so every zero of
`det A(z)` lies strictly inside the unit circle and the filter and its inverse
are both causal. The continuation state is the `order` samples of history per
channel, independent of the generated span.

The cross-spectral matrix may be given as files (`psd_files` plus `csd_files`,
in the same form as `CorrelatedNoiseSimulator`) or directly as an array
(`target_matrices` with `target_frequencies`). A pair without a CSD file means
zero coherence. A complex CSD's phase is carried into the cross-channel lag
covariances; the stored value is the one-sided cross-spectrum whose
autocovariance is its inverse transform, so a CSD exported by a tool that
defines the opposite conjugation should be stored conjugated. The metadata
records the fit method, the order, the per-band relative Frobenius residual of
the modelled cross-spectral matrix against the target, the conditioning
diagnostics and the state size. An in-band target that is not positive definite
raises `FitError` rather than being silently changed, unless a relative ridge is
requested through `regularization_epsilon`, which is applied as a zero-lag
(white) floor.

```python
from gwmock_noise import MultichannelNoiseSimulator

simulator = MultichannelNoiseSimulator(
    psd_files={"E1": "E1_psd.txt", "E2": "E2_psd.txt"},
    csd_files={("E1", "E2"): "E1_E2_csd.txt"},
    order=256,
    detectors=["E1", "E2"],
    sampling_frequency=4096.0,
    low_frequency_cutoff=5.0,
)
strain = simulator.generate(4.0, 4096.0, ["E1", "E2"], seed=42)
print(simulator.metadata["multichannel_noise"]["fit_residual"])
```

**Relationship to `CorrelatedARNoiseSimulator`.** The incumbent multichannel
generator takes the per-frequency Cholesky factor of the same target and
truncates it to a VMA, so its filter is causal only through truncation; the
Whittle factor is causal by construction, and its state is a bounded recursion
history of the same order. The two are compared on a shared target in the test
suite, where the Whittle generator's band-integrated PSD and CSD errors are
smaller at equal order.

The comparison against an independent exact multivariate circulant embedding
(Helgason, Pipiras and Abry 2011) is not part of this branch: the paper-side
reference does not exist yet and that arm is recorded as gated and unanchored.
The generator is therefore verified against the target's own PSD/CSD definition
and the analytic per-band comparisons, not against an independent exact
multivariate reference.

## Resuming a stopped stream

The colored and overlap-save FIR simulators can persist the small state a
stopped stream needs to resume, without writing generated strain to disk.
`export_state()` returns a picklable snapshot of the bit-generator state and the
chunk counter (plus the filter memory for the FIR simulator), and
`import_state(snapshot)` on an identically configured simulator restores it by
regenerating the previous window from the bit-generator state:

```python
snapshot = simulator.export_state()
# Write snapshot with pickle or JSON, then stop the process.
resumed = ColoredNoiseSimulator(
    psd_file="example_psd.txt",
    detectors=["H1", "L1"],
    sampling_frequency=4096.0,
)
resumed.import_state(snapshot)
next_chunk = resumed.generate(4.0, 4096.0, ["H1"])
```

A stream stopped after any chunk and resumed this way is bit-identical to an
uninterrupted run; the resume boundary is not restricted to the first chunk.

## See also

- **`ParallelAdapter`** (`gwmock_noise.parallel`) — parallelize
  independent-detector simulators; read the API docs for backend limitations on
  correlated simulators.
- **`open_stream` / `take`** — public helpers for opening and collecting
  stateful chunk streams; see `gwmock_noise.simulators` in the
  [API reference](../api/index.md).
- **[Custom simulators](custom_simulators.md)** — implement the `NoiseSimulator`
  protocol so `open_stream(...)` can consume your simulator without
  package-internal hooks.
- **Diagnostics** (`gwmock_noise.diagnostics`) — PSD estimation and simple
  statistical checks for validating realizations.
