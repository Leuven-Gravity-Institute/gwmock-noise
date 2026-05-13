# Minimal usage

This page answers "how do I do X?" for every common task. Each entry shows the
exact command or code to use — no explanations, no background.

---

## I want to run a basic noise simulation (CLI)

Produces **Gaussian white noise** — flat PSD, uncorrelated between detectors.

Create a config file (`config.toml`):

```toml
detectors = ["H1", "L1"]
duration = 4.0
sampling_frequency = 4096.0
[output]
directory = "./output"
prefix = "noise"
seed = 42
```

Run it:

```bash
gwmock-noise simulate config.toml
```

---

## I want to run a basic noise simulation (Python)

Produces **Gaussian white noise** — flat PSD, uncorrelated between detectors.

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
result = DefaultNoiseSimulator().run(config)
print(result.output_paths)
```

---

## I want to use a custom PSD (colored noise)

Produces **colored noise** with the exact PSD shape from your file — common for
realistic detector noise curves.

```python
config = NoiseConfig(
    detectors=["H1"],
    duration=8.0,
    components=[{"simulator": "colored", "psd_file": Path("my_psd.txt")}],
)
```

`psd_file` inside the colored component also accepts HTTP(S) URLs.

---

## I want to use an Einstein Telescope PSD preset

Produces **colored noise** matching a published Einstein Telescope design
sensitivity curve — no external PSD file needed.

```python
config = NoiseConfig(components=[{"simulator": "colored", "psd_file": "ET_D_psd"}])
```

Available presets: `ET_D_psd`, `ET_10_HF_psd`, `ET_10_full_cryo_psd`,
`ET_15_HF_psd`, `ET_15_full_cryo_psd`, `ET_20_HF_psd`, `ET_20_full_cryo_psd`.

---

## I want to generate correlated noise for a detector network

Produces **colored noise with cross-detector correlations** — the noise between
detectors follows your CSD specification (e.g. for Schumann or common
environmental coupling).

```python
config = NoiseConfig(
    detectors=["H1", "L1"],
    components=[
        {
            "simulator": "correlated",
            "psd_files": {"H1": Path("h1_psd.txt"), "L1": Path("l1_psd.txt")},
            "csd_files": {"H1-L1": Path("hl_csd.txt")},
        }
    ],
)
```

---

## I want to add spectral lines

Injects **narrow-band sinusoidal tones** (e.g. 60 Hz mains hum, calibration
lines) on top of the underlying noise.

```python
from gwmock_noise import SpectralLine

config = NoiseConfig(
    detectors=["H1"],
    duration=8.0,
    components=[
        {"simulator": "white"},
        {"simulator": "spectral_lines", "lines": [SpectralLine(frequency=60.0, amplitude=1e-22)]},
    ],
)
```

---

## I want to add glitches (transient noise bursts)

Injects **short-duration noise transients** (blips, scattered light) on top of
the underlying noise — mimics real detector glitches.

```python
from gwmock_noise import BlipGlitch, LogNormalAmplitudeDistribution

config = NoiseConfig(
    detectors=["H1"],
    duration=8.0,
    components=[
        {
            "simulator": "glitches",
            "models": [
                BlipGlitch(
                    rate=0.5,
                    width=0.01,
                    amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.5),
                )
            ],
        }
    ],
)
```

Other glitch kinds: `ScatteredLightGlitch`.

---

## I want to use gengli-backed blip glitches

Injects **realistic blip glitches** sampled from a GravitySpy population — more
physically accurate than the built-in parametric glitch model.

```bash
uv pip install "gwmock-noise[gengli]"
```

First build a population file from a GravitySpy CSV:

```bash
gwmock-noise build-blip-glitch-table --gravity-spy-csv gravity_spy.csv --out glitches.h5
```

Then use it in config:

```python
from gwmock_noise import GengliBlipGlitch, LogNormalAmplitudeDistribution

config = NoiseConfig(
    detectors=["L1"],
    duration=8.0,
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

---

## I want to include Schumann-resonance coupling

Produces **magnetically coupled correlated noise** from Earth's Schumann
resonances — adds frequency-dependent correlation between detector sites.

```python
from gwmock_noise import SchumannNoiseSimulator, SchumannParams

sim = SchumannNoiseSimulator(
    detectors=["H1", "L1"],
    detector_positions=[(43.6, 10.5), (46.5, 9.4)],  # lat, lon in degrees
    schumann_params=SchumannParams(),
)
```

---

## I want to use the AR (autoregressive) model

Produces **colored noise via an autoregressive filter** — an alternative to
FFT-based coloring, useful for low-latency or streaming applications.

```python
from gwmock_noise import ARNoiseSimulator, NoiseConfig

sim = ARNoiseSimulator(order=16, detectors=["H1"], duration=4.0, sampling_frequency=4096.0)
```

Can also be configured via `NoiseConfig.components`:

```python
config = NoiseConfig(
    detectors=["H1"],
    duration=4.0,
    components=[{"simulator": "ar", "psd_file": Path("psd.txt"), "order": 16}],
)
```

---

## I want to use time-varying PSD (non-stationary noise)

Produces **colored noise whose PSD changes over time** — simulates evolving
detector conditions (e.g. moving from science mode to injection mode).

```python
config = NoiseConfig(
    detectors=["H1"],
    duration=16.0,
    components=[
        {
            "simulator": "colored",
            "psd_schedule": [(0.0, Path("early_psd.txt")), (8.0, Path("late_psd.txt"))],
        }
    ],
)
```

---

## I want to combine multiple components in one simulation

Builds one simulation from an ordered list of peer components.

```python
from gwmock_noise import BlipGlitch, LogNormalAmplitudeDistribution, SpectralLine

config = NoiseConfig(
    detectors=["H1"],
    duration=8.0,
    components=[
        {"simulator": "colored", "psd_file": "ET_D_psd"},
        {"simulator": "spectral_lines", "lines": [SpectralLine(frequency=60.0, amplitude=1.0e-3)]},
        {
            "simulator": "glitches",
            "models": [
                BlipGlitch(
                    rate=0.25,
                    width=0.01,
                    amplitude_distribution=LogNormalAmplitudeDistribution(mean=0.5, std=0.0),
                )
            ],
        },
    ],
)
```

See `examples/noise_config_multiple_components.toml` for a runnable CLI example.

---

## I want to stream noise in chunks

Produces **noise one chunk at a time** with stateful continuation — chunks
concatenate seamlessly as if generated in a single call. Useful for
memory-constrained or real-time processing.

```python
from gwmock_noise import ColoredNoiseSimulator, open_stream, take

sim = ColoredNoiseSimulator(psd_file="psd.txt", detectors=["H1", "L1"], sampling_frequency=4096.0)
stream = open_stream(sim, chunk_duration=4.0, sampling_frequency=4096.0, detectors=["H1"])
chunk = next(stream)

# Or collect a specific duration
strain_dict = take(stream, total_duration=12.0, chunk_duration=4.0, sampling_frequency=4096.0)
```

---

## I want to get real detector noise from GWOSC

Fetches **real LIGO/Virgo strain data** from the Gravitational-Wave Open Science
Centre — use this when you need authentic detector noise for a specific GPS
interval.

```bash
uv pip install "gwmock-noise[gwosc]"
```

```python
from gwmock_noise.gwosc import GwoscNoiseConfig, GwoscNoiseFetcher

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000,
    gps_end=1135137000,
)
fetcher = GwoscNoiseFetcher(config)
clean_data = fetcher.fetch_clean()

# Or inspect segments before downloading
print(fetcher.clean_segments)

# Cache files locally to avoid repeated downloads
config = GwoscNoiseConfig(
    detectors=["H1"], gps_start=1135136000, gps_end=1135137000,
    cache_dir=Path("./gwosc_cache"),
)
```

---

## I want to filter out GW events or data-quality issues from GWOSC

Excludes **known gravitational-wave signals** and **data-quality flagged
segments** from real GWOSC data — leaves only clean noise for your analysis.

```python
from gwmock_noise.gwosc import FilterType, GwoscFilterConfig

config = GwoscNoiseConfig(
    detectors=["H1", "L1"],
    gps_start=1135136000, gps_end=1135137000,
    filters=GwoscFilterConfig(
        filter_types=[FilterType.HIGH_CONFIDENCE_GW],
        far_threshold=1.0,
        event_padding=16.0,
    ),
)
```

---

## I want to use real GWOSC noise with the simulator interface

Wraps **real GWOSC strain data** as a `NoiseSimulator` — makes it
interchangeable with synthetic simulators (works with `open_stream`, parallel
execution, etc.).

```python
from gwmock_noise import GwoscNoiseSimulator, open_stream
from gwmock_noise.gwosc import GwoscNoiseConfig

config = GwoscNoiseConfig(detectors=["H1"], gps_start=1135136000, gps_end=1135137000)
sim = GwoscNoiseSimulator(config)
strain = sim.generate(duration=1000.0, sampling_frequency=4096.0, detectors=["H1"])
```

---

## I want to run simulations in parallel

Runs **multiple independent noise simulations concurrently** across threads or
processes — accelerates batch jobs for parameter scans or Monte Carlo studies.

```python
from gwmock_noise import DefaultNoiseSimulator, NoiseConfig, ParallelAdapter

sim = DefaultNoiseSimulator()
parallel = ParallelAdapter(sim, backend="process")  # or "thread"
results = parallel.run([config_1, config_2, config_3])
```

Note: correlated simulators are not parallelizable.

---

## I want to check the quality of simulated noise

Validates that simulated noise **matches expected PSD shape** and passes basic
stationarity/gaussianity checks — use this to catch implementation bugs or
misconfiguration.

```python
from gwmock_noise import estimate_psd, compare_psd, run_diagnostics

freqs, psd = estimate_psd(strain, fs=4096.0)
result = compare_psd(freqs, psd, reference_freqs, reference_psd)
diagnostics = run_diagnostics(strain, fs=4096.0)
```

---

## I want to output GWF frame files instead of NumPy

Writes noise as **GWF frame files** (LIGO/Virgo standard format) instead of raw
NumPy arrays — required for downstream tools that expect frame-based input.

```python
config = NoiseConfig(
    detectors=["H1", "L1"],
    output=OutputConfig(directory=Path("output"), format="gwf", gps_start=1234567890),
)
```

Requires `uv pip install "gwmock-noise[frame]"`.

---

## I want to estimate PSD from real GWOSC data and use it for synthetic simulation

Computes a PSD from real detector data, then feeds it to a **colored noise
simulator** — lets you generate synthetic noise that mimics real detector
behaviour.

```python
from pathlib import Path

import numpy as np
from gwmock_noise.gwosc import GwoscNoiseConfig, GwoscNoiseFetcher
from gwmock_noise.diagnostics import estimate_psd
from gwmock_noise import ColoredNoiseSimulator

config = GwoscNoiseConfig(detectors=["H1"], gps_start=1135136000, gps_end=1135146000)
fetcher = GwoscNoiseFetcher(config)
clean = fetcher.fetch_clean()
freqs, psd = estimate_psd(clean["H1"][0].value, sampling_frequency=4096.0)

# ColoredNoiseSimulator.psd_file is a str | Path (or URL str), not in-memory arrays —
# write the (frequency, PSD) columns from estimate_psd(), then pass that path.
psd_path = Path("estimated_psd.txt")
np.savetxt(psd_path, np.column_stack((freqs, psd)))
sim = ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=4096.0)
```
