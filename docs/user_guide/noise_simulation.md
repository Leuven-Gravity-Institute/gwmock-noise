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

| Field                | Type                   | Description                                                                                       |
| -------------------- | ---------------------- | ------------------------------------------------------------------------------------------------- |
| `detectors`          | `list[str]`            | Names of detectors to simulate (for example `H1`, `L1`)                                           |
| `duration`           | `float`                | Duration of the realization in seconds (`> 0`)                                                    |
| `sampling_frequency` | `float`                | Sampling frequency in Hz (`> 0`)                                                                  |
| `components`         | `list[str \| mapping]` | Ordered simulator components; each entry is a simulator name or mapping                           |
| `output.directory`   | `path`                 | Output directory for generated files                                                              |
| `output.prefix`      | `str`                  | Prefix for output file names                                                                      |
| `output.format`      | `str`                  | Artifact format written by `run(config)`: `npy` (default) or `gwf`                                |
| `output.gps_start`   | `float`                | GPS start time used for timestamped formats such as `gwf`                                         |
| `output.channel`     | `str`                  | Channel name suffix for `gwf` output, assembled as `{detector}:{channel}` (default: `MOCK_NOISE`) |
| `output.channels`    | `dict[str, str]`       | Per-detector full channel names (e.g. `{"H1": "H1:STRAIN_NOISE"}`); overrides `channel` when set  |
| `seed`               | `int` or `null`        | Optional random seed for reproducibility                                                          |

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
and bundled Einstein Telescope preset names. The built-in presets are
`ET_D_psd`, `ET_10_HF_psd`, `ET_10_full_cryo_psd`, `ET_15_HF_psd`,
`ET_15_full_cryo_psd`, `ET_20_HF_psd`, and `ET_20_full_cryo_psd`.

The upstream `gwmock` package is expected to import and compose
`gwmock_noise.NoiseConfig` into its own configuration model and to drive a noise
simulator that implements the `gwmock_noise.BaseNoiseSimulator` interface.

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
