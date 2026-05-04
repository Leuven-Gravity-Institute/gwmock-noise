# Noise simulation

This page shows how to run a basic noise simulation with `gwmock-noise`, using
the simulator interface and the configuration-file-based CLI.

## Quick example (CLI, TOML)

Create a configuration file, for example:

```toml
# examples/noise_config_example.toml
detectors = ["H1", "L1"]
duration = 4.0
sampling_frequency = 4096.0

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

| Field                   | Type        | Description                                                        |
| ----------------------- | ----------- | ------------------------------------------------------------------ |
| `detectors`             | list[str]   | Names of detectors to simulate (for example `H1`, `L1`)            |
| `duration`              | float       | Duration of the realization in seconds (`> 0`)                     |
| `sampling_frequency`    | float       | Sampling frequency in Hz (`> 0`)                                   |
| `output.directory`      | path        | Output directory for generated files                               |
| `output.prefix`         | str         | Prefix for output file names                                       |
| `output.format`         | str         | Artifact format written by `run(config)`: `npy` (default) or `gwf` |
| `output.gps_start`      | float       | GPS start time used for timestamped formats such as `gwf`          |
| `output.channel_prefix` | str         | Channel-name prefix for `gwf` output (default: `MOCK`)             |
| `seed`                  | int or null | Optional random seed for reproducibility                           |

For integration with the upstream `gwmock` package, the same structure can be
nested under a `noise` key inside a larger configuration file. In that case the
CLI still works; it automatically looks for a `noise` section if present.

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

The upstream `gwmock` package is expected to import and compose
`gwmock_noise.NoiseConfig` into its own configuration model and to drive a noise
simulator that implements the `gwmock_noise.BaseNoiseSimulator` interface.

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
