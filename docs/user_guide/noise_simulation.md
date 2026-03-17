# Noise simulation

This page shows how to run a basic noise simulation with `gwsim-noise`, using
the simulator interface and the configuration-file based CLI.

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
gwsim-noise simulate examples/noise_config_example.toml
```

This will create one JSON metadata file per detector in the configured output
directory (for example `output/noise_H1.json`). In this first milestone the
files contain configuration metadata only; later versions will write full noise
time series.

## Configuration

Noise simulations are configured with a Pydantic model
`gwsim_noise.NoiseConfig`. When using the CLI, the configuration is loaded from
TOML, YAML, or JSON into the same model.

Supported top-level fields:

| Field                | Type        | Description                                             |
| -------------------- | ----------- | ------------------------------------------------------- |
| `detectors`          | list[str]   | Names of detectors to simulate (for example `H1`, `L1`) |
| `duration`           | float       | Duration of the realization in seconds (`> 0`)          |
| `sampling_frequency` | float       | Sampling frequency in Hz (`> 0`)                        |
| `output.directory`   | path        | Output directory for generated files                    |
| `output.prefix`      | str         | Prefix for output file names                            |
| `seed`               | int or null | Optional random seed for reproducibility                |

For integration with the upstream `gwsim` package, the same structure can be
nested under a `noise` key inside a larger configuration file. In that case the
CLI still works; it automatically looks for a `noise` section if present.

## Programmatic usage

You can also construct configurations and run the simulator directly from
Python:

```python
from pathlib import Path

from gwsim_noise import DefaultNoiseSimulator, NoiseConfig, OutputConfig

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

The upstream `gwsim` package is expected to import and compose
`gwsim_noise.NoiseConfig` into its own configuration model and to drive a noise
simulator that implements the `gwsim_noise.BaseNoiseSimulator` interface.
