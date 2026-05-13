# gwmock-noise

[![Python CI](https://github.com/Leuven-Gravity-Institute/gwmock-noise/actions/workflows/ci.yml/badge.svg)](https://github.com/Leuven-Gravity-Institute/gwmock-noise/actions/workflows/ci.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/Leuven-Gravity-Institute/gwmock-noise/main.svg)](https://results.pre-commit.ci/latest/github/Leuven-Gravity-Institute/gwmock-noise/main)
[![Documentation Status](https://github.com/Leuven-Gravity-Institute/gwmock-noise/actions/workflows/documentation.yml/badge.svg)](https://leuven-gravity-institute.github.io/gwmock-noise/)
[![codecov](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-noise/graph/badge.svg?token=COF8341N60)](https://codecov.io/gh/Leuven-Gravity-Institute/gwmock-noise)
[![PyPI Version](https://img.shields.io/pypi/v/gwmock-noise)](https://pypi.org/project/gwmock-noise/)
[![Python Versions](https://img.shields.io/pypi/pyversions/gwmock-noise)](https://pypi.org/project/gwmock-noise/)
[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DOI](https://zenodo.org/badge/1180562350.svg)](https://doi.org/10.5281/zenodo.20032221)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

A Python package for simulating gravitational-wave detector **instrumental**
noise (colored and correlated models, lines, glitches, optional
Schumann-resonance coupling, and CLI-driven batch runs).

## Documentation

Published docs (user guide and API reference):

[https://leuven-gravity-institute.github.io/gwmock-noise/](https://leuven-gravity-institute.github.io/gwmock-noise/)

To build and preview locally (requires the `docs` dependency group):

```bash
uv sync --group docs
uv run zensical serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Quick start

### Install

```bash
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install gwmock-noise
```

Optional extras (declared in `pyproject.toml`):

- `gwmock-noise[gwpy]` — GWpy-based helpers (for example `GWpyAdapter`)
- `gwmock-noise[frame]` — GWF frame output (`FrameWriter` / GWpy GWF stack)
- `gwmock-noise[gengli]` — gengli-backed blip glitches (`GengliBlipGlitch`)

### CLI

Create a TOML (or YAML/JSON) config and run:

```bash
gwmock-noise simulate examples/noise_config_example.toml
```

To turn a GravitySpy CSV export into a gengli population file:

```bash
gwmock-noise build-blip-glitch-table --gravity-spy-csv gravity_spy.csv --out glitches.h5
```

See [examples/noise_config_example.toml](examples/noise_config_example.toml) and
the [noise simulation](docs/user_guide/noise_simulation.md) guide for fields and
output layout.

### Python API

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
for detector, path in result.output_paths.items():
    print(detector, "->", path)
```

For PSD/CSD-driven colored noise, spectral lines, glitches, and streaming
behavior, use the same `NoiseConfig` fields described in the user guide and API
reference. For stateful continuation across chunk boundaries, prefer the public
`open_stream(...)` helper over reseeding separate runs.

The gengli backend plugs into the same `glitches=` surface:

```python
from pathlib import Path

from gwmock_noise import (
    DefaultNoiseSimulator,
    GengliBlipGlitch,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    OutputConfig,
)

config = NoiseConfig(
    detectors=["L1"],
    duration=8.0,
    psd_file=Path("noise_psd.txt"),
    glitches=[
        GengliBlipGlitch.from_population_file(
            "glitches.h5",
            rate=0.25,
            psd_file=Path("noise_psd.txt"),
            amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        )
    ],
    output=OutputConfig(directory=Path("output"), prefix="gengli"),
)
DefaultNoiseSimulator().run(config)
```

`NoiseConfig.psd_file` also accepts bundled Einstein Telescope preset names, so
you can use `psd_file="ET_10_full_cryo_psd"` without managing the PSD file
yourself. Available presets are `ET_D_psd`, `ET_10_HF_psd`,
`ET_10_full_cryo_psd`, `ET_15_HF_psd`, `ET_15_full_cryo_psd`, `ET_20_HF_psd`,
and `ET_20_full_cryo_psd`. Local paths and HTTP(S) URLs remain supported too
(for remote sources, use `.txt` or `.csv`).

## Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock-noise`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

### Requirements

- Python 3.12 or higher
- Operating System: Linux, macOS, or Windows

**Note:** The package is built and tested against Python 3.12–3.14. When
creating a virtual environment with `uv`, specify the Python version to ensure
compatibility: `uv venv --python 3.12` (replace `3.12` with your preferred
version in the 3.12–3.14 range). This avoids potential issues with unsupported
Python versions.

### Install from PyPI

The recommended way to install `gwmock-noise` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-noise
```

### Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-noise.git
cd gwmock-noise
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

#### Development Installation

To set up for development (linting, tests, and docs tooling live in **uv
dependency groups**, not `[project.optional-dependencies]`):

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-noise.git
cd gwmock-noise

# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev --group docs

# Install pre-commit hooks
uv run prek install
```

### Verify Installation

Check that `gwmock-noise` is installed correctly:

```bash
gwmock-noise --help
```

```bash
python -c "import gwmock_noise; print(gwmock_noise.__version__)"
```

## Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Release Schedule

Releases follow a fixed schedule: every Tuesday at 00:00 UTC, unless an emergent
bugfix is required. This ensures predictable updates while allowing flexibility
for critical issues. Users can view upcoming changes in the draft release on the
[GitHub Releases page](https://github.com/Leuven-Gravity-Institute/gwmock-noise/releases).

## Testing

Run the test suite:

```bash
uv run pytest
```

## License

This project is licensed under the 3-Clause BSD License - see the
[LICENSE](LICENSE) file for details.

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/Leuven-Gravity-Institute/gwmock-noise/issues/new/choose)
or contact the maintainers.
