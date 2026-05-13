# Quick start

This page is the shortest path from install to a working simulation. For install
options and dependency groups, see [Installation](installation.md). For every
`NoiseConfig` field and output format, see
[Noise simulation](noise_simulation.md).

!!! tip "Just want a quick answer?" See [Minimal Usage](minimal_usage.md) for
copy-paste CLI and Python snippets organised by what you want to do.

## 1. Install from PyPI

```bash
uv venv --python 3.12
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install gwmock-noise
```

Optional integrations (see `pyproject.toml` `[project.optional-dependencies]`):

```bash
uv pip install "gwmock-noise[gwpy]"    # GWpy adapters
uv pip install "gwmock-noise[frame]" # GWF output (GWpy + LALSuite stack)
uv pip install "gwmock-noise[gengli]" # gengli-backed blip glitches
```

## 2. Run the CLI

From the repository root (or with your own config path):

```bash
gwmock-noise simulate examples/noise_config_example.toml
```

Use `gwmock-noise --help` and `gwmock-noise simulate --help` for Typer/Rich
usage.

To build a gengli population file from a GravitySpy CSV export:

```bash
gwmock-noise build-blip-glitch-table --gravity-spy-csv gravity_spy.csv --out glitches.h5
```

## 3. Call the Python API

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

`DefaultNoiseSimulator().run(config)` validates the config, builds the
appropriate backend (white noise by default; PSD/CSD, lines, glitches, Schumann,
etc. when configured), and writes NumPy strain files plus JSON sidecars unless
`output.format` requests GWF output.

## 4. Clone and develop

Contributors should clone the repository and sync **uv dependency groups** (the
project does not publish a `[dev]` or `[docs]` extra on PyPI):

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-noise.git
cd gwmock-noise
uv venv --python 3.12
source .venv/bin/activate
uv sync --group dev --group docs
```

Install Git hooks (from the repo root):

```bash
uv run pre-commit install
```

## 5. Tests and docs

```bash
uv run pytest
uv run zensical serve   # local documentation preview
```

## Next steps

- [Noise simulation](noise_simulation.md) — configuration tables, nested `noise`
  configs, and `gwf` output
- [API reference](../api/index.md) — all public modules
- [Testing](testing.md) — markers, coverage, and CI defaults
- [Contributing](../contributing.md) — pull requests and conventions
