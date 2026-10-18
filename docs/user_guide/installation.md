# Installation

We recommend using `uv` to manage virtual environments for installing
`gwmock-noise`.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

## Requirements

- Python 3.12 or higher
- Operating System: Linux, macOS, or Windows

<!-- prettier-ignore-start -->
!!! note
    The package is built and tested against Python 3.12–3.15. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility:
    `uv venv --python 3.12` (replace `3.12` with your preferred version in the 3.12–3.15 range).
    On Python 3.15, h5py publishes no wheel yet, so the `hdf5` extra (and anything that pulls it in) may
    not install there; everything else does.
<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install `gwmock-noise` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwmock-noise
```

### Optional dependencies (extras)

Published wheels expose these optional dependency groups:

| Extra    | Purpose                                                                               |
| -------- | ------------------------------------------------------------------------------------- |
| `hdf5`   | HDF5 files: `output.format = "hdf5"`, blip population files, HDF5 SNR tables (`h5py`) |
| `gwpy`   | GWpy-based helpers (`GWpyAdapter`, parts of the GWF stack)                            |
| `frame`  | GWF frame writing (`FrameWriter`; pulls `gwpy` and `lalsuite`)                        |
| `gengli` | gengli-backed blip glitches (`GengliBlipGlitch`)                                      |
| `gwosc`  | Real noise from GWOSC (`GwoscNoiseFetcher`; pulls `gwpy` and `gwosc`)                 |

A feature whose extra is missing raises `ModuleNotFoundError` naming the extra
to install. The default output format, `npy`, needs none of them.

```bash
uv pip install "gwmock-noise[hdf5]"
uv pip install "gwmock-noise[gwpy]"
uv pip install "gwmock-noise[frame]"
uv pip install "gwmock-noise[gengli]"
uv pip install "gwmock-noise[gwosc]"
```

Development and documentation tooling **are not extras**; they live in uv
`[dependency-groups]` in `pyproject.toml` (`dev`, `docs`, `build`, `release`).
Use `uv sync --group dev` (and `--group docs` when building the site) from a git
checkout.

## Install from Source

For the latest development version:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-noise.git
cd gwmock-noise
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Development installation

To install the package editable with lint/test/doc dependencies:

```bash
git clone git@github.com:Leuven-Gravity-Institute/gwmock-noise.git
cd gwmock-noise

uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync --group dev --group docs

uv run prek install
```

## Verify installation

Check that the CLI and import path work:

```bash
gwmock-noise --help
```

```bash
python -c "import gwmock_noise; print(gwmock_noise.__version__)"
```

## Core runtime dependencies

The default wheel installs:

- **numpy** — arrays, FFTs, RNG
- **pydantic** — configuration models
- **scipy** — signal processing and numerics
- **pyyaml** — YAML config loading
- **typer** — CLI (`gwmock-noise` entry point)

## Getting help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://github.com/Leuven-Gravity-Institute/gwmock-noise/issues)
3. Open a new issue with your OS, Python version, full traceback, and minimal reproduction steps

<!-- prettier-ignore-end -->
