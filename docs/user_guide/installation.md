# Installation

<!-- prettier-ignore-start -->

!!!warning
    Adjust the content as needed for your specific package.

<!-- prettier-ignore-end -->

We recommend using `uv` to manage virtual environments for installing
`gwsim-noise`.

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
!!!note
    The package is built and tested against Python 3.12-3.14. When creating a virtual environment with `uv`,
    specify the Python version to ensure compatibility:
    `uv venv --python 3.12` (replace `3.12` with your preferred version in the 3.12-3.14 range).
    This avoids potential issues with unsupported Python versions.

<!-- prettier-ignore-end -->

## Install from PyPI

The recommended way to install `gwsim-noise` is from PyPI:

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install gwsim-noise
```

### Optional Dependencies

For development or specific features:

```bash
# Development dependencies (testing, linting, etc.)
uv pip install gwsim-noise[dev]

# Documentation dependencies
uv pip install gwsim-noise[docs]

# All dependencies
uv pip install gwsim-noise[dev,docs]
```

## Install from Source

For the latest development version:

```bash
git clone git@github.com:isaac-cf-wong/gwsim_noise.git
cd gwsim_noise
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install .
```

### Development Installation

To set up for development:

```bash
git clone git@github.com:isaac-cf-wong/gwsim_noise.git
cd gwsim_noise

# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install ".[dev]"

# Install the commitlint dependencies
npm install

# Install pre-commit hooks
pre-commit install
pre-commit install --hook-type commit-msg
```

## Verify Installation

Check that `gwsim-noise` is installed correctly:

```bash
gwsim-noise --help
```

```bash
python -c "import gwsim_noise; print(gwsim_noise.__version__)"
```

## Dependencies

### Core Dependencies

- **typer**: CLI framework

## Getting Help

<!-- prettier-ignore-start -->

1. Check the [troubleshooting guide](../dev/troubleshooting.md)
2. Search existing [issues](https://github.com/isaac-cf-wong/gwsim_noise/issues)
3. Create a new issue with:
    - Your operating system and Python version
    - Full error message
    - Steps to reproduce the problem

<!-- prettier-ignore-end -->
