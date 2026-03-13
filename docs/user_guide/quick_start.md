# Quick Start

Welcome to **gwsim_noise**! This package provides tools for simulating gravitational wave detector noise.

## Getting Started

### 1. Installation

Install the package from PyPI:

```bash
pip install gwsim-noise
```

Or using `uv`:

```bash
uv pip install gwsim-noise
```

### 2. Set Up Your Development Environment

If you're contributing or developing:

```bash
# Clone the repository
git clone git@github.com:isaac-cf-wong/gwsim_noise.git
cd gwsim_noise

# Create a virtual environment
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with all dependencies
uv pip install -e ".[dev,docs,test]"
```

### 3. Install Pre-commit Hooks and Commit Message Validation

The project uses pre-commit hooks and commitlint to maintain code quality and follow the [conventional commits](https://www.conventionalcommits.org/) standard.

**Install commitlint dependencies:**

```bash
npm install
```

**Install pre-commit hooks:**

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

### 4. Run Tests

Verify everything works:

```bash
pytest
```

You should see all tests pass with coverage reporting.

### 5. Build Documentation

Generate and view the documentation:

```bash
zensical serve
```

Open [http://localhost:8000](http://localhost:8000) in your browser to see the docs.

## Development Workflow

### Code Structure

Your package code is in `src/gwsim_noise/`:

- `src/gwsim_noise/__init__.py`: Package initialization and exports
- `src/gwsim_noise/__main__.py`: Entry point for `python -m gwsim_noise`
- Add your modules in subdirectories as needed

### Writing Code

1. **Follow the Style**: Code is automatically formatted with Ruff (120 char line length)
2. **Add Type Hints**: Use modern Python typing
3. **Write Tests**: Add tests in `tests/` directory
4. **Update Docs**: Add docstrings for API documentation

### Example: Running the Sample Code

Try the included example:

```bash
# Run as a module
python -m gwsim_noise

# Or use the CLI
gwsim-noise --help
```

### Building and Publishing

When ready to release:

1. Run tests and checks locally: `pytest && pre-commit run --all-files`
2. Create a git tag for the new version: `git tag v1.0.0` (replace with your version)
3. Push the tag to the remote repository: `git push origin v1.0.0`
4. The GitHub Actions workflow will automatically create a release and publish to PyPI

## Key Configurations

- **pyproject.toml**: Package metadata, dependencies, tool configs
- **.pre-commit-config.yaml**: Pre-commit hooks for code quality
- **cliff.toml**: Configuration for automatic changelog generation using git-cliff
- **zensical.toml**: Documentation configuration
- **.github/workflows/**: CI/CD pipelines

## Next Steps

- Explore the [API documentation](../api/index.md)
- Check out the [troubleshooting guide](../dev/troubleshooting.md)
- Read the [contributing guidelines](../contributing.md)
