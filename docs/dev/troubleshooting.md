# Troubleshooting

This guide covers common issues you might encounter when working on
**gwmock-noise** and how to resolve them.

## Setup Issues

### Pre-commit Hook Installation Fails

**Problem:** `pre-commit install` returns an error or hooks don't run on commit.

**Solutions:**

<!-- prettier-ignore-start -->

1. Ensure you're in the project root directory
2. Verify Python virtual environment is activated
3. Reinstall pre-commit:

    ```bash
    pip uninstall pre-commit
    pip install pre-commit
    pre-commit install
    ```

4. Check if `.git` directory exists (must be a git repository)
5. Try running manually: `pre-commit run --all-files`

<!-- prettier-ignore-end -->

### Commit message conventions

Pre-commit here runs on staged files, not on the commit message. Still follow
[Conventional Commits](https://www.conventionalcommits.org/) for commits and PR
titles; see [Contributing](../contributing.md).

### Virtual Environment Issues

**Problem:** Packages can't be found or dependencies conflict.

**Solutions:**

<!-- prettier-ignore-start -->

1. Create a fresh virtual environment:

    ```bash
    rm -rf .venv
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

2. Upgrade pip:

    ```bash
    python -m pip install --upgrade pip
    ```

3. Install dependencies (dev and docs tooling live in **uv dependency groups**,
   not `[project.optional-dependencies]`; see
   [Installation](../user_guide/installation.md)):

    ```bash
    uv sync --group dev --group docs
    ```

4. Verify installation:

    ```bash
    python -c "import gwmock_noise; print(gwmock_noise.__version__)"
    ```

<!-- prettier-ignore-end -->

### Python Version Mismatch

**Problem:** `python -m venv .venv` fails or tests don't run with wrong Python
version.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check your Python version:

    ```bash
    python --version
    ```

2. Ensure Python 3.12 or higher is installed (the package targets 3.12–3.14;
   see `requires-python` in `pyproject.toml`)
3. Use a supported Python version when creating the venv:

    ```bash
    python3.12 -m venv .venv
    ```

4. Or use uv for version management:

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```

<!-- prettier-ignore-end -->

## Testing Issues

### Pytest Fails to Collect Tests

**Problem:** `pytest` returns "no tests collected" or import errors.

**Solutions:**

<!-- prettier-ignore-start -->

1. Verify test file naming: Must be `test_*.py` or `*_test.py`
2. Verify test function naming: Must start with `test_`
3. Run from the repository root; `pyproject.toml` sets `testpaths = "tests"`
   and `pythonpath = ["src"]` for discovery
4. Run pytest with verbose output:

    ```bash
    uv run pytest -vv
    ```

5. Check test discovery:

    ```bash
    uv run pytest --collect-only
    ```

<!-- prettier-ignore-end -->

### Import Errors in Tests

**Problem:** Tests can't import `gwmock_noise` or its submodules.

**Solutions:**

<!-- prettier-ignore-start -->

1. Install the project and dev dependencies (editable install is implied by
   `uv sync` from the repo root):

    ```bash
    uv sync --group dev
    ```

2. Verify package layout: Python sources live under `src/gwmock_noise/`
3. Check `pyproject.toml` has correct `packages` configuration
4. Run from project root directory
5. Verify `__init__.py` exists in package directory

<!-- prettier-ignore-end -->

### Coverage Report Issues

**Problem:** Coverage report shows 0% or missing files.

**Solutions:**

<!-- prettier-ignore-start -->

1. Run pytest with coverage:

    ```bash
    uv run pytest --cov=src --cov-report=html
    ```

2. Check `[tool.coverage.*]` and `[tool.pytest.ini_options]` in `pyproject.toml`
3. Ensure source files have proper imports
4. Verify test files import from `src/` layout correctly

<!-- prettier-ignore-end -->

## Pre-commit Hook Issues

### Hooks Running Too Slowly

**Problem:** Pre-commit takes a very long time or times out.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check which hooks are slow:

    ```bash
    pre-commit run --all-files --verbose
    ```

2. Consider excluding large files:

    ```yaml
    exclude: |
      (?x)^(
        large_data_file.csv|
        node_modules/
      )$
    ```

3. Run specific hooks:

    ```bash
    pre-commit run ruff --all-files  # Just ruff
    ```

<!-- prettier-ignore-end -->

### Formatting Changes After Commit

**Problem:** Pre-commit auto-fixes files, but you didn't expect it.

**Solutions:**

<!-- prettier-ignore-start -->

1. This is normal behavior - review the changes
2. Stage the new changes:

    ```bash
    git add .
    git commit -m "feat: restage after pre-commit"  # Use a valid Conventional Commit title
    ```

3. Modify tool settings if behavior is unwanted (in `pyproject.toml`)
4. Disable specific hooks temporarily:

    ```bash
    SKIP=ruff pre-commit run --all-files
    ```

<!-- prettier-ignore-end -->

### "Unstaged Changes" After Running Hooks

**Problem:** Pre-commit modified files but they're not staged.

**Solutions:**

<!-- prettier-ignore-start -->

1. This is expected - review changes:

    ```bash
    git diff
    ```

2. Stage the changes:

    ```bash
    git add .
    ```

3. Try committing again
4. Or use `git add -A` to stage all changes before commit

<!-- prettier-ignore-end -->

## CI/CD Issues

### CI Workflow Fails on Push

**Problem:** GitHub Actions workflow fails unexpectedly.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check the Actions tab in GitHub for error details
2. Run tests locally first:

    ```bash
    uv run pytest
    pre-commit run --all-files
    ```

3. Common causes:

    - Dependency installation failed: Compare with CI (`uv sync --group dev --frozen` in `.github/workflows/ci.yml`)
    - Python version mismatch: Verify Python versions in workflow matrix
    - Missing dependencies: Add to `pyproject.toml`
    - Pre-commit failures: Fix locally first

4. Re-run failed jobs from GitHub Actions UI

<!-- prettier-ignore-end -->

### CodeQL Analysis Takes Too Long

**Problem:** CI is slow due to CodeQL analysis.

**Solutions:**

1. This is normal (~2-3 minutes per run)
2. To disable CodeQL:
    - Remove or edit the workflow in `.github/workflows/codeql.yml`
    - Keep Bandit in pre-commit for basic security
3. Or assess whether CodeQL is worth the CI time for this repository
4. CodeQL provides value for security-critical projects

### Release Workflow Fails

**Problem:** Release or publish workflow doesn't work.

**Solutions:**

1. Verify tag format matches pattern: `v[0-9]+.[0-9]+.[0-9]+*`
    - Good: `v1.0.0`, `v1.2.3-alpha`
    - Bad: `1.0.0`, `release-1.0.0`
2. Check CI workflow passed first (required by release workflow)
3. Verify git-cliff configuration in `cliff.toml`
4. For publishing:
    - The publish workflow uses OIDC (`id-token: write`); configure a **trusted
      publisher** on PyPI for this GitHub repo, or verify secrets if you use
      token-based publishing
    - Inspect
      [`.github/workflows/publish.yml`](https://github.com/Leuven-Gravity-Institute/gwmock-noise/blob/main/.github/workflows/publish.yml)
      for the exact steps

## Documentation Issues

### Zensical Site Won't Build

**Problem:** `uv run zensical serve` or `uv run zensical build` fails.

**Solutions:**

<!-- prettier-ignore-start -->

1. Verify Zensical and doc dependencies are installed:

    ```bash
    uv sync --group docs
    ```

2. Check `zensical.toml` syntax (must be valid TOML)
3. Verify markdown files exist and paths are correct
4. Check for circular includes or missing includes
5. Run with verbose output:

    ```bash
    uv run zensical build --verbose
    ```

<!-- prettier-ignore-end -->

### Documentation Not Updating on GitHub Pages

**Problem:** You pushed changes but the docs aren't updated online.

**Solutions:**

1. Verify GitHub Pages is enabled:
    - Go to repository Settings → Pages
    - Under "Build and deployment", select "GitHub Actions" as the source
    - This allows the documentation workflow to deploy directly
2. Check documentation workflow ran successfully:
    - Go to Actions tab
    - Look for the **Documentation** workflow
      (`.github/workflows/documentation.yml`)
3. Verify changes were pushed to the correct branch
4. Wait 1-2 minutes for Pages to build
5. Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)
6. Check browser cache isn't serving old version

### API Documentation Not Generating

**Problem:** API reference pages are empty or show errors.

**Solutions:**

<!-- prettier-ignore-start -->

1. Verify docstrings are present on public APIs under `src/gwmock_noise/`

2. Ensure the docs dependency group is installed (includes `mkdocstrings-python`):

    ```bash
    uv sync --group docs
    ```

3. Verify navigation in `zensical.toml` includes the **API** section and that
   `docs/api/index.md` uses mkdocstrings directives (for example `::: gwmock_noise`)
4. Rebuild the site locally:

    ```bash
    uv run zensical build --verbose
    ```

5. Ensure re-exported symbols in `src/gwmock_noise/__init__.py` match what you
   expect in the API reference

<!-- prettier-ignore-end -->

## Dependencies & Package Issues

### "ModuleNotFoundError" When Running CLI

**Problem:** Running `gwmock-noise --help` fails with module not found.

**Solutions:**

<!-- prettier-ignore-start -->

1. Install the package in development mode from the repo root:

    ```bash
    uv sync --group dev
    ```

2. Verify entry points in `pyproject.toml`:

    ```toml
    [project.scripts]
    gwmock-noise = "gwmock_noise.cli.main:app"
    ```

3. Check the specified Typer app exists and is callable
4. Remember PyPI distribution name vs import path:

    - Distribution / CLI script name: `gwmock-noise` (hyphen; `[project]` `name`
      and `[project.scripts]` key)
    - Import package: `gwmock_noise` (underscore; directory under `src/`)

<!-- prettier-ignore-end -->

### Dependency Conflicts

**Problem:** `pip install` fails with conflict messages.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check Python version:

    ```bash
    python --version
    ```

2. Create fresh virtual environment:

    ```bash
    rm -rf .venv && python -m venv .venv
    source .venv/bin/activate
    ```

3. Upgrade pip:

    ```bash
    python -m pip install --upgrade pip
    ```

4. Install with verbose output to see conflict:

    ```bash
    uv sync --group dev --group docs --verbose
    ```

5. Check `pyproject.toml` for overly restrictive version constraints

<!-- prettier-ignore-end -->

### Newer Version of Tool Breaks Things

**Problem:** Pre-commit hooks or tools updated and now fail.

**Solutions:**

<!-- prettier-ignore-start -->

1. Check what changed:

    ```bash
    pre-commit autoupdate --dry-run
    ```

2. Update individual tool:

    ```bash
    pre-commit autoupdate --repo https://github.com/pre-commit/pre-commit-hooks
    ```

3. Test changes:

    ```bash
    pre-commit run --all-files
    ```

4. Pin to known-good version in `.pre-commit-config.yaml`:

    ```yaml
    rev: v1.0.0 # Specific version instead of latest
    ```

<!-- prettier-ignore-end -->

## Getting Help

If you encounter issues not listed here:

<!-- prettier-ignore-start -->

1. **Check existing issues**: Search
   [gwmock-noise issues](https://github.com/Leuven-Gravity-Institute/gwmock-noise/issues)
2. **Review logs carefully**: Error messages usually point to the root cause
3. **Search documentation**: Published docs are at
   [https://leuven-gravity-institute.github.io/gwmock-noise/](https://leuven-gravity-institute.github.io/gwmock-noise/)
4. **Try minimal reproduction**: Isolate the problem to a single file/command
5. **Ask for help**: Open an
   [issue](https://github.com/Leuven-Gravity-Institute/gwmock-noise/issues/new/choose) with:
    - Your environment (Python version, OS)
    - Steps to reproduce
    - Full error message/logs
    - What you've already tried

<!-- prettier-ignore-end -->
