# Testing

This guide describes how **gwmock-noise** runs its automated tests locally and
in CI, and how defaults in `pyproject.toml` differ from a bare `pytest`
invocation.

## Layout

```text
tests/
├── conftest.py       # Shared fixtures
├── test_*.py         # Unit and integration modules
└── ...
```

Tests import the package from `src/` via `pythonpath = ["src"]` in
`pyproject.toml`, so you do **not** need `PYTHONPATH=src` when using
`uv run pytest`.

## Running tests

From a checkout with the `dev` dependency group:

```bash
uv sync --group dev
uv run pytest
```

The project configures default `pytest` options (coverage, markers, reports). To
see the effective settings:

```toml
--8<-- "pyproject.toml:79:91"
```

### Useful overrides

```bash
# Skip default addopts (e.g. to run without coverage locally)
uv run pytest --override-ini="addopts="

# Integration tests only (when marked)
uv run pytest -m integration

# Single file or test
uv run pytest tests/test_config.py -q
uv run pytest tests/test_config.py::test_example -q
```

## Markers

Defined under `[tool.pytest.ini_options]`:

| Marker        | Meaning                                |
| ------------- | -------------------------------------- |
| `integration` | Slower or environment-dependent checks |
| `slow`        | Long-running numerical tests           |
| `unit`        | Fast offline tests                     |

By default, `addopts` excludes `integration` tests. Opt in explicitly when
needed.

## Coverage

Coverage is enabled through `addopts` (XML for CI, terminal missing lines
locally). Generate an HTML report when debugging gaps:

```bash
uv run pytest --override-ini="addopts=-m 'not integration' --cov src --cov-report=html"
```

Open `htmlcov/index.html` in a browser.

## CI

GitHub Actions runs the same toolchain (`uv`, `pytest`, Ruff, etc.) on pull
requests and the default branch. Keep new tests deterministic (fixed seeds) and
prefer `tmp_path` for filesystem side effects.

## Further reading

- [Quick start](quick_start.md) — clone and `uv sync --group dev`
- [Code quality](../dev/code_quality.md) — Ruff, Bandit, pre-commit
- [Pytest documentation](https://docs.pytest.org/)
- [Coverage.py](https://coverage.readthedocs.io/)
