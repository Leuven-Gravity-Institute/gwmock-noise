# Adding a new simulation type

This guide is for contributors who want to add a new built-in simulation backend
to `gwmock-noise`.

## Start here

Before writing code, decide which of these two contribution paths fits your
work:

1. **Standalone simulator** — implement the public `NoiseSimulator` protocol so
   the simulator works with `open_stream(...)` and other public helpers, but do
   not wire it into `NoiseConfig` or `DefaultNoiseSimulator`.
2. **Built-in simulator** — implement a concrete `ConfigurableNoiseSimulator`.
   It will be discovered automatically and can be selected by
   `DefaultNoiseSimulator().run(config)`.

If you only need a reusable simulator class, start with the standalone path.
Choose the built-in path only when the new backend should become part of the
package's default configuration-driven workflow.

## Where code should live

Use the package layout by responsibility:

| Area                            | What belongs there                             |
| ------------------------------- | ---------------------------------------------- |
| `src/gwmock_noise/config/`      | Configuration schema and config loading only   |
| `src/gwmock_noise/gaussian/`    | Gaussian-noise-adjacent runtime models/helpers |
| `src/gwmock_noise/glitches/`    | Glitch models and glitch-specific helpers      |
| `src/gwmock_noise/gwosc/`       | Real-noise fetching and GWOSC-specific models  |
| `src/gwmock_noise/simulators/`  | Simulator implementations and wrappers         |
| `src/gwmock_noise/diagnostics/` | Validation/statistical analysis helpers        |

For a new simulation type, the simulator implementation itself should normally
go in **`src/gwmock_noise/simulators/<name>.py`**.

## Minimal implementation path

For a new standalone simulator:

1. Add `src/gwmock_noise/simulators/<name>.py`.
2. Implement the public `NoiseSimulator` protocol described in
   [Custom simulators](../user_guide/custom_simulators.md).
3. Export it from `src/gwmock_noise/simulators/__init__.py`.
4. Re-export it from `src/gwmock_noise/__init__.py` if it should be part of the
   main public API.
5. Add focused tests under `tests/`.
6. Add user-facing docs if the simulator is intended for external use.

Use this path when the simulator can be constructed directly in Python without
new config schema.

## Built-in implementation path

If the new simulation type should work through `NoiseConfig` and
`DefaultNoiseSimulator`, the usual steps are:

1. **Add the simulator implementation**

    Put the runtime logic in `src/gwmock_noise/simulators/<name>.py`.

2. **Implement the config-driven hook**

    Make the simulator a concrete subclass of
    `gwmock_noise.simulators.ConfigurableNoiseSimulator` and implement:
    - `simulator_name`
    - `from_component(cls, component, config)`

    The discovery registry only registers concrete subclasses of that base
    class, so helper functions, wrappers, and abstract mixins are ignored
    automatically.

3. **Wire it into default orchestration**

    No central registry edit is required. The class is discovered automatically.
    Put simulator-specific arguments in the component `options` payload, while
    shared runtime fields (`detectors`, `duration`, `sampling_frequency`,
    `output`, `seed`) stay on `NoiseConfig`.

4. **Export it**

    Add exports in `src/gwmock_noise/simulators/__init__.py` and, if
    appropriate, `src/gwmock_noise/__init__.py`.

5. **Document it**

    Add a usage example to the relevant page in `docs/user_guide/` and update
    the API docs if the public surface changed.

6. **Test it**

    Add:
    - unit tests for validation and constructor behavior
    - simulator tests for output shape, determinism, and metadata
    - streaming tests if `generate_stream(...)` maintains state
    - slow/statistical tests when spectral or distributional correctness matters

## What good tests look like here

Prefer tests that lock down behavior contributors are likely to break during
future refactors:

- config validation rejects invalid combinations early
- `generate(...)` returns the requested detectors and sample count
- seeded runs are deterministic
- metadata describes the active implementation clearly
- streaming output is continuous across chunk boundaries
- statistical/spectral properties match expectations within tolerance

If the simulator is expensive or depends on longer realizations, mark those
checks with `@pytest.mark.slow`.

## Documentation checklist

When a new built-in simulation type is added, update the docs in the same PR:

- `docs/user_guide/minimal_usage.md` for the shortest example
- `docs/user_guide/noise_simulation.md` for background and configuration details
- `docs/api/index.md` if a new public module should appear in the API reference

If the simulator is only a low-level helper or experimental surface, document it
where contributors will find it, but avoid adding unnecessary top-level user
guide content.

## Registration model

Built-in simulators are discovered automatically from the
`gwmock_noise.simulators` package. Only concrete subclasses of
`ConfigurableNoiseSimulator` are registered, which keeps helper modules and
wrappers out of the built-in selection path.

Users select built-in simulators by adding entries to `NoiseConfig.components`.
Each component is a `{simulator, ...options}` mapping, so new built-ins do not
require changes to the top-level config schema.

## Suggested workflow for contributors

1. Read [Contributing](../contributing.md) and [Code Quality](code_quality.md).
2. Read the nearest existing simulator in `src/gwmock_noise/simulators/` and
   copy its testing pattern.
3. Add the implementation first.
4. Add component wiring only inside the simulator class itself if the simulator
   should be part of the built-in `run(config)` path.
5. Add tests before polishing exports and docs.
6. Finish by running:

```bash
uv run ruff check .
uv run pytest
uv build
```
