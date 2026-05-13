---
title: API Reference
description: Public API for the gwmock-noise package.
icon: material/api
---

Reference material is generated from Python docstrings via
[mkdocstrings](https://mkdocstrings.github.io/). The sections below mirror the
main import paths used in applications and in the CLI.

## Top-level package

Re-exports for common workflows (`NoiseConfig`, simulators, `ParallelAdapter`,
lazy diagnostics/output symbols).

::: gwmock_noise

## Configuration

Pydantic schema models and `load_config` for TOML/YAML/JSON.

::: gwmock_noise.config

## Gaussian noise models and helpers

Spectral-line definitions plus PSD reference helpers used by Gaussian-noise
configuration and simulators.

::: gwmock_noise.gaussian

## Glitch models

Runtime glitch model definitions, including phenomenological models and the
optional gengli-backed blip implementation.

::: gwmock_noise.glitches

## Simulators

Simulator implementations, the `NoiseSimulator` protocol,
`ConfigurableNoiseSimulator`, `SimulationResult`, and streaming helpers such as
`open_stream` and `take`.

::: gwmock_noise.simulators

## Parallel execution

`ParallelAdapter` runs independent-detector simulators across threads or
processes (with constraints for correlated backends).

::: gwmock_noise.parallel

## Diagnostics

PSD estimation/comparison and simple stationarity/gaussianity checks used in
tests and notebooks.

::: gwmock_noise.diagnostics

## Output adapters

Lazy exports for `GWpyAdapter` and `FrameWriter` (optional `gwpy` / `frame`
extras).

::: gwmock_noise.output

## GWOSC real-noise fetching

Models, segment filtering, data fetching, and a `NoiseSimulator` wrapper for
retrieving real detector strain data from GWOSC.

::: gwmock_noise.gwosc

::: gwmock_noise.simulators.real_noise

## Command-line interface

Typer application and the `simulate` command used by the `gwmock-noise` console
script.

::: gwmock_noise.cli.main

::: gwmock_noise.cli.simulate

## Package version

::: gwmock_noise.version
