"""Tests for the noise simulator interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators import (
    BaseNoiseSimulator,
    ColoredNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    NoiseSimulator,
    SimulationResult,
)


class DuckNoiseSimulator:
    """Minimal duck-typed simulator for protocol checks."""

    def __init__(self) -> None:
        """Set the protocol-required attributes."""
        self.duration = 1.0
        self.sampling_frequency = 1024.0
        self.detectors = ["H1"]
        self.seed = 7

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Return minimal per-detector arrays."""
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        return {detector: np.zeros(1) for detector in detectors}

    @property
    def metadata(self) -> dict[str, Any]:
        """Return representative metadata."""
        return {"implementation": "duck"}


def test_default_simulator_run(tmp_path: Path) -> None:
    """DefaultNoiseSimulator creates output files and returns result."""
    config = NoiseConfig(
        detectors=["H1", "L1"],
        duration=4.0,
        output=OutputConfig(directory=tmp_path, prefix="test"),
    )
    simulator = DefaultNoiseSimulator()
    result = simulator.run(config)

    assert isinstance(result, SimulationResult)
    assert result.config is config
    assert set(result.output_paths.keys()) == {"H1", "L1"}
    assert (tmp_path / "test_H1.json").exists()
    assert (tmp_path / "test_L1.json").exists()

    content = (tmp_path / "test_H1.json").read_text()
    assert "H1" in content
    assert "4.0" in content
    assert "4096" in content


def test_simulation_result_attributes() -> None:
    """SimulationResult has expected attributes for upstream consumers."""
    config = NoiseConfig()
    result = SimulationResult(output_paths={"H1": Path("/tmp/h1.json")}, config=config)
    assert result.output_paths["H1"] == Path("/tmp/h1.json")
    assert result.config is config


def test_base_simulator_is_abstract() -> None:
    """BaseNoiseSimulator cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseNoiseSimulator()


def test_runtime_protocol_accepts_duck_typed_simulator() -> None:
    """Runtime protocol checks accept duck-typed simulators."""
    simulator = DuckNoiseSimulator()

    assert isinstance(simulator, NoiseSimulator)


def test_default_simulator_satisfies_noise_protocol() -> None:
    """DefaultNoiseSimulator satisfies the runtime-checkable protocol."""
    simulator = DefaultNoiseSimulator()

    assert isinstance(simulator, NoiseSimulator)


def test_protocol_is_importable_from_top_level_package() -> None:
    """NoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.NoiseSimulator is NoiseSimulator


def test_colored_simulator_is_importable_from_top_level_package() -> None:
    """ColoredNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.ColoredNoiseSimulator is ColoredNoiseSimulator


def test_correlated_simulator_is_importable_from_top_level_package() -> None:
    """CorrelatedNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.CorrelatedNoiseSimulator is CorrelatedNoiseSimulator
