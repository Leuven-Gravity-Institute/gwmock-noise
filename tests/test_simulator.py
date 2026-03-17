"""Tests for the noise simulator interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from gwsim_noise.config import NoiseConfig, OutputConfig
from gwsim_noise.simulators import BaseNoiseSimulator, DefaultNoiseSimulator, SimulationResult


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
