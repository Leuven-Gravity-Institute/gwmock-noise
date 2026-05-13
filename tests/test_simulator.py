"""Tests for the noise simulator interface."""

from __future__ import annotations

import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gwmock_noise
from gwmock_noise.config import NoiseComponentConfig, NoiseConfig, OutputConfig
from gwmock_noise.parallel import ParallelAdapter
from gwmock_noise.simulators import (
    AddLines,
    ARNoiseSimulator,
    BaseNoiseSimulator,
    BlipGlitch,
    ColoredNoiseSimulator,
    CorrelatedARNoiseSimulator,
    CorrelatedNoiseSimulator,
    DefaultNoiseSimulator,
    InjectGlitches,
    LogNormalAmplitudeDistribution,
    NoiseSimulator,
    ScatteredLightGlitch,
    SchumannNoiseSimulator,
    SchumannParams,
    SimulationResult,
    SpectralLineSimulator,
    take,
)
from gwmock_noise.simulators.registry import available_simulator_names, discover_configurable_simulators


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

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ):
        """Yield minimal per-detector arrays lazily."""
        while True:
            yield self.generate(chunk_duration, sampling_frequency, detectors, seed)
            seed = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Return representative metadata."""
        return {"implementation": "duck"}


def test_default_simulator_run(tmp_path: Path) -> None:
    """DefaultNoiseSimulator creates data artifacts and metadata sidecars."""
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
    assert result.output_paths["H1"] == tmp_path / "test_H1.npy"
    assert result.output_paths["L1"] == tmp_path / "test_L1.npy"
    assert (tmp_path / "test_H1.npy").exists()
    assert (tmp_path / "test_H1.json").exists()
    assert (tmp_path / "test_L1.npy").exists()
    assert (tmp_path / "test_L1.json").exists()

    strain = np.load(result.output_paths["H1"])
    assert strain.shape == (round(config.duration * config.sampling_frequency),)
    assert strain.dtype == float

    metadata = json.loads((tmp_path / "test_H1.json").read_text())
    assert metadata["detector"] == "H1"
    assert metadata["artifact_format"] == "npy"
    assert metadata["artifact_path"] == str(tmp_path / "test_H1.npy")
    assert metadata["implementation"] == "white"


def test_default_simulator_generate_returns_reproducible_white_noise() -> None:
    """Default generate() returns seeded white-noise arrays."""
    simulator = DefaultNoiseSimulator()

    first = simulator.generate(duration=2.0, sampling_frequency=8.0, detectors=["H1", "L1"], seed=7)
    second = simulator.generate(duration=2.0, sampling_frequency=8.0, detectors=["H1", "L1"], seed=7)

    assert first["H1"].shape == (16,)
    np.testing.assert_allclose(first["H1"], second["H1"])
    np.testing.assert_allclose(first["L1"], second["L1"])


def test_default_simulator_run_uses_frame_writer_for_gwf_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run() routes explicit GWF output through FrameWriter."""
    captured: dict[str, object] = {}

    class StubFrameWriter:
        def __init__(
            self,
            base: NoiseSimulator,
            gps_start: float,
            output_dir: Path,
            channel_prefix: str = "MOCK",
            prefix: str = "",
        ) -> None:
            captured["base"] = base
            captured["gps_start"] = gps_start
            captured["output_dir"] = output_dir
            captured["channel_prefix"] = channel_prefix
            captured["prefix"] = prefix
            self.output_dir = output_dir
            self.prefix = prefix

        def write(
            self,
            duration: float,
            sampling_frequency: float,
            detectors: list[str],
            seed: int | None = None,
        ) -> dict[str, Path]:
            captured["duration"] = duration
            captured["sampling_frequency"] = sampling_frequency
            captured["detectors"] = list(detectors)
            captured["seed"] = seed
            output_paths: dict[str, Path] = {}
            for detector in detectors:
                name = f"{self.prefix}_{detector}.gwf" if self.prefix else f"{detector}.gwf"
                output_path = self.output_dir / name
                output_path.write_bytes(b"gwf")
                output_paths[detector] = output_path
            return output_paths

    monkeypatch.setattr("gwmock_noise.simulators.default.FrameWriter", StubFrameWriter)

    config = NoiseConfig(
        detectors=["H1"],
        duration=2.0,
        sampling_frequency=128.0,
        output=OutputConfig(
            directory=tmp_path,
            prefix="frame",
            format="gwf",
            gps_start=100.5,
            channel_prefix="SIM",
        ),
        seed=11,
    )

    result = DefaultNoiseSimulator().run(config)

    assert result.output_paths["H1"] == tmp_path / "frame_H1.gwf"
    assert captured == {
        "base": captured["base"],
        "gps_start": 100.5,
        "output_dir": tmp_path,
        "channel_prefix": "SIM",
        "prefix": "frame",
        "duration": 2.0,
        "sampling_frequency": 128.0,
        "detectors": ["H1"],
        "seed": 11,
    }

    metadata = json.loads((tmp_path / "frame_H1.json").read_text())
    assert metadata["artifact_format"] == "gwf"
    assert metadata["artifact_path"] == str(tmp_path / "frame_H1.gwf")


def test_default_simulator_uses_zero_base_for_glitch_only_configuration(tmp_path: Path) -> None:
    """Glitch-only components still use the internal zero-noise base."""
    config = NoiseConfig(
        detectors=["H1"],
        duration=2.0,
        sampling_frequency=128.0,
        output=OutputConfig(directory=tmp_path, prefix="glitch_only"),
        components=[
            {
                "simulator": "glitches",
                "models": [
                    BlipGlitch(
                        rate=0.2,
                        width=0.01,
                        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
                    )
                ],
            }
        ],
    )
    simulator = DefaultNoiseSimulator()
    runtime = simulator._configure_simulator(config)
    assert isinstance(runtime, InjectGlitches)
    assert runtime.metadata["base_implementation"] == "zero"


def test_default_simulator_rejects_empty_spectral_line_component() -> None:
    """A malformed spectral-line component is rejected during runtime construction."""
    config = NoiseConfig.model_construct(
        detectors=["H1"],
        duration=2.0,
        sampling_frequency=128.0,
        output=OutputConfig(directory=Path("."), prefix="x"),
        seed=None,
        components=[NoiseComponentConfig(simulator="spectral_lines", options={"lines": []})],
    )
    with pytest.raises(ValueError, match="at least one spectral line"):
        DefaultNoiseSimulator()._configure_simulator(config)


def test_registry_discovers_only_configurable_simulators() -> None:
    """Automatic discovery registers only concrete config-driven simulators."""
    simulator_classes = discover_configurable_simulators()
    discovered_names = {simulator_class.simulator_name for simulator_class in simulator_classes}
    discovered_class_names = {simulator_class.__name__ for simulator_class in simulator_classes}

    assert {
        "ar",
        "colored",
        "correlated",
        "correlated_ar",
        "glitches",
        "schumann",
        "spectral_lines",
        "white",
    } <= discovered_names
    assert "white" in available_simulator_names()
    assert "AddLines" not in discovered_class_names
    assert "InjectGlitches" not in discovered_class_names
    assert "DefaultNoiseSimulator" not in discovered_class_names


def test_default_simulator_supports_explicit_ar_selection(tmp_path: Path) -> None:
    """A component list can route DefaultNoiseSimulator to AR noise."""
    psd_path = tmp_path / "flat_psd.txt"
    frequencies = np.linspace(0.0, 64.0, 65)
    np.savetxt(psd_path, np.column_stack((frequencies, np.full_like(frequencies, 2.0e-3))))

    config = NoiseConfig(
        detectors=["H1"],
        duration=2.0,
        sampling_frequency=128.0,
        output=OutputConfig(directory=tmp_path, prefix="ar_explicit"),
        seed=9,
        components=[{"simulator": "ar", "psd_file": psd_path, "order": 8}],
    )

    DefaultNoiseSimulator().run(config)

    metadata = json.loads((tmp_path / "ar_explicit_H1.json").read_text())
    assert metadata["implementation"] == "autoregressive"
    assert metadata["autoregressive_noise"]["order"] == 8


def test_default_simulator_rejects_unknown_explicit_simulator() -> None:
    """Unknown component simulator names fail with the discovered-name list."""
    config = NoiseConfig(components=[{"simulator": "not_a_backend"}])
    with pytest.raises(ValueError, match="Unknown simulator component"):
        DefaultNoiseSimulator()._configure_simulator(config)


def test_default_simulator_explicit_white_base_supports_additive_lines(tmp_path: Path) -> None:
    """White plus line components compose into one additive simulation."""
    config = NoiseConfig(
        detectors=["H1"],
        duration=2.0,
        sampling_frequency=128.0,
        output=OutputConfig(directory=tmp_path, prefix="white_lines"),
        seed=21,
        components=[
            "white",
            {
                "simulator": "spectral_lines",
                "lines": [gwmock_noise.SpectralLine(frequency=16.0, amplitude=1.0e-2, phase=0.0)],
            },
        ],
    )

    DefaultNoiseSimulator().run(config)

    metadata = json.loads((tmp_path / "white_lines_H1.json").read_text())
    assert metadata["implementation"] == "composed"
    assert [component["simulator"] for component in metadata["components"]] == ["white", "spectral_lines"]


def test_simulation_result_attributes() -> None:
    """SimulationResult has expected attributes for upstream consumers."""
    config = NoiseConfig()
    tmp_path = Path(tempfile.gettempdir()) / "h1.json"
    result = SimulationResult(output_paths={"H1": tmp_path}, config=config)
    assert result.output_paths["H1"] == tmp_path
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


def test_ar_simulator_is_importable_from_top_level_package() -> None:
    """ARNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.ARNoiseSimulator is ARNoiseSimulator


def test_correlated_simulator_is_importable_from_top_level_package() -> None:
    """CorrelatedNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.CorrelatedNoiseSimulator is CorrelatedNoiseSimulator


def test_correlated_ar_simulator_is_importable_from_top_level_package() -> None:
    """CorrelatedARNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.CorrelatedARNoiseSimulator is CorrelatedARNoiseSimulator


def test_spectral_line_simulator_is_importable_from_top_level_package() -> None:
    """SpectralLineSimulator is re-exported from the top-level package."""
    assert gwmock_noise.SpectralLineSimulator is SpectralLineSimulator


def test_add_lines_is_importable_from_top_level_package() -> None:
    """AddLines is re-exported from the top-level package."""
    assert gwmock_noise.AddLines is AddLines


def test_inject_glitches_is_importable_from_top_level_package() -> None:
    """InjectGlitches is re-exported from the top-level package."""
    assert gwmock_noise.InjectGlitches is InjectGlitches


def test_take_is_importable_from_simulators_package() -> None:
    """Take is re-exported from the simulators package."""
    assert gwmock_noise.take is take


def test_protocol_simulators_expose_streaming_generator_methods() -> None:
    """The built-in protocol simulators publish generator-based streaming methods."""
    assert inspect.isgeneratorfunction(DefaultNoiseSimulator.generate_stream)
    assert inspect.isgeneratorfunction(SpectralLineSimulator.generate_stream)
    assert inspect.isgeneratorfunction(AddLines.generate_stream)
    assert inspect.isgeneratorfunction(InjectGlitches.generate_stream)


def test_glitch_models_are_importable_from_top_level_package() -> None:
    """Glitch models are re-exported from the top-level package."""
    assert gwmock_noise.BlipGlitch is BlipGlitch
    assert gwmock_noise.ScatteredLightGlitch is ScatteredLightGlitch
    assert gwmock_noise.LogNormalAmplitudeDistribution is LogNormalAmplitudeDistribution


def test_parallel_adapter_is_importable_from_top_level_package() -> None:
    """ParallelAdapter is re-exported from the top-level package."""
    assert gwmock_noise.ParallelAdapter is ParallelAdapter


def test_schumann_simulator_is_importable_from_top_level_package() -> None:
    """SchumannNoiseSimulator is re-exported from the top-level package."""
    assert gwmock_noise.SchumannNoiseSimulator is SchumannNoiseSimulator
    assert gwmock_noise.SchumannParams is SchumannParams
