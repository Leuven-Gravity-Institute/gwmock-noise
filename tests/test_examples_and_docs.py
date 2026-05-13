"""Slow smoke tests for benchmark examples and offline documentation snippets."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _runtime_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment with ``src`` on ``PYTHONPATH`` (matches uninstalled dev runs)."""
    runtime_env = os.environ.copy()
    pythonpath = str(SRC_DIR)
    if runtime_env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{runtime_env['PYTHONPATH']}"
    runtime_env["PYTHONPATH"] = pythonpath
    if extra is not None:
        runtime_env.update(extra)
    return runtime_env


def _run_python(*args: str, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run Python with the project source tree on ``PYTHONPATH``."""
    runtime_env = _runtime_env(env)

    return subprocess.run(  # noqa: S603
        [sys.executable, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=runtime_env,
    )


@pytest.mark.slow
def test_benchmark_examples_run_with_small_inputs(tmp_path: Path) -> None:
    """Benchmark example scripts run successfully with reduced parameters."""
    commands = [
        [
            "examples/benchmark_ar_vs_fft.py",
            "--duration",
            "1",
            "--sampling-frequency",
            "128",
            "--order",
            "8",
        ],
        [
            "examples/benchmark_correlated_ar_vs_fft.py",
            "--duration",
            "1",
            "--sampling-frequency",
            "128",
            "--order",
            "8",
        ],
    ]

    for command in commands:
        result = _run_python(*command, cwd=REPO_ROOT, env={"MPLBACKEND": "Agg", "TMPDIR": str(tmp_path)})
        assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_example_configs_run(tmp_path: Path) -> None:
    """Shipped example configs remain runnable from the CLI."""
    for config_name in [
        "examples/noise_config_example.toml",
        "examples/noise_config_example.yaml",
        "examples/noise_config_multiple_components.toml",
    ]:
        source = REPO_ROOT / config_name
        workdir = tmp_path / Path(config_name).name.replace(".", "_")
        workdir.mkdir()
        local_config = workdir / source.name
        local_config.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        result = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "gwmock_noise.cli.main", "simulate", local_config.name],
            cwd=workdir,
            capture_output=True,
            text=True,
            check=False,
            env=_runtime_env({"TMPDIR": str(tmp_path)}),
        )
        assert result.returncode == 0, result.stderr
        assert list(workdir.rglob("*.npy"))


@pytest.mark.slow
def test_offline_documentation_snippets_run(tmp_path: Path) -> None:
    """Representative offline snippets from the user guide remain executable."""
    code = """
from pathlib import Path
from collections.abc import Iterator
from typing import Any

import numpy as np

from gwmock_noise import (
    ARNoiseSimulator,
    BlipGlitch,
    ColoredNoiseSimulator,
    DefaultNoiseSimulator,
    LogNormalAmplitudeDistribution,
    NoiseConfig,
    NoiseSimulator,
    OutputConfig,
    SpectralLine,
    open_stream,
)

workdir = Path("doc-smoke")
workdir.mkdir(exist_ok=True)

# quick_start / minimal_usage basic simulation
result = DefaultNoiseSimulator().run(
    NoiseConfig(
        detectors=["H1", "L1"],
        duration=1.0,
        sampling_frequency=128.0,
        output=OutputConfig(directory=workdir / "output", prefix="noise"),
        seed=42,
    )
)
assert sorted(result.output_paths) == ["H1", "L1"]

# minimal_usage custom PSD / preset / correlated config / lines / glitches
psd_path = workdir / "psd.txt"
np.savetxt(psd_path, np.column_stack((np.array([0.0, 64.0]), np.array([1.0e-3, 1.0e-3]))))
config = NoiseConfig(
    detectors=["H1"],
    duration=2.0,
    components=[{"simulator": "colored", "psd_file": psd_path}],
)
assert config.components[0].simulator == "colored"
assert config.components[0].options["psd_file"] == psd_path
assert NoiseConfig(components=[{"simulator": "colored", "psd_file": "ET_D_psd"}]).components[0].simulator == "colored"
network_config = NoiseConfig(
    detectors=["H1", "L1"],
    components=[
        {"simulator": "correlated", "psd_files": {"H1": psd_path, "L1": psd_path}, "csd_files": {"H1-L1": psd_path}},
        {"simulator": "spectral_lines", "lines": [SpectralLine(frequency=32.0, amplitude=1.0e-3)]},
        {
            "simulator": "glitches",
            "models": [
                BlipGlitch(
                    rate=0.5,
                    width=0.01,
                    amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
                )
            ],
        },
    ],
)
assert [component.simulator for component in network_config.components] == ["correlated", "spectral_lines", "glitches"]

# minimal_usage AR / streaming
ar_sim = ARNoiseSimulator(order=8, detectors=["H1"], duration=1.0, sampling_frequency=128.0, psd_file=psd_path)
assert ar_sim.generate(1.0, 128.0, ["H1"], seed=5)["H1"].shape == (128,)
stream = open_stream(
    ColoredNoiseSimulator(psd_file=psd_path, detectors=["H1"], sampling_frequency=128.0),
    chunk_duration=0.5,
    sampling_frequency=128.0,
    detectors=["H1"],
    seed=7,
)
assert next(stream)["H1"].shape == (64,)

# custom_simulators
class RampNoiseSimulator:
    def __init__(self) -> None:
        self.duration = 1.0
        self.sampling_frequency = 8.0
        self.detectors = ["H1"]
        self.seed = None
        self._offset = 0

    def generate(
        self,
        duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> dict[str, np.ndarray]:
        n_samples = round(duration * sampling_frequency)
        self.duration = duration
        self.sampling_frequency = sampling_frequency
        self.detectors = list(detectors)
        self.seed = seed
        return {detector: np.arange(n_samples, dtype=float) for detector in detectors}

    def generate_stream(
        self,
        chunk_duration: float,
        sampling_frequency: float,
        detectors: list[str],
        seed: int | None = None,
    ) -> Iterator[dict[str, np.ndarray]]:
        n_samples = round(chunk_duration * sampling_frequency)
        while True:
            start = self._offset
            stop = start + n_samples
            self._offset = stop
            yield {detector: np.arange(start, stop, dtype=float) for detector in detectors}

    @property
    def metadata(self) -> dict[str, Any]:
        return {"implementation": "ramp"}


simulator: NoiseSimulator = RampNoiseSimulator()
doc_stream = open_stream(
    simulator,
    chunk_duration=0.5,
    sampling_frequency=8.0,
    detectors=["H1", "L1"],
    seed=7,
)
first_chunk = next(doc_stream)
assert sorted(first_chunk) == ["H1", "L1"]
assert np.array_equal(first_chunk["H1"], np.arange(4, dtype=float))
"""
    script_path = tmp_path / "docs_smoke.py"
    script_path.write_text(code, encoding="utf-8")

    result = _run_python(str(script_path), cwd=tmp_path)
    assert result.returncode == 0, result.stderr
