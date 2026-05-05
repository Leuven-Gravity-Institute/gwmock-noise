"""Tests for bundled PSD preset resolution."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest_mock

from gwmock_noise import DefaultNoiseSimulator
from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.simulators.colored import ColoredNoiseSimulator

PSD_PACKAGE = "gwmock_noise.data.psd"
PRESET_NAME = "ET_10_full_cryo_psd"


def _bundled_psd_path(name: str) -> Path:
    """Return the filesystem path of a bundled PSD preset."""
    return Path(str(resources.files(PSD_PACKAGE).joinpath(f"{name}.txt")))


def test_noise_config_resolves_bundled_psd_preset() -> None:
    """Bare preset names resolve to packaged PSD files."""
    config = NoiseConfig(psd_file=PRESET_NAME)
    assert config.psd_file == _bundled_psd_path(PRESET_NAME)


def test_noise_config_normalizes_absolute_psd_path_string(tmp_path: Path) -> None:
    """Explicit path strings continue to resolve as filesystem paths."""
    psd_path = tmp_path / "noise_psd.txt"
    np.savetxt(psd_path, np.column_stack((np.array([0.0, 128.0]), np.array([1.0, 1.0]))))

    config = NoiseConfig(psd_file=str(psd_path))
    assert config.psd_file == psd_path


def test_noise_config_preserves_http_psd_url() -> None:
    """HTTP(S) PSD references remain URL strings."""
    psd_url = "https://example.com/noise_psd.txt"

    config = NoiseConfig(psd_file=psd_url)
    assert config.psd_file == psd_url


def test_colored_noise_simulator_preserves_http_psd_url(mocker: pytest_mock.MockerFixture) -> None:
    """ColoredNoiseSimulator keeps URL PSDs as remote references."""
    psd_url = "https://example.com/noise_psd.txt"
    mocker.patch(
        "gwmock_noise.simulators.colored.load_spectral_series",
        return_value=(np.array([0.0, 128.0]), np.array([1.0, 1.0])),
    )

    simulator = ColoredNoiseSimulator(psd_file=psd_url, detectors=["H1"], sampling_frequency=256.0)
    assert simulator.psd_file == psd_url


def test_preset_noise_matches_direct_bundled_file(tmp_path: Path) -> None:
    """Preset resolution produces byte-identical output to the bundled file path."""
    preset_output = tmp_path / "preset"
    direct_output = tmp_path / "direct"
    direct_psd_path = _bundled_psd_path(PRESET_NAME)

    simulator = DefaultNoiseSimulator()
    simulator.run(
        NoiseConfig(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=256.0,
            output=OutputConfig(directory=preset_output, prefix="noise"),
            seed=1234,
            psd_file=PRESET_NAME,
        )
    )
    simulator.run(
        NoiseConfig(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=256.0,
            output=OutputConfig(directory=direct_output, prefix="noise"),
            seed=1234,
            psd_file=direct_psd_path,
        )
    )

    assert (preset_output / "noise_H1.npy").read_bytes() == (direct_output / "noise_H1.npy").read_bytes()
