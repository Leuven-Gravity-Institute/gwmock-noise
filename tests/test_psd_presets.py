"""Tests for bundled PSD preset resolution."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pytest
import pytest_mock

from gwmock_noise import DefaultNoiseSimulator
from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.glitches import BlipGlitch, LogNormalAmplitudeDistribution
from gwmock_noise.simulators._spectral import load_spectral_series
from gwmock_noise.simulators.colored import ColoredNoiseSimulator

PSD_PACKAGE = "gwmock_noise.data.psd"
PRESET_NAME = "ET_10_full_cryo_psd"


def _bundled_psd_path(name: str) -> Path:
    """Return the filesystem path of a bundled PSD preset."""
    return Path(str(resources.files(PSD_PACKAGE).joinpath(f"{name}.txt")))


def test_colored_component_preserves_bundled_psd_reference() -> None:
    """Bare preset names can be carried through component construction."""
    simulator = DefaultNoiseSimulator()._configure_simulator(
        NoiseConfig(components=[{"simulator": "colored", "psd_file": PRESET_NAME}])
    )
    assert isinstance(simulator, ColoredNoiseSimulator)
    assert simulator.psd_file == Path(PRESET_NAME)


def test_colored_component_normalizes_absolute_psd_path_string(tmp_path: Path) -> None:
    """Explicit path strings continue to resolve as filesystem paths."""
    psd_path = tmp_path / "noise_psd.txt"
    np.savetxt(psd_path, np.column_stack((np.array([0.0, 128.0]), np.array([1.0, 1.0]))))

    simulator = DefaultNoiseSimulator()._configure_simulator(
        NoiseConfig(components=[{"simulator": "colored", "psd_file": str(psd_path)}])
    )
    assert isinstance(simulator, ColoredNoiseSimulator)
    assert simulator.psd_file == psd_path


def test_colored_component_preserves_http_psd_url(mocker: pytest_mock.MockerFixture) -> None:
    """HTTP(S) PSD references remain URL strings."""
    psd_url = "https://example.com/noise_psd.txt"
    mocker.patch(
        "gwmock_noise.simulators.colored.load_spectral_series",
        return_value=(np.array([0.0, 128.0]), np.array([1.0, 1.0])),
    )

    simulator = DefaultNoiseSimulator()._configure_simulator(
        NoiseConfig(components=[{"simulator": "colored", "psd_file": psd_url}])
    )
    assert isinstance(simulator, ColoredNoiseSimulator)
    assert simulator.psd_file == psd_url


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
            components=[{"simulator": "colored", "psd_file": PRESET_NAME}],
        )
    )
    simulator.run(
        NoiseConfig(
            detectors=["H1"],
            duration=4.0,
            sampling_frequency=256.0,
            output=OutputConfig(directory=direct_output, prefix="noise"),
            seed=1234,
            components=[{"simulator": "colored", "psd_file": direct_psd_path}],
        )
    )

    assert (preset_output / "noise_H1.npy").read_bytes() == (direct_output / "noise_H1.npy").read_bytes()


ALIGO_PRESETS = (
    "aLIGO_O3_actual_H1_psd",
    "aLIGO_O3_actual_L1_psd",
    "aLIGO_O4_high_projected_psd",
    "aLIGO_O4_low_projected_psd",
)

# Sky- and orientation-averaged 1.4+1.4 solar-mass BNS inspiral ranges quoted for the
# corresponding runs in Abbott et al., Living Reviews in Relativity 23, 3 (2020). They anchor
# bundled column to an external reference: a file that tabulated amplitude rather than
# power spectral density would miss these by orders of magnitude.
PUBLISHED_BNS_RANGE_MPC = {
    "aLIGO_O3_actual_H1_psd": 110.0,
    "aLIGO_O3_actual_L1_psd": 135.0,
    "aLIGO_O4_high_projected_psd": 190.0,
    "aLIGO_O4_low_projected_psd": 175.0,
}


def _bns_inspiral_range_mpc(frequencies: np.ndarray, psd: np.ndarray) -> float:
    """Return the averaged 1.4+1.4 solar-mass BNS inspiral range for a one-sided PSD."""
    gravitational_constant = 6.67430e-11
    speed_of_light = 299792458.0
    solar_mass = 1.98892e30
    megaparsec = 3.0856775814913673e22

    chirp_mass = (1.4 * 1.4) ** 0.6 / (2.8) ** 0.2 * solar_mass
    total_mass = 2.8 * solar_mass
    f_isco = speed_of_light**3 / (6**1.5 * np.pi * gravitational_constant * total_mass)
    amplitude = (
        np.sqrt(5.0 / 24.0)
        * np.pi ** (-2.0 / 3.0)
        * speed_of_light
        * (gravitational_constant * chirp_mass / speed_of_light**3) ** (5.0 / 6.0)
    )

    band = (frequencies >= 10.0) & (frequencies <= f_isco)
    integrand = (amplitude * frequencies[band] ** (-7.0 / 6.0)) ** 2 / psd[band]
    horizon_metres = np.sqrt(4.0 * np.trapezoid(integrand, frequencies[band])) / 8.0
    return horizon_metres / megaparsec / 2.2648


@pytest.mark.parametrize("preset", ALIGO_PRESETS)
def test_aligo_preset_resolves_by_bare_name(preset: str) -> None:
    """Each aLIGO preset loads from a bare name identically to its bundled path."""
    by_name = load_spectral_series(preset, kind="PSD")
    by_path = load_spectral_series(_bundled_psd_path(preset), kind="PSD")

    np.testing.assert_array_equal(by_name[0], by_path[0])
    np.testing.assert_array_equal(by_name[1], by_path[1])


@pytest.mark.parametrize("preset", ALIGO_PRESETS)
def test_aligo_preset_is_a_usable_psd_table(preset: str) -> None:
    """Each aLIGO preset is strictly ordered in frequency and strictly positive."""
    frequencies, values = load_spectral_series(preset, kind="PSD")

    assert frequencies.size > 0
    assert np.all(np.diff(frequencies) > 0.0)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0.0)


@pytest.mark.parametrize("preset", ALIGO_PRESETS)
def test_aligo_preset_reproduces_published_bns_range(preset: str) -> None:
    """The bundled column is a PSD, cross-checked against the published BNS range."""
    frequencies, values = load_spectral_series(preset, kind="PSD")

    computed = _bns_inspiral_range_mpc(frequencies, values)
    published = PUBLISHED_BNS_RANGE_MPC[preset]
    assert abs(computed - published) / published < 0.10


@pytest.mark.parametrize("preset", ALIGO_PRESETS)
def test_aligo_preset_resolves_through_glitch_psd_file(preset: str) -> None:
    """Glitch models accept the bare preset name wherever they accept a PSD path."""
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.0)
    glitch = BlipGlitch(rate=0.1, amplitude_distribution=amplitude, width=0.01, psd_file=preset, snr=12.0)
    direct = BlipGlitch(
        rate=0.1, amplitude_distribution=amplitude, width=0.01, psd_file=_bundled_psd_path(preset), snr=12.0
    )

    np.testing.assert_array_equal(glitch._psd_frequencies, direct._psd_frequencies)
    np.testing.assert_array_equal(glitch._psd_values, direct._psd_values)


def test_bundled_psd_provenance_is_recorded() -> None:
    """Every bundled preset is named in the provenance record shipped beside it."""
    provenance = resources.files(PSD_PACKAGE).joinpath("PROVENANCE.md").read_text(encoding="utf-8")

    bundled = sorted(
        entry.name.removesuffix(".txt")
        for entry in resources.files(PSD_PACKAGE).iterdir()
        if entry.name.endswith(".txt")
    )
    assert bundled
    missing = [name for name in bundled if name not in provenance]
    assert not missing, f"presets absent from PROVENANCE.md: {missing}"
