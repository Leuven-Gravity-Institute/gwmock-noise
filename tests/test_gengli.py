"""Tests for the gengli-backed glitch model."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from gwmock_noise import GengliBlipGlitch, LogNormalAmplitudeDistribution
from gwmock_noise.glitches import _coloring as coloring_module
from gwmock_noise.glitches import gengli as gengli_module
from gwmock_noise.glitches.gengli import read_blip_population_file, write_blip_population_file
from gwmock_noise.simulators.colored import PSD_WINDOW_WIDTH_HZ


def _write_flat_psd(path: Path) -> None:
    """Write a small non-negative PSD table."""
    frequencies = np.linspace(0.0, 256.0, 65)
    psd = np.linspace(0.0, 2.0, 65)
    np.savetxt(path, np.column_stack((frequencies, psd)))


def test_from_population_file_round_trips_schema(tmp_path: Path) -> None:
    """Population helper writes the schema consumed by the model loader."""
    population_file = tmp_path / "population.h5"
    samples = np.array([6.0, 8.0, 10.0], dtype=float)

    write_blip_population_file(population_file, snr_samples=samples, metadata={"source": "test"})

    loaded = read_blip_population_file(population_file)
    np.testing.assert_allclose(loaded, samples)
    with h5py.File(population_file, "r") as handle:
        assert handle.attrs["source"] == "test"


def test_generate_waveform_uses_population_and_colors_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Waveform generation routes through gengli and PSD coloring."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([7.0, 11.0]))
    _write_flat_psd(psd_file)

    captured: dict[str, float | int | str] = {}

    class StubGenerator:
        def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
            captured["seed"] = seed
            captured["snr"] = snr
            captured["srate"] = srate
            captured["glitch_type"] = glitch_type
            return np.hanning(64)

    monkeypatch.setattr(
        gengli_module,
        "_load_gengli",
        lambda: SimpleNamespace(glitch_generator=lambda detector: StubGenerator()),
    )

    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.5,
        psd_file=psd_file,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=2.0, std=0.0),
    )

    waveform = model.generate_waveform(256.0, rng=np.random.default_rng(3))

    assert waveform.shape == (64,)
    assert np.max(np.abs(waveform)) > 0.0
    assert captured["snr"] in {7.0, 11.0}
    assert captured["srate"] == 256.0
    assert captured["glitch_type"] == "Blip"


def test_generate_waveform_requires_optional_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The model raises a focused error when gengli is unavailable."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([8.0]))
    _write_flat_psd(psd_file)

    monkeypatch.setattr(
        gengli_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )

    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.5,
        psd_file=psd_file,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )

    with pytest.raises(ImportError, match="requires the optional dependency 'gengli'"):
        model.generate_waveform(256.0, rng=np.random.default_rng(2))


def test_top_level_export_exposes_gengli_blip_glitch() -> None:
    """The package re-exports the gengli model from the top level."""
    assert GengliBlipGlitch.__name__ == "GengliBlipGlitch"


@pytest.mark.integration
def test_serialize_reports_population_and_psd_paths(tmp_path: Path) -> None:
    """Serialize includes file-backed gengli configuration."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([9.0, 12.0]))
    _write_flat_psd(psd_file)

    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.25,
        psd_file=psd_file,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )

    assert model.serialize() == {
        "kind": "gengli_blip",
        "rate": 0.25,
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "population_file": str(population_file),
        "psd_file": str(psd_file),
        "gengli_detector": "L1",
        "low_frequency_cutoff": 2.0,
        "high_frequency_cutoff": None,
        "population_size": 2,
    }


def test_color_glitch_uses_absolute_width_taper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_color_glitch's PSD taper alpha scales with bandwidth to keep a fixed Hz-wide edge."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([7.0]))
    _write_flat_psd(psd_file)

    captured_alphas: list[float] = []
    original_tukey_window = coloring_module._tukey_window

    def _spy_tukey_window(length: int, alpha: float) -> np.ndarray:
        captured_alphas.append(alpha)
        return original_tukey_window(length, alpha=alpha)

    monkeypatch.setattr(coloring_module, "_tukey_window", _spy_tukey_window)

    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.5,
        psd_file=psd_file,
        low_frequency_cutoff=8.0,
        high_frequency_cutoff=96.0,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=2.0, std=0.0),
    )
    model._color_glitch(np.hanning(64), sampling_frequency=256.0)

    assert captured_alphas == pytest.approx([2.0 * PSD_WINDOW_WIDTH_HZ / (96.0 - 8.0)])
