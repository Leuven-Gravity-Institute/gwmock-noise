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


def test_draw_reports_the_population_snr_and_the_realized_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A gengli draw records the SNR it asked gengli for and the one it ended up with."""
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([7.0, 11.0]))
    _write_flat_psd(psd_file)

    requested: dict[str, float] = {}

    class StubGenerator:
        def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
            _ = (seed, srate, glitch_type)
            requested["snr"] = snr
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

    draw = model._draw(256.0, rng=np.random.default_rng(3))

    assert draw.glitch_class == "Blip"
    assert draw.target_snr == requested["snr"]
    assert draw.amplitude == 2.0
    # gengli imposes the target on the *whitened* waveform, so the realized SNR against
    # the coloring PSD is a different number rather than the target restated.
    colored = coloring_module.color_whitened_waveform(
        np.hanning(64),
        sampling_frequency=256.0,
        psd_frequencies=model._psd_frequencies,
        psd_values=model._psd_values,
        low_frequency_cutoff=model.low_frequency_cutoff,
        high_frequency_cutoff=model.high_frequency_cutoff,
    )
    expected = 2.0 * coloring_module.optimal_snr(colored, sampling_frequency=256.0)
    assert draw.realized_snr == pytest.approx(expected, rel=1e-12)


def test_draw_preserves_the_historical_rng_draw_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recording the target SNR must not permute the gengli seed and the SNR draws.

    The two are consecutive draws from one generator, so swapping them would change
    every waveform this model has ever produced while every test about SNR still passed.
    """
    population_file = tmp_path / "population.h5"
    psd_file = tmp_path / "psd.txt"
    write_blip_population_file(population_file, snr_samples=np.array([7.0, 11.0]))
    _write_flat_psd(psd_file)

    class StubGenerator:
        def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
            _ = (snr, srate, glitch_type)
            return np.full(8, float(seed), dtype=float)

    monkeypatch.setattr(
        gengli_module,
        "_load_gengli",
        lambda: SimpleNamespace(glitch_generator=lambda detector: StubGenerator()),
    )
    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.5,
        psd_file=psd_file,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
    )

    draw = model._draw(256.0, rng=np.random.default_rng(3))

    # The seed gengli was handed is the first draw from a fresh generator, ahead of the
    # population-SNR draw and the amplitude draw.
    expected_seed = int(np.random.default_rng(3).integers(0, gengli_module.UINT32_EXCLUSIVE_MAX))
    assert draw.waveform.size == 8
    reference = np.full(8, float(expected_seed), dtype=float)
    colored = coloring_module.color_whitened_waveform(
        reference,
        sampling_frequency=256.0,
        psd_frequencies=model._psd_frequencies,
        psd_values=model._psd_values,
        low_frequency_cutoff=model.low_frequency_cutoff,
        high_frequency_cutoff=model.high_frequency_cutoff,
    )
    np.testing.assert_allclose(draw.waveform, colored.time_series, rtol=0.0, atol=0.0)


# The sampled-target path. The reference these tests hold the model to is the strain
# itself: every tail check recomputes the SNR from the generated waveform with its own
# inner product rather than reading back what the model reports.
SAMPLED_PSD_UPPER_HZ = 4096.0
SAMPLED_RATE_HZ = 4096.0


class _ScaledTemplateGenerator:
    """A gengli stand-in returning one fixed whitened shape scaled by the requested SNR.

    A fixed shape makes the colored strain's SNR exactly proportional to the target, so
    the tail of the strain's SNRs is the tail of the targets with the threshold moved by
    one constant factor -- which is what lets a tail index be read back out of strain.
    """

    def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
        _ = (seed, srate, glitch_type)
        return snr * np.hanning(256)


class _SeededGenerator:
    """A gengli stand-in whose waveform depends on both the seed and the target SNR."""

    def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
        _ = (srate, glitch_type)
        return snr * np.hanning(64) * np.random.default_rng(seed).normal(size=64)


def _install_generator(monkeypatch: pytest.MonkeyPatch, stub: object) -> None:
    monkeypatch.setattr(
        gengli_module,
        "_load_gengli",
        lambda: SimpleNamespace(glitch_generator=lambda detector: stub),
    )


def _write_positive_psd(path: Path) -> None:
    """Write a strictly positive flat PSD covering the full test band."""
    frequencies = np.linspace(0.0, SAMPLED_PSD_UPPER_HZ, 129)
    np.savetxt(path, np.column_stack((frequencies, np.ones_like(frequencies))))


def _strain_snr(waveform: np.ndarray, psd_file: Path, sampling_frequency: float) -> float:
    """Recompute the optimal SNR of a strain against the raw PSD table."""
    table = np.loadtxt(psd_file)
    frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / sampling_frequency)
    psd = np.interp(frequencies, table[:, 0], table[:, 1], left=0.0, right=0.0)
    valid = (frequencies >= 2.0) & (frequencies < sampling_frequency / 2.0) & (psd > 0.0)
    waveform_fd = np.fft.rfft(waveform) / sampling_frequency
    delta_frequency = sampling_frequency / waveform.size
    return float(np.sqrt(4.0 * delta_frequency * np.sum(np.abs(waveform_fd[valid]) ** 2 / psd[valid])))


def _tail_index(snrs: np.ndarray, threshold: float) -> float:
    """Bias-corrected Hill estimator of the survival exponent above ``threshold``."""
    exceedances = np.asarray(snrs, dtype=float)
    exceedances = exceedances[exceedances >= threshold]
    return float((exceedances.size - 1) / np.sum(np.log(exceedances / threshold)))


def _sampled_model(psd_file: Path, snr: object, *, std: float = 0.0) -> GengliBlipGlitch:
    return GengliBlipGlitch(
        rate=0.5,
        psd_file=psd_file,
        snr=snr,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=std),
    )


def test_tail_estimator_recovers_a_known_index() -> None:
    """The estimator is anchored on numpy's own Pareto sampler, not on the code under test."""
    reference = 10.0 * (1.0 + np.random.default_rng(3).pareto(1.34, size=20000))

    assert _tail_index(reference, 10.0) == pytest.approx(1.34, rel=0.03)


@pytest.mark.parametrize("alpha", [0.40, 3.11])
@pytest.mark.parametrize("per_class", [False, True])
def test_power_law_snr_tail_index_is_recovered_from_the_strain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    alpha: float,
    per_class: bool,
) -> None:
    """The generated strain carries the configured tail index, alone or as a per-class entry.

    The coloring factor is measured from the strain of a unit-target draw, so the
    threshold the index is estimated above is located without reading any target back
    from the model.
    """
    psd_file = tmp_path / "psd.txt"
    _write_positive_psd(psd_file)
    _install_generator(monkeypatch, _ScaledTemplateGenerator())
    minimum = 10.0
    distribution = {"distribution": "power_law", "minimum": minimum, "alpha": alpha}

    unit = _sampled_model(psd_file, 1.0)._draw(SAMPLED_RATE_HZ, rng=np.random.default_rng(0))
    coloring_factor = _strain_snr(unit.waveform, psd_file, SAMPLED_RATE_HZ)
    assert coloring_factor > 0.0

    model = _sampled_model(psd_file, {"Blip": distribution} if per_class else distribution)
    rng = np.random.default_rng(20261006)
    n_draws = 4000
    strain_snrs = np.array(
        [_strain_snr(model._draw(SAMPLED_RATE_HZ, rng=rng).waveform, psd_file, SAMPLED_RATE_HZ) for _ in range(n_draws)]
    )

    assert np.min(strain_snrs) >= minimum * coloring_factor * (1.0 - 1e-9)
    # The Hill estimator's relative standard error is 1/sqrt(n), about 1.6% here.
    assert _tail_index(strain_snrs, minimum * coloring_factor * (1.0 - 1e-9)) == pytest.approx(alpha, rel=0.08)


def test_sampled_draw_reports_the_target_it_handed_gengli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recorded target is the one gengli received, and the strain matches the realized SNR."""
    psd_file = tmp_path / "psd.txt"
    _write_positive_psd(psd_file)
    requested: list[float] = []

    class RecordingGenerator(_ScaledTemplateGenerator):
        def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
            requested.append(snr)
            return _ScaledTemplateGenerator.get_glitch(self, seed=seed, snr=snr, srate=srate, glitch_type=glitch_type)

    _install_generator(monkeypatch, RecordingGenerator())
    model = _sampled_model(psd_file, {"distribution": "power_law", "minimum": 10.0, "alpha": 1.34, "maximum": 600.0})
    rng = np.random.default_rng(5)
    draws = [model._draw(SAMPLED_RATE_HZ, rng=rng) for _ in range(20)]

    assert [draw.target_snr for draw in draws] == requested
    assert all(10.0 <= target <= 600.0 for target in requested)
    assert len(set(requested)) == len(requested)
    for draw in draws:
        assert draw.glitch_class == "Blip"
        assert draw.realized_snr == pytest.approx(_strain_snr(draw.waveform, psd_file, SAMPLED_RATE_HZ), rel=1e-9)


def test_empirical_snr_draws_only_from_the_supplied_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empirical specification hands gengli only values from its table."""
    psd_file = tmp_path / "psd.txt"
    _write_positive_psd(psd_file)
    _install_generator(monkeypatch, _ScaledTemplateGenerator())
    table = [8.0, 12.5, 40.0]
    model = _sampled_model(psd_file, {"distribution": "empirical", "samples": table})
    rng = np.random.default_rng(9)

    targets = {model._draw(SAMPLED_RATE_HZ, rng=rng).target_snr for _ in range(200)}

    assert targets == set(table)


def test_scalar_snr_consumes_nothing_from_the_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fixed target leaves only the gengli seed and the amplitude on the stream."""
    psd_file = tmp_path / "psd.txt"
    _write_positive_psd(psd_file)
    _install_generator(monkeypatch, _SeededGenerator())
    model = _sampled_model(psd_file, 9.0, std=0.25)

    rng = np.random.default_rng(11)
    draws = [model._draw(SAMPLED_RATE_HZ, rng=rng) for _ in range(3)]

    reference = np.random.default_rng(11)
    for draw in draws:
        _ = reference.integers(0, gengli_module.UINT32_EXCLUSIVE_MAX)
        assert draw.amplitude == model.amplitude_distribution.sample(reference)
        assert draw.target_snr == 9.0
    assert rng.random() == reference.random()


def test_population_file_realizations_are_bit_for_bit_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The population-file path draws what it always drew, stream position included.

    The pinned numbers were produced before ``snr`` existed. Routing the population
    draw through the new specification -- or drawing a target unconditionally -- would
    move the waveforms, the targets or the state the stream is left in.
    """
    psd_file = tmp_path / "psd.txt"
    population_file = tmp_path / "population.h5"
    _write_positive_psd(psd_file)
    write_blip_population_file(population_file, snr_samples=np.array([7.0, 11.0, 23.0]))
    _install_generator(monkeypatch, _SeededGenerator())
    model = GengliBlipGlitch.from_population_file(
        population_file,
        rate=0.5,
        psd_file=psd_file,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.25),
    )

    rng = np.random.default_rng(20261006)
    draws = [model._draw(SAMPLED_RATE_HZ, rng=rng) for _ in range(3)]

    assert [draw.waveform[0] for draw in draws] == pytest.approx(
        [-1.9168500305791134, -2.8265718627156424, 0.23283331461251955], rel=1e-12
    )
    assert [float(np.sum(draw.waveform**2)) for draw in draws] == pytest.approx(
        [8007.126678364175, 13120.974792656727, 2818.1626830569226], rel=1e-12
    )
    assert [draw.target_snr for draw in draws] == [23.0, 23.0, 11.0]
    assert [draw.realized_snr for draw in draws] == pytest.approx(
        [1.9773036750635968, 2.5311511161874387, 1.1730541324194668], rel=1e-12
    )
    assert [draw.glitch_class for draw in draws] == ["Blip", "Blip", "Blip"]
    assert rng.random() == pytest.approx(0.5899998082309823, rel=1e-12)


def test_snr_configuration_errors(tmp_path: Path) -> None:
    """Exactly one target source, a PSD, and only the class gengli generates."""
    psd_file = tmp_path / "psd.txt"
    population_file = tmp_path / "population.h5"
    _write_positive_psd(psd_file)
    write_blip_population_file(population_file, snr_samples=np.array([7.0]))
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.0)

    with pytest.raises(ValueError, match="exactly one of 'population_file' or 'snr'"):
        GengliBlipGlitch(rate=0.5, psd_file=psd_file, amplitude_distribution=amplitude)
    with pytest.raises(ValueError, match="exactly one of 'population_file' or 'snr'"):
        GengliBlipGlitch(
            rate=0.5, population_file=population_file, psd_file=psd_file, snr=8.0, amplitude_distribution=amplitude
        )
    with pytest.raises(TypeError, match="requires a psd_file"):
        GengliBlipGlitch(rate=0.5, snr=8.0, amplitude_distribution=amplitude)
    with pytest.raises(ValueError, match="exactly the 'Blip' class"):
        GengliBlipGlitch(rate=0.5, psd_file=psd_file, snr={"Koi_Fish": 8.0}, amplitude_distribution=amplitude)
    with pytest.raises(ValueError, match="exactly the 'Blip' class"):
        GengliBlipGlitch(rate=0.5, psd_file=psd_file, snr={"Blip": 8.0, "Tomte": 9.0}, amplitude_distribution=amplitude)
    with pytest.raises(ValueError, match="finite and greater than zero"):
        GengliBlipGlitch(rate=0.5, psd_file=psd_file, snr=-1.0, amplitude_distribution=amplitude)
    with pytest.raises(ValueError, match="distribution must be one of"):
        GengliBlipGlitch(
            rate=0.5, psd_file=psd_file, snr={"distribution": "lognormal"}, amplitude_distribution=amplitude
        )


def test_sampled_snr_serializes_and_replays_through_a_configuration(tmp_path: Path) -> None:
    """A configured mapping builds the model, and its metadata rebuilds the same one."""
    from gwmock_noise.glitches.models import normalize_glitch_models

    psd_file = tmp_path / "psd.txt"
    _write_positive_psd(psd_file)
    entry = {
        "kind": "gengli_blip",
        "rate": 0.25,
        "psd_file": str(psd_file),
        "amplitude_distribution": {"distribution": "lognormal", "mean": 1.0, "std": 0.0},
        "snr": {"Blip": {"distribution": "power_law", "minimum": 10.0, "alpha": 1.34, "maximum": 600.0}},
    }

    (model,) = normalize_glitch_models([entry])
    metadata = model.serialize()

    assert metadata["population_file"] is None
    assert "population_size" not in metadata
    assert metadata["snr"] == {"Blip": {"distribution": "power_law", "minimum": 10.0, "alpha": 1.34, "maximum": 600.0}}
    (replayed,) = normalize_glitch_models([metadata])
    assert replayed.serialize() == metadata
