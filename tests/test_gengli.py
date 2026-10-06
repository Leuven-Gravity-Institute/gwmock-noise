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
        "detectors": None,
        "network": None,
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


# Every sample of the three population-path draws pinned below, generated by running
# the population configuration of that test against the implementation as it stood
# before ``snr`` was added. Each value is the ``repr`` of a float64, which round-trips
# exactly.
POPULATION_PIN_WAVEFORMS = (
    np.array(
        [
            -1.9168500305791134,
            0.7755948230716853,
            -1.7927925266971865,
            0.8198465231450471,
            -2.310060855112982,
            0.8344691964979821,
            -1.884945693890822,
            3.1907709030213898,
            0.6162186344118967,
            -1.465784230010286,
            1.72705513218173,
            0.9766872969461357,
            2.3650065068705475,
            0.2025431590039266,
            -3.326815342747872,
            0.984333260348566,
            -16.027579849696775,
            -6.774133929868236,
            0.11228268272386464,
            -10.2547135604418,
            29.469428433302674,
            10.250395010878872,
            -7.207595392196627,
            -33.34981811615054,
            3.1358788979988916,
            0.9023420104387423,
            18.505461340943828,
            0.07948022941603707,
            -5.421026733857175,
            13.91299331174938,
            -15.083455634174419,
            23.14596039357541,
            2.616243164584991,
            3.1901484707998424,
            38.10543780386382,
            10.124784218804288,
            12.418865958387151,
            9.032448516398338,
            -30.207814646427167,
            -21.053443606736987,
            -10.799109254706485,
            -11.095238305122946,
            -4.937194815722565,
            -14.343287579721144,
            5.7399411332880605,
            -3.416948806531707,
            5.989846631742381,
            4.7274361306966295,
            -12.772431120408738,
            9.058315105931596,
            4.758933713767598,
            -0.19555726458711623,
            -4.410097080728456,
            -0.3116066952754415,
            -1.974270413031487,
            4.705510844596311,
            -1.254292129620859,
            2.8718113189560146,
            -0.06341559977900978,
            0.5229006113802808,
            -2.3098049202556084,
            1.1457337390542648,
            -1.8610479944340859,
            0.8060270197354641,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            -2.8265718627156424,
            2.2892310107147704,
            -3.0783324653618016,
            0.9532008967922976,
            -1.978107751094217,
            1.7881612574523873,
            -2.5921128875831703,
            2.5923384455632887,
            -0.8180705457255297,
            0.48585268072215515,
            -4.038045853799597,
            -10.991083651085244,
            -21.020990160396966,
            -3.6952831563575903,
            12.078991416881324,
            4.179719978678812,
            19.862641029990364,
            -8.686065692222133,
            -8.686834790954538,
            22.752013924603936,
            11.28202228650398,
            -11.238145244837787,
            -19.53987944565921,
            19.11147403380924,
            14.434252097698034,
            -11.160436009091388,
            8.833747925349877,
            51.60826143842063,
            -10.411325384376909,
            -37.18791767820852,
            30.812379183926243,
            5.023142537221745,
            -24.507634617992842,
            11.973298037636141,
            16.89677234481153,
            -21.931302711734684,
            -0.35604704451416536,
            -5.119380426071667,
            1.6108523381129447,
            9.155410440736441,
            23.826554754362412,
            21.285085002238745,
            -2.6590554499172283,
            -4.6506698278630925,
            -23.22233768897672,
            -15.151715254874563,
            8.506619040444969,
            -12.697173620825218,
            -15.743484775694665,
            -25.62895348175754,
            -3.4007117534774944,
            -1.2169978629379026,
            3.5224917754758303,
            7.862995951277201,
            0.012204250536399472,
            2.824302049416479,
            2.041202514293427,
            -0.09627281172328661,
            -3.057903982317636,
            1.6884101378863505,
            -2.982866940967545,
            1.679231022549084,
            -2.8004175568614698,
            2.199268583870916,
        ],
        dtype=np.float64,
    ),
    np.array(
        [
            0.23283331461251955,
            -0.47649843497134514,
            0.13849078276376972,
            -0.6453379206026436,
            0.4357846008268049,
            1.739783871489161,
            -0.5726383673747019,
            1.8305696225902803,
            4.577236064191227,
            -1.5373171584887366,
            -0.38109906402501215,
            6.364696247504971,
            5.039419968647369,
            -0.06471453420699533,
            -0.32636348064118187,
            1.477613742442925,
            9.526788109947114,
            5.5852142952961055,
            -0.5420104443895387,
            -2.3617761790793077,
            -5.525317844432793,
            12.496211847898763,
            -11.692736993070714,
            4.387928076450001,
            -11.535693713483496,
            -10.754810715601732,
            1.7688276269623382,
            -20.44038018952811,
            18.938983532539638,
            -3.865674615806695,
            -2.2098355245473504,
            -1.4666199136454792,
            1.9003212640830358,
            2.4446727300800757,
            3.706994720579282,
            -10.72385795336814,
            -16.405940719889678,
            5.7085414013902165,
            5.840236463262041,
            18.109501067300027,
            5.102988353192747,
            10.346145585389223,
            -5.9950527806988525,
            1.2905776303487493,
            3.216817024045076,
            -3.8625236749494274,
            -9.249940581885784,
            -0.5913772393403968,
            2.471272619613949,
            -6.970351799120855,
            0.7383345873098159,
            -1.0390033262019445,
            -1.0193003251855093,
            -6.493634615452121,
            -1.9891022579165596,
            1.042420726547397,
            2.82567233482515,
            -0.5072286691067144,
            0.49220448209621576,
            0.24725021752538764,
            0.3095949695941485,
            -0.7276701177449313,
            0.182231278448932,
            -0.5423500050377005,
        ],
        dtype=np.float64,
    ),
)


def test_population_file_realizations_are_bit_for_bit_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The population-file path draws what it always drew, stream position included.

    The pinned numbers were produced before ``snr`` existed. Routing the population
    draw through the new specification -- or drawing a target unconditionally -- would
    move the waveforms, the targets or the state the stream is left in.

    What the random stream decides is pinned exactly: the full generator state after
    the draws, three consecutive probes of it, the seed handed to gengli, the drawn
    target and the class. Those are integer arithmetic in PCG64 and identical on every
    platform. The waveforms and realized SNRs are pinned sample by sample at a relative
    tolerance of 1e-12, because they pass through ``exp``, ``log`` and an FFT whose
    last bits may differ between the platforms the suite runs on. That tolerance is a
    few thousand ulps; any change to what is drawn or in which order moves every
    sample by order one, so it cannot hide behind it.
    """
    psd_file = tmp_path / "psd.txt"
    population_file = tmp_path / "population.h5"
    _write_positive_psd(psd_file)
    write_blip_population_file(population_file, snr_samples=np.array([7.0, 11.0, 23.0]))
    seeds: list[int] = []

    class RecordingGenerator(_SeededGenerator):
        def get_glitch(self, *, seed: int, snr: float, srate: float, glitch_type: str) -> np.ndarray:
            seeds.append(seed)
            return _SeededGenerator.get_glitch(self, seed=seed, snr=snr, srate=srate, glitch_type=glitch_type)

    _install_generator(monkeypatch, RecordingGenerator())
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

    assert seeds == [3139255103, 2372820508, 4032004037]
    for draw, expected in zip(draws, POPULATION_PIN_WAVEFORMS, strict=True):
        assert draw.waveform.dtype == np.float64
        assert draw.waveform.shape == expected.shape == (64,)
        np.testing.assert_allclose(draw.waveform, expected, rtol=1e-12, atol=1e-12 * float(np.max(np.abs(expected))))
    assert [draw.amplitude for draw in draws] == pytest.approx(
        [0.7165360788212201, 1.1110086679465017, 1.2220225349736609], rel=1e-12
    )
    assert rng.bit_generator.state == {
        "bit_generator": "PCG64",
        "state": {
            "state": 174314073433689513517317652637121482939,
            "inc": 306950386964110764321915108463935786421,
        },
        "has_uint32": 0,
        "uinteger": 2355802781,
    }
    probes = [rng.random() for _ in range(3)]
    assert probes[0] == pytest.approx(0.5899998082309823, rel=1e-12)
    assert probes == [0.5899998082309823, 0.2707395834904839, 0.8881301256116383]


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
