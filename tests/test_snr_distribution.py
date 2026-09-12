"""Tests for sampled target-SNR distributions on PSD-calibrated glitch models.

The measured reference these tests are written against is the Omicron SNR of the
seven Gravity Spy classes over LIGO's O3, whose survival above SNR 10 goes as
``s ** -alpha`` with a per-class index spanning 0.40 (Koi_Fish) to 3.11
(Fast_Scattering). What matters is not that a sampler was configured with such an
index but that the *generated strain* carries it, so every acceptance check here
recomputes the SNR from the waveform and estimates the index back out of it.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from gwmock_noise import (
    BlipGlitch,
    EmpiricalSNRDistribution,
    LogNormalAmplitudeDistribution,
    PowerLawSNRDistribution,
    ScatteredLightGlitch,
    SNRDistribution,
)
from gwmock_noise.glitches.models import normalize_glitch_models
from gwmock_noise.glitches.snr import (
    draw_target_snr,
    load_snr_samples,
    normalize_snr,
    parse_snr_distribution,
    serialize_snr,
)

# Measured Pareto tail indices of the Omicron SNR distribution above SNR 10,
# pooled over H1 and L1 across O3, for two classes at opposite ends of the range.
MEASURED_TAIL_INDEX = {"Koi_Fish": 0.40, "Fast_Scattering": 3.11}
MEASURED_THRESHOLD = 10.0


def _write_flat_psd(path: Path) -> None:
    """Write a strictly positive flat PSD covering the full test band."""
    frequencies = np.linspace(0.0, 4096.0, 129)
    np.savetxt(path, np.column_stack((frequencies, np.ones_like(frequencies))))


def _optimal_snr(waveform: np.ndarray, psd_file: Path, sampling_frequency: float) -> float:
    """Recompute the optimal SNR of a waveform against the raw PSD table."""
    table = np.loadtxt(psd_file)
    frequencies = np.fft.rfftfreq(waveform.size, d=1.0 / sampling_frequency)
    psd = np.interp(frequencies, table[:, 0], table[:, 1], left=0.0, right=0.0)
    valid = (frequencies >= 2.0) & (frequencies < sampling_frequency / 2.0) & (psd > 0.0)
    waveform_fd = np.fft.rfft(waveform) / sampling_frequency
    delta_frequency = sampling_frequency / waveform.size
    return float(np.sqrt(4.0 * delta_frequency * np.sum(np.abs(waveform_fd[valid]) ** 2 / psd[valid])))


def _binomial_standard_error(probability: float, n_draws: int) -> float:
    """Standard error of an observed fraction, for sizing a tolerance."""
    return float(np.sqrt(probability * (1.0 - probability) / n_draws))


def tail_index(snrs: np.ndarray, threshold: float) -> float:
    """Estimate the Pareto tail index of a sample above ``threshold``.

    The Hill/maximum-likelihood estimator ``(n - 1) / sum(log(s / threshold))``,
    bias-corrected for a finite sample, which recovers the survival exponent the
    measured table quotes. Its relative standard error is ``1 / sqrt(n)``, so the
    tolerances below are sized against the sample each test draws.
    """
    exceedances = np.asarray(snrs, dtype=float)
    exceedances = exceedances[exceedances >= threshold]
    return float((exceedances.size - 1) / np.sum(np.log(exceedances / threshold)))


def test_tail_index_estimator_recovers_a_known_index() -> None:
    """The estimator used by the acceptance checks is itself anchored.

    Drawn from numpy's own Pareto sampler rather than from the code under test,
    so a systematically wrong estimator cannot agree with a systematically wrong
    sampler and hide both.
    """
    rng = np.random.default_rng(3)
    reference = MEASURED_THRESHOLD * (1.0 + rng.pareto(1.34, size=20000))

    assert tail_index(reference, MEASURED_THRESHOLD) == pytest.approx(1.34, rel=0.03)


def test_power_law_sampler_matches_its_analytic_survival() -> None:
    """Draws reproduce ``S(s) = (s / minimum) ** -alpha`` at several decades."""
    distribution = PowerLawSNRDistribution(minimum=MEASURED_THRESHOLD, alpha=1.34)
    rng = np.random.default_rng(0)

    draws = np.array([distribution.sample(rng) for _ in range(40000)])

    assert draws.min() >= MEASURED_THRESHOLD
    for factor in (2.0, 10.0, 50.0):
        expected = factor**-1.34
        # The observed fraction is a binomial mean, so the tolerance has to be its
        # standard error rather than a flat relative one: at the deepest decade only
        # a few hundred draws survive the cut and a fixed 5 % would fail by chance.
        assert np.mean(draws >= factor * MEASURED_THRESHOLD) == pytest.approx(
            expected, abs=4.0 * _binomial_standard_error(expected, draws.size)
        )


def test_power_law_truncation_bounds_the_draw() -> None:
    """A configured maximum caps the draw and keeps the shape below it."""
    distribution = PowerLawSNRDistribution(minimum=10.0, alpha=0.4, maximum=100.0)
    rng = np.random.default_rng(1)

    draws = np.array([distribution.sample(rng) for _ in range(20000)])

    assert draws.min() >= 10.0
    assert draws.max() <= 100.0
    # Truncated survival at s: ((s/min)^-a - (max/min)^-a) / (1 - (max/min)^-a).
    survival_at_maximum = 10.0**-0.4
    expected = (5.0**-0.4 - survival_at_maximum) / (1.0 - survival_at_maximum)
    assert np.mean(draws >= 50.0) == pytest.approx(expected, abs=4.0 * _binomial_standard_error(expected, draws.size))


def test_empirical_distribution_draws_from_the_supplied_table() -> None:
    """Every draw is a table value, and each value comes up about equally often."""
    distribution = EmpiricalSNRDistribution(samples=[8.0, 16.0, 64.0])
    rng = np.random.default_rng(2)

    draws = np.array([distribution.sample(rng) for _ in range(3000)])

    assert set(np.unique(draws)) == {8.0, 16.0, 64.0}
    for value in (8.0, 16.0, 64.0):
        assert np.count_nonzero(draws == value) == pytest.approx(1000, abs=120)


def test_empirical_distribution_reads_text_and_hdf5_tables(tmp_path: Path) -> None:
    """A table of observed SNRs can be supplied as a file instead of inline."""
    text_file = tmp_path / "snrs.txt"
    np.savetxt(text_file, np.array([11.0, 22.0, 33.0]))
    hdf5_file = tmp_path / "snrs.h5"
    with h5py.File(hdf5_file, "w") as handle:
        handle.create_dataset("snr", data=np.array([11.0, 22.0, 33.0]))

    np.testing.assert_allclose(load_snr_samples(text_file), [11.0, 22.0, 33.0])
    np.testing.assert_allclose(load_snr_samples(hdf5_file), [11.0, 22.0, 33.0])

    rng = np.random.default_rng(0)
    for source in (text_file, hdf5_file):
        draws = {EmpiricalSNRDistribution(file=source).sample(rng) for _ in range(200)}
        assert draws == {11.0, 22.0, 33.0}


def test_single_value_text_table_is_accepted(tmp_path: Path) -> None:
    """A one-row text table loads as a table, not as a scalar."""
    text_file = tmp_path / "one.txt"
    text_file.write_text("13.5\n")

    assert load_snr_samples(text_file).shape == (1,)
    assert EmpiricalSNRDistribution(file=text_file).sample(np.random.default_rng(0)) == 13.5


def test_distribution_configuration_errors(tmp_path: Path) -> None:
    """Invalid distribution configurations are refused at construction."""
    with pytest.raises(ValueError, match="minimum must be finite and greater than zero"):
        PowerLawSNRDistribution(minimum=0.0, alpha=1.0)
    with pytest.raises(ValueError, match="alpha must be finite and greater than zero"):
        PowerLawSNRDistribution(minimum=10.0, alpha=-1.0)
    with pytest.raises(ValueError, match="maximum must be greater than minimum"):
        PowerLawSNRDistribution(minimum=10.0, alpha=1.0, maximum=10.0)
    with pytest.raises(TypeError, match="alpha must be a number"):
        PowerLawSNRDistribution(minimum=10.0, alpha="steep")
    with pytest.raises(ValueError, match="requires distribution='power_law'"):
        PowerLawSNRDistribution(minimum=10.0, alpha=1.0, distribution="empirical")

    with pytest.raises(ValueError, match="exactly one of 'samples' or 'file'"):
        EmpiricalSNRDistribution()
    with pytest.raises(ValueError, match="exactly one of 'samples' or 'file'"):
        EmpiricalSNRDistribution(samples=[1.0], file=tmp_path / "snrs.txt")
    with pytest.raises(ValueError, match="at least one SNR sample"):
        EmpiricalSNRDistribution(samples=[])
    with pytest.raises(ValueError, match="greater than zero"):
        EmpiricalSNRDistribution(samples=[5.0, -1.0])
    with pytest.raises(ValueError, match="only finite SNR samples"):
        EmpiricalSNRDistribution(samples=[5.0, float("inf")])
    with pytest.raises(TypeError, match="sequence of numbers"):
        EmpiricalSNRDistribution(samples=[5.0, "loud"])
    with pytest.raises(FileNotFoundError, match="SNR sample file not found"):
        EmpiricalSNRDistribution(file=tmp_path / "absent.txt")

    empty_hdf5 = tmp_path / "empty.h5"
    with h5py.File(empty_hdf5, "w") as handle:
        handle.create_dataset("other", data=np.array([1.0]))
    with pytest.raises(ValueError, match="must contain the 'snr' dataset"):
        EmpiricalSNRDistribution(file=empty_hdf5)

    with pytest.raises(ValueError, match="requires distribution='empirical'"):
        EmpiricalSNRDistribution(samples=[5.0], distribution="power_law")
    with pytest.raises(TypeError, match="sequence of numbers"):
        EmpiricalSNRDistribution(samples="loud")

    two_dimensional = tmp_path / "grid.h5"
    with h5py.File(two_dimensional, "w") as handle:
        handle.create_dataset("snr", data=np.ones((2, 3)))
    with pytest.raises(ValueError, match="must be one-dimensional"):
        EmpiricalSNRDistribution(file=two_dimensional)

    with pytest.raises(ValueError, match="distribution must be one of"):
        normalize_snr({"distribution": "gaussian", "mean": 10.0})
    with pytest.raises(TypeError, match="numbers or a distribution mapping"):
        normalize_snr("loud")
    with pytest.raises(ValueError, match="finite and greater than zero"):
        normalize_snr(0.0)
    with pytest.raises(TypeError, match="snr distribution must be a mapping"):
        parse_snr_distribution(12.0)


def test_base_distribution_is_abstract() -> None:
    """The base class defines the contract without implementing it."""
    with pytest.raises(NotImplementedError):
        SNRDistribution().sample(np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        SNRDistribution().serialize()


def test_a_distribution_still_needs_a_psd_to_calibrate_against(tmp_path: Path) -> None:
    """A sampled target is refused without a PSD, exactly as a fixed one is."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.0)

    with pytest.raises(ValueError, match="snr requires psd_file"):
        BlipGlitch(rate=0.5, amplitude_distribution=amplitude, snr=PowerLawSNRDistribution(minimum=10.0, alpha=1.3))
    with pytest.raises(ValueError, match="values must be finite and greater than zero"):
        BlipGlitch(rate=0.5, amplitude_distribution=amplitude, psd_file=psd_file, snr=0.0)
    with pytest.raises(ValueError, match="values must be finite and greater than zero"):
        ScatteredLightGlitch(rate=0.5, amplitude_distribution=amplitude, psd_file=psd_file, snr=-3.0)


def test_scalar_specification_consumes_no_randomness() -> None:
    """A fixed target draws nothing, so it cannot perturb any other draw."""
    rng = np.random.default_rng(0)
    reference = np.random.default_rng(0).random()

    assert draw_target_snr(12.0, rng) == 12.0
    assert rng.random() == reference


def test_serialize_round_trips_through_a_configuration(tmp_path: Path) -> None:
    """A serialized distribution reconstructs the model that produced it."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    table_file = tmp_path / "snrs.txt"
    np.savetxt(table_file, np.array([9.0, 18.0]))

    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        psd_file=psd_file,
        snr={"distribution": "power_law", "minimum": 10.0, "alpha": 1.34, "maximum": 600.0},
    )
    serialized = model.serialize()
    assert serialized["snr"] == {
        "distribution": "power_law",
        "minimum": 10.0,
        "alpha": 1.34,
        "maximum": 600.0,
    }

    replayed = normalize_glitch_models([serialized | {"kind": "blip"}])[0]
    assert replayed.snr == PowerLawSNRDistribution(minimum=10.0, alpha=1.34, maximum=600.0)
    np.testing.assert_array_equal(
        replayed.generate_waveform(4096.0, rng=np.random.default_rng(4)),
        model.generate_waveform(4096.0, rng=np.random.default_rng(4)),
    )

    # A file-backed empirical table replays as a reference to the file, not as a
    # copy of its contents pasted into the metadata.
    assert serialize_snr(EmpiricalSNRDistribution(file=table_file)) == {
        "distribution": "empirical",
        "file": str(table_file),
    }
    assert serialize_snr(EmpiricalSNRDistribution(samples=[9.0, 18.0])) == {
        "distribution": "empirical",
        "samples": [9.0, 18.0],
    }
    assert serialize_snr(12.0) == 12.0


def test_blip_glitch_strain_carries_the_configured_tail_index(tmp_path: Path) -> None:
    """A parametric model's *generated strain* reproduces the requested tail.

    The index is estimated from SNRs recomputed off each waveform, never from the
    target the sampler drew, so a model that sampled correctly and then calibrated
    to something else would fail here.
    """
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        width=0.01,
        psd_file=psd_file,
        snr=PowerLawSNRDistribution(minimum=MEASURED_THRESHOLD, alpha=MEASURED_TAIL_INDEX["Fast_Scattering"]),
    )

    rng = np.random.default_rng(17)
    recovered = np.array(
        [_optimal_snr(model.generate_waveform(4096.0, rng=rng), psd_file, 4096.0) for _ in range(2000)]
    )

    assert recovered.min() >= MEASURED_THRESHOLD
    assert tail_index(recovered, MEASURED_THRESHOLD) == pytest.approx(MEASURED_TAIL_INDEX["Fast_Scattering"], rel=0.1)


def test_scattered_light_glitch_records_the_drawn_target(tmp_path: Path) -> None:
    """The per-event target reaches the draw and matches the strain it produced."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = ScatteredLightGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.0),
        duration=0.5,
        psd_file=psd_file,
        snr=EmpiricalSNRDistribution(samples=[6.0, 60.0]),
    )

    rng = np.random.default_rng(5)
    draws = [model._draw(4096.0, rng=rng) for _ in range(40)]

    assert {draw.target_snr for draw in draws} == {6.0, 60.0}
    for draw in draws:
        assert _optimal_snr(draw.waveform, psd_file, 4096.0) == pytest.approx(draw.target_snr, rel=1e-8)
        assert draw.realized_snr == pytest.approx(draw.target_snr, rel=1e-8)


def test_amplitude_multiplier_scales_the_sampled_target(tmp_path: Path) -> None:
    """The amplitude distribution keeps multiplying a sampled target, as before."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=3.0, std=0.0),
        width=0.01,
        psd_file=psd_file,
        snr=EmpiricalSNRDistribution(samples=[7.0]),
    )

    draw = model._draw(4096.0, rng=np.random.default_rng(0))

    assert draw.target_snr == 7.0
    assert _optimal_snr(draw.waveform, psd_file, 4096.0) == pytest.approx(21.0, rel=1e-8)


def test_sampled_targets_replay_for_a_fixed_seed(tmp_path: Path) -> None:
    """A sampled model is as reproducible as the fixed-SNR path it generalizes."""
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    model = BlipGlitch(
        rate=0.5,
        amplitude_distribution=LogNormalAmplitudeDistribution(mean=1.0, std=0.3),
        width=0.01,
        psd_file=psd_file,
        snr=PowerLawSNRDistribution(minimum=10.0, alpha=1.34),
    )

    first = [model._draw(4096.0, rng=rng) for rng in [np.random.default_rng(9)] for _ in range(5)]
    second = [model._draw(4096.0, rng=rng) for rng in [np.random.default_rng(9)] for _ in range(5)]

    assert [draw.target_snr for draw in first] == [draw.target_snr for draw in second]
    assert len({draw.target_snr for draw in first}) == 5
    for one, other in zip(first, second, strict=True):
        np.testing.assert_array_equal(one.waveform, other.waveform)


def test_scalar_snr_realizations_are_bit_for_bit_unchanged(tmp_path: Path) -> None:
    """The fixed-SNR path still draws what it always drew, stream position included.

    The pinned numbers were produced before sampled targets existed. A scalar
    target must consume nothing from the generator, so the values and the state
    it is left in both have to survive; drawing the target unconditionally would
    move every later draw and fail this.
    """
    psd_file = tmp_path / "psd.txt"
    _write_flat_psd(psd_file)
    amplitude = LogNormalAmplitudeDistribution(mean=1.0, std=0.25)

    blip = BlipGlitch(rate=0.5, amplitude_distribution=amplitude, width=0.01, psd_file=psd_file, snr=12.0)
    rng = np.random.default_rng(20260912)
    draws = [blip._draw(4096.0, rng=rng) for _ in range(3)]

    assert [draw.waveform[0] for draw in draws] == pytest.approx(
        [2.851834153233524, -1.2954497177224333, 1.6639663060170926], rel=1e-12
    )
    assert [draw.realized_snr for draw in draws] == pytest.approx(
        [14.730393760815133, 10.7861296277195, 11.563621264479705], rel=1e-12
    )
    assert [draw.target_snr for draw in draws] == [12.0, 12.0, 12.0]
    assert rng.random() == pytest.approx(0.4363133640580499, rel=1e-12)

    scattered = ScatteredLightGlitch(rate=0.5, amplitude_distribution=amplitude, psd_file=psd_file, snr=7.0)
    rng = np.random.default_rng(20260912)
    scattered_draws = [scattered._draw(4096.0, rng=rng) for _ in range(3)]

    assert [draw.realized_snr for draw in scattered_draws] == pytest.approx(
        [9.014425851434668, 7.576197426068159, 8.653078586946055], rel=1e-12
    )
    assert rng.random() == pytest.approx(0.9820447395788969, rel=1e-12)
