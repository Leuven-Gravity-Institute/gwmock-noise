"""Fit and continuation tests for the ARMA / state-space noise simulator.

The M4b contract for the ARMA generator is parallel to the Levinson AR file:
the fit exposes its orders, state size and per-band PSD residual; pole
placement anchors spectral lines; the generated band powers match the target;
the model is stationary after burn-in and resumable; and a fit that cannot be
done raises instead of degrading.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from gwmock_noise.simulators import ARMANoiseSimulator
from gwmock_noise.simulators._fit import FitError
from gwmock_noise.simulators._spectral import load_spectral_series

ALIGO_PSD = "aLIGO_O4_high_projected_psd"
ET_PSD = "ET_D_psd"
SAMPLING_FREQUENCY = 4096.0


def _synthetic_line_target(
    *,
    sampling_frequency: float = 512.0,
    line_frequency: float = 100.0,
    line_width: float = 0.5,
    n_samples: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a smooth band-limited target carrying one narrow line."""
    frequencies = np.fft.rfftfreq(2 * (n_samples - 1), d=1.0 / sampling_frequency)
    band = (frequencies >= 5.0) & (frequencies <= 200.0)
    values = np.zeros_like(frequencies)
    values[band] = 1.0e-3 * (100.0 / frequencies[band]) ** 2
    half_width = line_width / 2.0
    resonance = np.exp(-0.5 * ((frequencies - line_frequency) / half_width) ** 2)
    values += 1.0e-2 * resonance
    return frequencies, values


def _band_errors(  # noqa: PLR0913
    frequencies: np.ndarray,
    target: np.ndarray,
    estimate: np.ndarray,
    *,
    low: float,
    high: float,
    n_bands: int = 8,
) -> list[float]:
    """Return the relative band-integrated error of an estimate against a target."""
    edges = np.geomspace(low, high, n_bands + 1)
    errors = []
    for band_low, band_high in pairwise(edges):
        selected = (frequencies >= band_low) & (frequencies < band_high)
        if not np.any(selected):
            continue
        reference = float(np.sum(target[selected]))
        errors.append(abs(float(np.sum(estimate[selected])) - reference) / reference)
    return errors


@pytest.mark.parametrize(
    ("psd_file", "low_frequency"),
    [(ALIGO_PSD, 20.0), (ET_PSD, 5.0)],
)
def test_fitted_psd_matches_target_per_band(psd_file: str, low_frequency: float) -> None:
    """The tabulated aLIGO and ET-D curves (lines included) are reproduced per band."""
    simulator = ARMANoiseSimulator(
        psd_file=psd_file,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        ar_order=192,
        ma_order=32,
        low_frequency_cutoff=low_frequency,
    )
    frequencies, target = simulator.target_psd_curve
    model = simulator.model_psd_curve
    errors = _band_errors(frequencies, target, model, low=low_frequency, high=2000.0)
    assert errors
    assert max(errors) <= 0.6


@pytest.mark.parametrize(
    ("psd_file", "low_frequency"),
    [(ALIGO_PSD, 20.0), (ET_PSD, 5.0)],
)
def test_generated_psd_matches_target_per_band(psd_file: str, low_frequency: float) -> None:
    """A long realization recovers the target band powers, not just the model curve."""
    simulator = ARMANoiseSimulator(
        psd_file=psd_file,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        ar_order=192,
        ma_order=32,
        low_frequency_cutoff=low_frequency,
    )
    strains = [simulator.generate(32.0, SAMPLING_FREQUENCY, ["H1"], seed=seed)["H1"] for seed in range(6)]
    frequencies, estimate = welch(np.concatenate(strains), fs=SAMPLING_FREQUENCY, nperseg=16384, noverlap=8192)
    target_frequencies, target_values = load_spectral_series(psd_file, kind="PSD")
    target_on_grid = np.interp(frequencies, target_frequencies, target_values, left=0.0, right=0.0)
    errors = _band_errors(frequencies, target_on_grid, estimate, low=low_frequency, high=2000.0)
    assert max(errors) <= 0.6


def test_metadata_records_orders_state_and_residual() -> None:
    """The metadata exposes the state size, both orders and the per-band residual."""
    simulator = ARMANoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        ar_order=128,
        ma_order=16,
        low_frequency_cutoff=20.0,
    )
    described = simulator.metadata["autoregressive_moving_average"]
    assert described["fit_method"] == "pole-placement-log-spectrum-matching"
    assert described["ar_order"] == 128
    assert described["ma_order"] == 16
    assert described["state_bytes"] > 0
    assert described["state_size"] == max(128, 16 + 1)
    residual = described["fit_residual"]
    assert residual["bands"]
    assert residual["worst_relative_error"] == max(band["relative_error"] for band in residual["bands"])
    json.dumps(simulator.metadata)


def test_pole_placement_anchors_a_spectral_line() -> None:
    """A requested line frequency becomes a pole at that angle in the denominator."""
    frequencies, values = _synthetic_line_target()
    simulator = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=48,
        ma_order=16,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
        line_frequencies=[100.0],
        line_widths=[0.5],
    )
    placed = simulator.metadata["autoregressive_moving_average"]["placed_lines"]
    assert [line["frequency"] for line in placed] == [100.0]
    denominator = np.concatenate(([1.0], simulator.denominator_coefficients))
    pole_angles = np.abs(np.angle(np.roots(denominator)))
    target_angle = 2.0 * np.pi * 100.0 / 512.0
    assert np.min(np.abs(pole_angles - target_angle)) < 1.0e-3


def test_pole_placement_adds_a_resonance_at_the_line_frequency() -> None:
    """The placed pole produces a model peak at the requested line frequency."""
    frequencies, values = _synthetic_line_target()
    simulator = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=64,
        ma_order=64,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
        line_frequencies=[100.0],
        line_widths=[2.0],
    )
    fitted_frequencies, target = simulator.target_psd_curve
    model = simulator.model_psd_curve
    line_window = (fitted_frequencies >= 99.0) & (fitted_frequencies <= 101.0)
    local_index = np.argmax(model[line_window])
    peak_frequency = fitted_frequencies[line_window][local_index]
    assert abs(peak_frequency - 100.0) < 0.5
    assert model[line_window].max() > np.median(target[line_window])
    radius = simulator.metadata["autoregressive_moving_average"]["conditioning"]["max_placed_pole_radius"]
    assert 0.0 < radius < 1.0


def test_model_is_stationary_after_burn_in() -> None:
    """Post burn-in segments share their band powers within sampling scatter."""
    simulator = ARMANoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=1024.0,
        ar_order=96,
        ma_order=16,
        low_frequency_cutoff=20.0,
        high_frequency_cutoff=400.0,
    )
    strain = simulator.generate(40.0, 1024.0, ["H1"], seed=3)["H1"]
    burn_in = int(2.0 * 1024.0)
    segment_length = (strain.size - burn_in) // 5
    band_powers = []
    for index in range(5):
        start = burn_in + index * segment_length
        segment = strain[start : start + segment_length]
        frequencies, psd = welch(segment, fs=1024.0, nperseg=2048, noverlap=1024)
        selected = (frequencies >= 20.0) & (frequencies <= 400.0)
        band_powers.append(float(np.sum(psd[selected]) * (frequencies[1] - frequencies[0])))
    band_powers = np.asarray(band_powers)
    assert np.max(band_powers) / np.min(band_powers) <= 1.6


def test_chunked_generation_is_bit_identical_to_single_shot() -> None:
    """Splitting a run into calls does not change the samples."""
    frequencies, values = _synthetic_line_target()
    single = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=32,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
    )
    chunked = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=32,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
    )
    one_shot = single.generate(2.0, 512.0, ["H1"], seed=11)["H1"]
    pieces = [chunked.generate(0.5, 512.0, ["H1"], seed=11 if index == 0 else None)["H1"] for index in range(4)]
    np.testing.assert_array_equal(one_shot, np.concatenate(pieces))


def test_resume_from_exported_state_is_bit_identical() -> None:
    """A stream stopped and resumed from its exported state matches an uninterrupted run."""
    frequencies, values = _synthetic_line_target()

    def make() -> ARMANoiseSimulator:
        return ARMANoiseSimulator(
            target_psd=values,
            target_frequencies=frequencies,
            detectors=["H1"],
            sampling_frequency=512.0,
            ar_order=32,
            ma_order=8,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=200.0,
        )

    uninterrupted = make()
    full = [uninterrupted.generate(0.5, 512.0, ["H1"], seed=17 if index == 0 else None)["H1"] for index in range(5)]

    stopped = make()
    head = [stopped.generate(0.5, 512.0, ["H1"], seed=17 if index == 0 else None)["H1"] for index in range(3)]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    tail = [resumed.generate(0.5, 512.0, ["H1"])["H1"] for _ in range(2)]

    for expected, actual in zip(full, head + tail, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_constructor_raises_when_the_denominator_cannot_be_stabilised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recursion that cannot guarantee stable poles must raise."""
    from gwmock_noise.simulators import arma

    def unstable(autocovariance: np.ndarray, order: int) -> object:
        raise FitError("Fitted AR denominator is unstable")

    monkeypatch.setattr(arma, "levinson_durbin", unstable)
    frequencies, values = _synthetic_line_target()
    with pytest.raises(FitError, match="unstable"):
        ARMANoiseSimulator(
            target_psd=values,
            target_frequencies=frequencies,
            detectors=["H1"],
            sampling_frequency=512.0,
            ar_order=32,
            ma_order=8,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=200.0,
        )


def test_missing_psd_raises(tmp_path: Path) -> None:
    """A missing PSD is reported through the documented loader error."""
    with pytest.raises(FileNotFoundError):
        ARMANoiseSimulator(
            psd_file=tmp_path / "absent.txt",
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def _write_flat_psd(path: Path, *, sampling_frequency: float = 256.0, n_points: int = 257) -> Path:
    """Write a flat PSD covering the detector band."""
    frequencies = np.linspace(0.0, sampling_frequency / 2.0, n_points)
    values = np.full_like(frequencies, 2.0e-3)
    np.savetxt(path, np.column_stack((frequencies, values)))
    return path


def test_arma_rejects_conflicting_or_missing_targets(tmp_path: Path) -> None:
    """Exactly one of psd_file or target_psd must be given."""
    psd_path = _write_flat_psd(tmp_path / "flat.txt")
    with pytest.raises(ValueError, match="Exactly one"):
        ARMANoiseSimulator(detectors=["H1"], sampling_frequency=256.0)
    with pytest.raises(ValueError, match="Exactly one"):
        ARMANoiseSimulator(psd_file=psd_path, target_psd=np.ones(8), detectors=["H1"])


def test_arma_validates_orders_and_block_size() -> None:
    """Order and block-size validation is enforced at construction."""
    with pytest.raises(ValueError, match="at least one"):
        ARMANoiseSimulator(target_psd=np.ones(16), ar_order=0, ma_order=0, detectors=["H1"])
    with pytest.raises(ValueError, match="non-negative"):
        ARMANoiseSimulator(target_psd=np.ones(16), ar_order=-1, detectors=["H1"])
    with pytest.raises(ValueError, match="block_size"):
        ARMANoiseSimulator(target_psd=np.ones(16), block_size=0, detectors=["H1"])
    with pytest.raises(ValueError, match="regularization"):
        ARMANoiseSimulator(target_psd=np.ones(16), regularization=-1.0, detectors=["H1"])


def test_arma_rejects_mismatched_line_widths() -> None:
    """line_widths is either empty or exactly as long as line_frequencies."""
    with pytest.raises(ValueError, match="line_widths"):
        ARMANoiseSimulator(
            target_psd=np.ones(16),
            detectors=["H1"],
            line_frequencies=[1.0, 2.0],
            line_widths=[0.5],
        )


def test_arma_rejects_a_line_outside_the_band() -> None:
    """A placed line frequency must lie inside the fitted band."""
    frequencies, values = _synthetic_line_target()
    with pytest.raises(FitError, match="outside the fitted band"):
        ARMANoiseSimulator(
            target_psd=values,
            target_frequencies=frequencies,
            detectors=["H1"],
            sampling_frequency=512.0,
            ar_order=16,
            ma_order=8,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=200.0,
            line_frequencies=[300.0],
        )


def test_arma_detects_a_prominent_line() -> None:
    """detect_lines places a pole at the most prominent target peak."""
    frequencies, values = _synthetic_line_target()
    simulator = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=64,
        ma_order=64,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
        detect_lines=True,
        max_lines=4,
        line_prominence=2.0,
    )
    placed = [line["frequency"] for line in simulator.metadata["autoregressive_moving_average"]["placed_lines"]]
    assert placed
    assert min(abs(frequency - 100.0) for frequency in placed) < 1.0


def test_arma_from_component_builds_from_options(tmp_path: Path) -> None:
    """The registry factory reads its orders and PSD from the component options."""
    from gwmock_noise.config import NoiseComponentConfig, NoiseConfig

    psd_path = _write_flat_psd(tmp_path / "flat.txt")
    component = NoiseComponentConfig(
        simulator="arma",
        options={
            "psd_file": str(psd_path),
            "ar_order": 16,
            "ma_order": 8,
            "low_frequency_cutoff": 5.0,
            "high_frequency_cutoff": 120.0,
        },
    )
    config = NoiseConfig(detectors=["H1"], duration=1.0, sampling_frequency=256.0)
    simulator = ARMANoiseSimulator.from_component(component, config)
    assert simulator.ar_order == 16
    assert simulator.ma_order == 8
    assert simulator.generate(0.5, 256.0, ["H1"], seed=1)["H1"].shape == (128,)


def test_arma_from_component_requires_a_target() -> None:
    """A component without a target is refused."""
    from gwmock_noise.config import NoiseComponentConfig, NoiseConfig

    component = NoiseComponentConfig(simulator="arma", options={})
    config = NoiseConfig(detectors=["H1"], duration=1.0, sampling_frequency=256.0)
    with pytest.raises(ValueError, match="psd_file"):
        ARMANoiseSimulator.from_component(component, config)


def test_arma_reconfigures_when_sampling_frequency_changes() -> None:
    """Changing the runtime sampling frequency refits the filter."""
    simulator = ARMANoiseSimulator(
        psd_file="ET_D_psd",
        detectors=["H1"],
        sampling_frequency=1024.0,
        ar_order=32,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
    )
    first = simulator.generate(0.5, 1024.0, ["H1"], seed=1)["H1"]
    second = simulator.generate(0.5, 512.0, ["H1"], seed=1)["H1"]
    assert first.shape == (512,)
    assert second.shape == (256,)


def test_arma_state_nbytes_grows_with_orders() -> None:
    """A larger state-space model costs more continuation bytes."""
    sizes = []
    for ar_order, ma_order in ((16, 4), (64, 16), (128, 32)):
        simulator = ARMANoiseSimulator(
            psd_file="ET_D_psd",
            detectors=["H1"],
            sampling_frequency=256.0,
            ar_order=ar_order,
            ma_order=ma_order,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=120.0,
        )
        simulator.generate(0.25, 256.0, ["H1"], seed=1)
        sizes.append(simulator.state_nbytes)
    assert all(later > earlier for earlier, later in pairwise(sizes))


def test_arma_generate_rejects_a_sub_sample_duration() -> None:
    """A duration that rounds to zero samples is refused."""
    simulator = ARMANoiseSimulator(
        psd_file="ET_D_psd",
        detectors=["H1"],
        sampling_frequency=256.0,
        ar_order=16,
        ma_order=4,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    with pytest.raises(ValueError, match="at least one sample"):
        simulator.generate(1.0e-9, 256.0, ["H1"])


def test_arma_export_state_rejects_mismatched_orders() -> None:
    """A snapshot from a different order pair is refused."""
    frequencies, values = _synthetic_line_target()
    exporter = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=16,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
    )
    exporter.generate(0.25, 512.0, ["H1"], seed=2)
    snapshot = exporter.export_state()
    other = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1"],
        sampling_frequency=512.0,
        ar_order=32,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
    )
    with pytest.raises(ValueError, match="orders"):
        other.import_state(snapshot)


def test_arma_resume_metadata_describes_the_filter() -> None:
    """The resume metadata carries the settings a stopped stream needs."""
    frequencies, values = _synthetic_line_target()
    simulator = ARMANoiseSimulator(
        target_psd=values,
        target_frequencies=frequencies,
        detectors=["H1", "L1"],
        sampling_frequency=512.0,
        ar_order=16,
        ma_order=8,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=200.0,
        seed=5,
    )
    simulator.generate(0.25, 512.0, ["H1", "L1"], seed=5)
    resume = simulator.resume_metadata()
    assert resume["ar_order"] == 16
    assert resume["ma_order"] == 8
    assert set(resume["rng_state"]) == {"H1", "L1"}
    assert simulator.resume_metadata_nbytes > 0
