"""Fit and continuation tests for the Levinson-Durbin autoregressive simulator.

The tests in this file pin the M4b contract for on the Levinson AR generator:

* the fit reports its conditioning diagnostics and a per-band PSD residual;
* stability is guaranteed by construction, never repaired after the fact;
* the generated PSD matches the tabulated target per band on the bundled
  aLIGO and ET-D curves, which carry the detector lines;
* the model is stationary after burn-in and resumable without cached strain.

Numbers quoted in the docstrings are the values observed while writing the
tests, so a later failure says what moved.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import welch

from gwmock_noise.simulators import ARNoiseSimulator
from gwmock_noise.simulators._fit import FitError, band_fit_residual, levinson_durbin
from gwmock_noise.simulators._spectral import load_spectral_series

ALIGO_PSD = "aLIGO_O4_high_projected_psd"
ET_PSD = "ET_D_psd"
SAMPLING_FREQUENCY = 4096.0

#: The fit order the tolerances below were measured at.
FIT_ORDER = 256

#: An order low enough that the aLIGO fitted-curve bound must reject it; the
#: measured worst band there is 0.1048, against 0.0439 at :data:`FIT_ORDER`.
UNDERFITTED_ORDER = 224

#: Anchored per-band tolerances. Each is ``1.25 * mean + 5 * sd`` of the
#: measured worst-band population, rounded up to two decimal places; see
#: ``docs/dev/anchored_quantities.md``.
ALIGO_FITTED_TOLERANCE = 0.06
ET_FITTED_TOLERANCE = 0.41
ALIGO_GENERATED_TOLERANCE = 0.27
ET_GENERATED_TOLERANCE = 0.66


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


def _welch_psd(strain: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a one-sided PSD with a long segment for a low-variance estimate."""
    return welch(strain, fs=sampling_frequency, nperseg=16384, noverlap=8192)


@pytest.mark.parametrize(
    ("psd_file", "low_frequency", "tolerance"),
    [(ALIGO_PSD, 20.0, ALIGO_FITTED_TOLERANCE), (ET_PSD, 5.0, ET_FITTED_TOLERANCE)],
)
def test_fitted_psd_matches_target_per_band(psd_file: str, low_frequency: float, tolerance: float) -> None:
    """The tabulated target is reproduced band by band across the fit band.

    The bound is measured, not chosen: the worst band of this deterministic fit
    is 0.0439 on aLIGO O4-high and 0.3203 on ET-D, and the tolerance carries
    25 % headroom over that. See ``docs/dev/anchored_quantities.md``.
    """
    simulator = ARNoiseSimulator(
        psd_file=psd_file,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=FIT_ORDER,
        low_frequency_cutoff=low_frequency,
    )
    frequencies, target = simulator.target_psd_curve
    model = simulator.model_psd_curve
    errors = _band_errors(frequencies, target, model, low=low_frequency, high=2000.0)
    assert errors
    assert max(errors) <= tolerance


@pytest.mark.parametrize(
    ("psd_file", "low_frequency", "tolerance"),
    [(ALIGO_PSD, 20.0, ALIGO_GENERATED_TOLERANCE), (ET_PSD, 5.0, ET_GENERATED_TOLERANCE)],
)
def test_generated_psd_matches_target_per_band(psd_file: str, low_frequency: float, tolerance: float) -> None:
    """A long realization recovers the target band powers, not just the model curve.

    The bound is measured over twenty independent seed groups of this same
    configuration: the worst band is 0.1061 +/- 0.0260 on aLIGO O4-high and
    0.4429 +/- 0.0210 on ET-D, and the tolerance is 1.25 times the mean plus
    five standard deviations. See ``docs/dev/anchored_quantities.md``.
    """
    simulator = ARNoiseSimulator(
        psd_file=psd_file,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=FIT_ORDER,
        low_frequency_cutoff=low_frequency,
    )
    strains = [simulator.generate(32.0, SAMPLING_FREQUENCY, ["H1"], seed=seed)["H1"] for seed in range(6)]
    frequencies, estimate = welch(np.concatenate(strains), fs=SAMPLING_FREQUENCY, nperseg=16384, noverlap=8192)
    target_frequencies, target_values = load_spectral_series(psd_file, kind="PSD")
    target_on_grid = np.interp(frequencies, target_frequencies, target_values, left=0.0, right=0.0)
    errors = _band_errors(frequencies, target_on_grid, estimate, low=low_frequency, high=2000.0)
    assert max(errors) <= tolerance


def test_fitted_tolerance_rejects_an_underfitted_model() -> None:
    """The aLIGO fitted-curve bound is tight enough to reject a lower-order fit.

    A tolerance that nothing can breach asserts nothing. The same fit at order
    224 instead of 256 has a measured worst band of 0.1048 on aLIGO O4-high,
    comfortably above the anchored 0.06 -- and below the 0.5 this bound
    replaced, which is why that one said nothing. Any widening of
    :data:`ALIGO_FITTED_TOLERANCE` past the underfitted value fails here.
    """
    simulator = ARNoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=UNDERFITTED_ORDER,
        low_frequency_cutoff=20.0,
    )
    frequencies, target = simulator.target_psd_curve
    errors = _band_errors(frequencies, target, simulator.model_psd_curve, low=20.0, high=2000.0)
    assert max(errors) > ALIGO_FITTED_TOLERANCE


def test_fit_records_levinson_conditioning_diagnostics() -> None:
    """The fit names its method and reports the pre-registered conditioning numbers."""
    simulator = ARNoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=256,
        low_frequency_cutoff=20.0,
    )
    fitted = simulator.metadata["autoregressive_noise"]
    assert fitted["fit_method"] == "levinson-durbin"
    conditioning = fitted["conditioning"]
    assert conditioning["max_reflection_coefficient"] < 1.0
    assert conditioning["min_prediction_error"] > 0.0
    assert conditioning["toeplitz_condition_number"] > 1.0
    assert conditioning["max_reflection_limit"] == 1.0


def test_model_poles_are_inside_the_unit_circle() -> None:
    """The recursion's reflection coefficients place every pole strictly inside."""
    simulator = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=256,
        low_frequency_cutoff=5.0,
    )
    denominator = np.concatenate(([1.0], simulator.autoregressive_coefficients))
    assert np.max(np.abs(np.roots(denominator))) < 1.0


def test_fit_records_per_band_residual() -> None:
    """The metadata carries the relative PSD error per band."""
    simulator = ARNoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=256,
        low_frequency_cutoff=20.0,
    )
    residual = simulator.metadata["autoregressive_noise"]["fit_residual"]
    assert residual["bands"]
    assert residual["band_count"] == len(residual["bands"])
    assert residual["worst_relative_error"] == max(band["relative_error"] for band in residual["bands"])
    for band in residual["bands"]:
        assert band["relative_error"] >= 0.0


def test_state_bytes_are_exposed_and_bounded() -> None:
    """The continuation state is the order-sized history, and its byte size is reported."""
    simulator = ARNoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1", "L1"],
        sampling_frequency=SAMPLING_FREQUENCY,
        order=64,
        low_frequency_cutoff=20.0,
    )
    simulator.generate(1.0, SAMPLING_FREQUENCY, ["H1", "L1"], seed=1)
    fitted = simulator.metadata["autoregressive_noise"]
    assert fitted["state_size"] == 64
    state_bytes = fitted["state_bytes"]
    assert state_bytes > 0
    assert simulator.state_nbytes == state_bytes
    for _ in range(20):
        simulator.generate(1.0, SAMPLING_FREQUENCY, ["H1", "L1"])
    assert simulator.state_nbytes <= state_bytes + 64


def test_chunked_generation_is_bit_identical_to_single_shot() -> None:
    """Splitting a run into calls does not change the samples."""
    single = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=32,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    chunked = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=32,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    one_shot = single.generate(2.0, 256.0, ["H1"], seed=7)["H1"]
    pieces = [chunked.generate(0.5, 256.0, ["H1"], seed=7 if index == 0 else None)["H1"] for index in range(4)]
    np.testing.assert_array_equal(one_shot, np.concatenate(pieces))


def test_resume_from_exported_state_is_bit_identical() -> None:
    """A stream stopped and resumed from its exported state matches an uninterrupted run."""

    def make() -> ARNoiseSimulator:
        return ARNoiseSimulator(
            psd_file=ET_PSD,
            detectors=["H1"],
            sampling_frequency=256.0,
            order=32,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=120.0,
        )

    uninterrupted = make()
    full = [uninterrupted.generate(0.5, 256.0, ["H1"], seed=13 if index == 0 else None)["H1"] for index in range(5)]

    stopped = make()
    head = [stopped.generate(0.5, 256.0, ["H1"], seed=13 if index == 0 else None)["H1"] for index in range(3)]
    snapshot = stopped.export_state()

    resumed = make()
    resumed.import_state(snapshot)
    tail = [resumed.generate(0.5, 256.0, ["H1"])["H1"] for _ in range(2)]

    for expected, actual in zip(full, head + tail, strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_model_is_stationary_after_burn_in() -> None:
    """Post burn-in segments share their band powers within sampling scatter."""
    simulator = ARNoiseSimulator(
        psd_file=ALIGO_PSD,
        detectors=["H1"],
        sampling_frequency=1024.0,
        order=128,
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


def test_levinson_durbin_returns_stable_filter_for_a_valid_covariance() -> None:
    """A positive-definite autocovariance yields reflection coefficients below one."""
    autocovariance = np.array([3.0, 2.0, 1.0])
    fit = levinson_durbin(autocovariance, 2)
    assert fit.coefficients.shape == (2,)
    assert np.max(np.abs(fit.reflection_coefficients)) < 1.0
    assert fit.prediction_error > 0.0
    assert np.all(np.diff(fit.prediction_errors) < 0.0)


def test_levinson_durbin_rejects_a_non_positive_definite_covariance() -> None:
    """An invalid covariance sequence is a fit failure, not a silent clamp."""
    with pytest.raises(FitError, match="reflection coefficient"):
        levinson_durbin(np.array([1.0, 2.0]), 1)


def test_constructor_raises_when_the_recursion_is_not_stable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fit that cannot guarantee stability must raise, never fall back."""
    from gwmock_noise.simulators import autoregressive

    def unstable(autocovariance: np.ndarray, order: int) -> object:
        raise FitError("Fitted model is unstable")

    monkeypatch.setattr(autoregressive, "levinson_durbin", unstable)
    with pytest.raises(FitError, match="unstable"):
        ARNoiseSimulator(
            psd_file=ET_PSD,
            detectors=["H1"],
            sampling_frequency=256.0,
            order=16,
            low_frequency_cutoff=5.0,
            high_frequency_cutoff=120.0,
        )


def test_band_fit_residual_reports_integrated_and_per_bin_errors() -> None:
    """The residual helper compares an estimate with a target band by band."""
    frequencies = np.linspace(1.0, 9.0, 9)
    target = np.ones_like(frequencies)
    estimate = 2.0 * target
    residual = band_fit_residual(frequencies, target, estimate, low_frequency=1.0, high_frequency=9.0, n_bands=2)
    assert len(residual["bands"]) == 2
    assert residual["worst_relative_error"] == pytest.approx(1.0)


def test_missing_psd_raises_fit_error(tmp_path: Path) -> None:
    """A missing PSD is still reported through the documented loader error."""
    with pytest.raises(FileNotFoundError):
        ARNoiseSimulator(
            psd_file=tmp_path / "absent.txt",
            detectors=["H1"],
            sampling_frequency=256.0,
        )


def test_metadata_is_json_serializable() -> None:
    """The metadata sidecar writer receives only plain JSON types."""
    simulator = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=32,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    json.dumps(simulator.metadata)


def test_levinson_durbin_handles_order_zero() -> None:
    """Order zero is a valid degenerate fit with a single prediction error."""
    fit = levinson_durbin(np.array([2.0]), 0)
    assert fit.coefficients.shape == (0,)
    assert fit.prediction_error == pytest.approx(2.0)
    assert fit.prediction_errors.shape == (1,)


def test_levinson_durbin_validates_its_inputs() -> None:
    """Short, non-finite, zero-variance and negative-order inputs are refused."""
    with pytest.raises(FitError, match="non-negative"):
        levinson_durbin(np.array([1.0]), -1)
    with pytest.raises(FitError, match="r\\[0\\] through r\\[order\\]"):
        levinson_durbin(np.array([1.0]), 3)
    with pytest.raises(FitError, match="finite"):
        levinson_durbin(np.array([1.0, np.nan]), 1)
    with pytest.raises(FitError, match="must be positive"):
        levinson_durbin(np.array([0.0, 0.0]), 1)


def test_toeplitz_condition_number_and_band_edges() -> None:
    """The conditioning and band helpers return the documented shapes."""
    from gwmock_noise.simulators._fit import geometric_band_edges, toeplitz_condition_number

    assert toeplitz_condition_number(np.array([1.0]), 0) == 1.0
    assert toeplitz_condition_number(np.array([1.0, 0.5]), 2) >= 1.0
    edges = geometric_band_edges(1.0, 100.0, 3)
    assert edges.shape == (4,)
    assert edges[0] == pytest.approx(1.0)
    assert edges[-1] == pytest.approx(100.0)


def test_export_state_rejects_mismatched_orders() -> None:
    """A snapshot from a different order is refused."""
    exporter = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=32,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    exporter.generate(0.5, 256.0, ["H1"], seed=2)
    snapshot = exporter.export_state()
    other = ARNoiseSimulator(
        psd_file=ET_PSD,
        detectors=["H1"],
        sampling_frequency=256.0,
        order=64,
        low_frequency_cutoff=5.0,
        high_frequency_cutoff=120.0,
    )
    with pytest.raises(ValueError, match="order"):
        other.import_state(snapshot)


def test_ar_simulator_rejects_a_negative_psd_sample(tmp_path: Path) -> None:
    """A negative PSD sample is a fit failure, not silently clamped to zero."""
    psd_path = tmp_path / "negative_psd.txt"
    frequencies = np.linspace(0.0, 128.0, 513)
    values = np.full_like(frequencies, 2.0e-3)
    values[256] = -1.0e-3
    np.savetxt(psd_path, np.column_stack((frequencies, values)))
    with pytest.raises(FitError, match="non-negative"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
            order=16,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )


def test_ar_simulator_reports_zero_variance_as_a_fit_error(tmp_path: Path) -> None:
    """The zero-variance target is reported through the documented fit exception."""
    psd_path = tmp_path / "zero_variance.txt"
    frequencies = np.linspace(0.0, 128.0, 513)
    values = np.zeros_like(frequencies)
    np.savetxt(psd_path, np.column_stack((frequencies, values)))
    with pytest.raises(FitError, match="zero variance"):
        ARNoiseSimulator(
            psd_file=psd_path,
            detectors=["H1"],
            sampling_frequency=256.0,
            order=16,
            low_frequency_cutoff=8.0,
            high_frequency_cutoff=96.0,
        )
