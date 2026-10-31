"""Tests for choosing the crossfade window from a covariance-error tolerance.

The bundled table is copied from a published data release by
``scripts/build_covariance_tolerance_table.py``. These tests pin three separable things:

**The table describes the bundled curves.** Each entry carries the SHA-256 of the curve its
errors were evaluated for, so a curve edited after the evaluation fails here rather than
silently inheriting errors that no longer describe it.

**The selection.** The simulator takes the shortest evaluated window whose error meets the
tolerance, at the evaluated windows and between them, and refuses every configuration the
table does not cover: another PSD, another sampling frequency, a tolerance no window meets,
and a tolerance only an unconverged evaluation meets.

**The configuration.** The errors hold for the continued target over the full band, so the
simulator must adopt it and must refuse the settings that would change it.
"""

from __future__ import annotations

import copy
import hashlib
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from gwmock_noise.config import NoiseConfig, OutputConfig
from gwmock_noise.data.covariance_tolerance import load_crossfade_errors
from gwmock_noise.simulators import ColoredNoiseSimulator, UnsupportedCovarianceToleranceError
from gwmock_noise.simulators import _covariance_tolerance as covariance_tolerance
from gwmock_noise.simulators._covariance_tolerance import choose_crossfade_window, continued_target, error_band
from gwmock_noise.simulators._spectral import _resolve_bundled_psd

TABLE = load_crossfade_errors()
PRESETS = sorted(TABLE["spectra"])
O4 = "aLIGO_O4_high_projected_psd"
CONTINUATION = {"continuation_hz": 5.0, "slope_octave": 2.0}


def _rows(preset: str, sampling_frequency: int = 4096) -> list[dict]:
    """Return one curve's evaluated windows at one rate."""
    return TABLE["spectra"][preset]["sampling_frequencies"][str(sampling_frequency)]["windows"]


def test_the_table_names_its_source() -> None:
    """The table records the release it was copied from and the generator it evaluated."""
    assert TABLE["source"]["doi"] == "10.5281/zenodo.23243286"
    assert len(TABLE["source"]["generator_commit"]) == 40
    assert TABLE["target"] == {"continuation_hz": 5.0, "slope_octave": 2.0, "continuation_points": 256}


@pytest.mark.parametrize("preset", PRESETS)
def test_each_entry_matches_the_bundled_curve(preset: str) -> None:
    """The bundled curve is byte for byte the one its errors were evaluated for."""
    path = _resolve_bundled_psd(preset)
    assert path is not None
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TABLE["spectra"][preset]["sha256"]


@pytest.mark.parametrize("preset", PRESETS)
def test_each_rate_lists_windows_in_ascending_order(preset: str) -> None:
    """Windows ascend and every evaluated error is positive and finite."""
    for rate in TABLE["spectra"][preset]["sampling_frequencies"].values():
        windows = [row["window_duration"] for row in rate["windows"]]
        assert windows == sorted(windows)
        assert len(windows) == len(set(windows))
        assert all(math.isfinite(row["eps_total"]) and row["eps_total"] > 0 for row in rate["windows"])


@pytest.mark.parametrize("preset", PRESETS)
def test_a_tolerance_equal_to_an_evaluated_error_selects_that_window(preset: str) -> None:
    """At each converged evaluated window, a tolerance equal to its error returns it."""
    for row in _rows(preset):
        if not row["converged"]:
            continue
        choice = choose_crossfade_window(preset, 4096.0, row["eps_total"])
        assert choice.window_duration == row["window_duration"]
        assert choice.covariance_error == row["eps_total"]


def test_a_tolerance_between_evaluated_windows_rounds_up() -> None:
    """Between two evaluated windows the longer one is chosen, never an interpolated one."""
    rows = {row["window_duration"]: row["eps_total"] for row in _rows(O4)}
    between = math.sqrt(rows[16.0] * rows[64.0])
    assert rows[64.0] < between < rows[16.0]

    choice = choose_crossfade_window(O4, 4096.0, between)

    assert choice.window_duration == 64.0
    assert choice.covariance_error == rows[64.0]


def test_a_tolerance_met_only_where_the_evaluation_did_not_converge_is_refused() -> None:
    """On aLIGO O4 at 4096 Hz the 256 s error is converged and every longer one is not."""
    rows = {row["window_duration"]: row for row in _rows(O4)}
    assert rows[256.0]["converged"]
    assert not rows[1024.0]["converged"]
    tolerance = 0.5 * (rows[256.0]["eps_total"] + rows[1024.0]["eps_total"])

    with pytest.raises(UnsupportedCovarianceToleranceError, match="did not converge"):
        choose_crossfade_window(O4, 4096.0, tolerance)


def test_a_tolerance_no_window_meets_is_refused() -> None:
    """A tolerance below every evaluated error names the smallest converged one, not an unconverged one."""
    converged = min(row["eps_total"] for row in _rows(O4) if row["converged"])
    assert converged > min(row["eps_total"] for row in _rows(O4))
    with pytest.raises(UnsupportedCovarianceToleranceError, match="smallest converged evaluated error") as raised:
        choose_crossfade_window(O4, 4096.0, 1e-9)
    assert repr(converged) in str(raised.value)


@pytest.mark.parametrize("preset", PRESETS)
def test_every_tolerance_at_8192_hz_is_refused(preset: str) -> None:
    """At 8192 Hz only windows whose evaluation did not converge were evaluated."""
    assert not any(row["converged"] for row in _rows(preset, 8192))
    with pytest.raises(UnsupportedCovarianceToleranceError):
        choose_crossfade_window(preset, 8192.0, 0.5)


def test_an_unevaluated_sampling_frequency_is_refused() -> None:
    """The errors depend on the rate, so an unevaluated rate is refused, not borrowed."""
    with pytest.raises(UnsupportedCovarianceToleranceError, match="4096"):
        choose_crossfade_window(O4, 2048.0, 0.1)


def test_a_bundled_curve_the_table_does_not_cover_is_refused() -> None:
    """A bundled curve with no evaluated errors is refused and the covered curves are named."""
    assert _resolve_bundled_psd("aLIGO_O4_low_projected_psd") is not None
    with pytest.raises(UnsupportedCovarianceToleranceError, match="window_duration directly"):
        choose_crossfade_window("aLIGO_O4_low_projected_psd", 4096.0, 0.1)


def test_a_copy_of_a_bundled_curve_is_refused(tmp_path: Path) -> None:
    """A file at another path is refused even with the bundled name, since it can be edited."""
    bundled = _resolve_bundled_psd(O4)
    assert bundled is not None
    copy = shutil.copy(bundled, tmp_path / f"{O4}.txt")

    with pytest.raises(UnsupportedCovarianceToleranceError, match="No covariance errors are bundled"):
        choose_crossfade_window(copy, 4096.0, 0.1)


def test_a_mistyped_path_named_like_a_bundled_curve_is_refused(tmp_path: Path) -> None:
    """A path that does not exist is reported, not silently replaced by the bundled curve of that name."""
    with pytest.raises(UnsupportedCovarianceToleranceError, match="No covariance errors are bundled"):
        choose_crossfade_window(tmp_path / "missing" / f"{O4}.txt", 4096.0, 0.1)


def test_a_bundled_curve_whose_bytes_changed_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A curve whose SHA-256 differs from the one the errors were evaluated for is refused."""
    table = copy.deepcopy(TABLE)
    table["spectra"][O4]["sha256"] = "0" * 64
    monkeypatch.setattr(covariance_tolerance, "load_crossfade_errors", lambda: table)
    with pytest.raises(UnsupportedCovarianceToleranceError, match="differs from the one"):
        choose_crossfade_window(O4, 4096.0, 0.1)


@pytest.mark.parametrize("tolerance", [np.float32(0.1), np.float64(0.1), np.int64(1)])
def test_a_numpy_scalar_tolerance_is_accepted(tolerance: float) -> None:
    """A numpy number is a number."""
    assert choose_crossfade_window(O4, 4096.0, tolerance).window_duration == 8.0


@pytest.mark.parametrize("tolerance", [True, np.True_, "0.1", None])
def test_a_tolerance_that_is_not_a_number_is_rejected(tolerance: object) -> None:
    """A boolean or a string is not a tolerance, even where Python would convert it."""
    with pytest.raises(TypeError, match="covariance_tolerance must be a number"):
        choose_crossfade_window(O4, 4096.0, tolerance)  # type: ignore[arg-type]


@pytest.mark.parametrize("preset", PRESETS)
def test_the_recorded_bands_follow_the_error_band_rule(preset: str) -> None:
    """Every recorded band is the paper's analysis band cut by the Nyquist guard, from the one rule."""
    low = 10.0 if preset.startswith("aLIGO") else 5.0
    for key, rate in TABLE["spectra"][preset]["sampling_frequencies"].items():
        assert rate["error_band_hz"] == list(error_band((low, 2048.0), float(key)))


@pytest.mark.parametrize("tolerance", [0.0, -1e-3, math.nan, math.inf])
def test_a_tolerance_that_is_not_positive_and_finite_is_rejected(tolerance: float) -> None:
    """Zero, negative and non-finite tolerances are invalid input, not unsupported ones."""
    with pytest.raises(ValueError, match="covariance_tolerance") as raised:
        choose_crossfade_window(O4, 4096.0, tolerance)
    assert not isinstance(raised.value, UnsupportedCovarianceToleranceError)


def test_the_continuation_follows_a_power_law_and_is_flat_below_five_hertz() -> None:
    """For a curve that is a pure power law, the continuation is that power law exactly."""
    frequencies = np.linspace(10.0, 1000.0, 991)
    psd = 1e-40 * (frequencies / 10.0) ** -4.0

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=256
    )

    continued = (grid >= 5.0) & (grid < 10.0)
    np.testing.assert_allclose(target[continued], 1e-40 * (grid[continued] / 10.0) ** -4.0, rtol=1e-10, atol=0.0)
    np.testing.assert_array_equal(target[grid < 5.0], target[grid == 5.0])
    np.testing.assert_allclose(target[grid < 5.0], 1e-40 * 0.5**-4.0, rtol=1e-10, atol=0.0)
    assert grid[0] == 0.0
    assert np.count_nonzero(continued) == 255


def test_a_table_that_reaches_nyquist_ends_there() -> None:
    """A table beyond the Nyquist frequency is cut there, with the curve's value at the Nyquist frequency."""
    frequencies = np.linspace(10.0, 4000.0, 400)
    psd = 1e-46 * (1.0 + frequencies / 100.0)

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
    )

    assert grid[-1] == 2048.0
    assert target[-1] == np.interp(2048.0, frequencies, psd)


def test_a_table_that_stops_below_nyquist_stops_there() -> None:
    """No zero is appended at the Nyquist frequency, which would make the simulator ramp down to it."""
    frequencies = np.linspace(10.0, 1000.0, 991)
    psd = np.full_like(frequencies, 1e-46)

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
    )

    assert grid[-1] == 1000.0
    assert np.all(target > 0.0)


def test_a_table_from_zero_hertz_is_continued_and_the_continuation_is_idempotent() -> None:
    """A table starting at 0 Hz, such as a continued target itself, is held flat below 5 Hz and otherwise kept."""
    frequencies = np.linspace(1.0, 2048.0, 2048)
    psd = 1e-46 * (1.0 + 100.0 / frequencies)
    settings = {"continuation_hz": 5.0, "slope_octave": 2.0, "continuation_points": 256}
    once = continued_target(frequencies, psd, 4096.0, **settings)

    twice = continued_target(*once, 4096.0, **settings)

    assert once[0][0] == 0.0
    flat = np.interp(5.0, *once)
    np.testing.assert_array_equal(twice[1][twice[0] <= 5.0], flat)
    np.testing.assert_array_equal(np.interp(frequencies, *twice), np.interp(frequencies, *once))


@pytest.mark.parametrize(
    ("frequencies", "psd", "message"),
    [
        ([10.0], [1e-46], "at least two"),
        ([10.0, 20.0, 15.0], [1e-46, 1e-46, 1e-46], "ascend"),
        ([10.0, 15.0, 20.0, np.nan], [1e-44, 8e-45, 5e-45, 1e-46], "finite and non-negative"),
        ([-1.0, 10.0, 20.0], [1e-46, 1e-46, 1e-46], "finite and non-negative"),
        ([10.0, 20.0, 30.0], [1e-46, -1e-46, 1e-46], "non-negative at every point"),
        ([10.0, 20.0, 30.0], [1e-46, np.nan, 1e-46], "non-negative at every point"),
        ([10.0, 20.0, 30.0], [1e-46, 0.0, 1e-46], "two or more points there, all positive"),
        ([10.0, 100.0, 1000.0], [1e-44, 1e-48, 1e-52], "two or more points there, all positive"),
    ],
)
def test_a_table_that_cannot_be_continued_is_refused(frequencies: list, psd: list, message: str) -> None:
    """A table the log-log fit cannot use is refused rather than continued to a target of not-a-number."""
    with pytest.raises(ValueError, match=message):
        continued_target(
            np.array(frequencies), np.array(psd), 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
        )


def test_the_simulator_adopts_the_evaluated_configuration() -> None:
    """The window, the band and the target are chosen together, and the metadata records them."""
    rows = {row["window_duration"]: row["eps_total"] for row in _rows(O4)}

    simulator = ColoredNoiseSimulator(
        psd_file=O4, detectors=["H1"], sampling_frequency=4096.0, duration=1.0, seed=3, covariance_tolerance=2e-3
    )

    assert simulator.window_duration == 64.0
    assert simulator.low_frequency_cutoff == 0.0
    recorded = simulator.metadata["colored_noise"]["covariance_tolerance"]
    assert recorded == {
        "requested": 2e-3,
        "preset": O4,
        "evaluated_error": rows[64.0],
        "error_band_hz": [10.0, 2044.0],
        "target": {"first_tabulated_hz": float(np.loadtxt(_resolve_bundled_psd(O4))[0, 0]), **CONTINUATION},
        "source": "doi:10.5281/zenodo.23243286",
    }
    strain = simulator.generate(1.0, 4096.0, ["H1"])
    assert strain["H1"].shape == (4096,)
    assert np.all(np.isfinite(strain["H1"]))


def test_the_simulator_is_given_the_continued_target_below_the_curve() -> None:
    """Below the curve's first point the simulator's PSD is the continuation, not zero."""
    simulator = ColoredNoiseSimulator(
        psd_file=O4, detectors=["H1"], sampling_frequency=4096.0, duration=1.0, covariance_tolerance=2e-3
    )
    grid = simulator._frequency_grid
    first = float(np.loadtxt(_resolve_bundled_psd(O4))[0, 0])

    assert np.all(simulator._psd[(grid > 1.0) & (grid < first)] > 0.0)


@pytest.mark.parametrize(
    "conflict",
    [
        {"window_duration": 64.0},
        {"low_frequency_cutoff": 10.0},
        {"high_frequency_cutoff": 1024.0},
    ],
)
def test_settings_the_tolerance_chooses_cannot_also_be_given(conflict: dict[str, float]) -> None:
    """A window or cutoff given with a tolerance would leave the configuration the errors hold for."""
    with pytest.raises(ValueError, match="cannot be combined with") as raised:
        ColoredNoiseSimulator(psd_file=O4, sampling_frequency=4096.0, covariance_tolerance=1e-2, **conflict)
    assert not isinstance(raised.value, UnsupportedCovarianceToleranceError)


def test_a_tolerance_cannot_be_combined_with_a_psd_array() -> None:
    """Only a bundled curve, by name, carries evaluated errors."""
    with pytest.raises(ValueError, match="psd_array"):
        ColoredNoiseSimulator(
            psd_array=np.ones(4097), delta_frequency=1.0, sampling_frequency=4096.0, covariance_tolerance=1e-2
        )


def test_generating_at_another_rate_is_refused() -> None:
    """The errors hold at the rate the window was chosen for."""
    simulator = ColoredNoiseSimulator(
        psd_file=O4, detectors=["H1"], sampling_frequency=4096.0, duration=1.0, covariance_tolerance=2e-3
    )

    with pytest.raises(UnsupportedCovarianceToleranceError, match="4096"):
        simulator.generate(1.0, 2048.0, ["H1"])


def test_without_a_tolerance_the_defaults_are_unchanged() -> None:
    """The cutoff and window keep their defaults, and no tolerance is recorded."""
    simulator = ColoredNoiseSimulator(psd_file=O4, sampling_frequency=4096.0)

    assert simulator.low_frequency_cutoff == 2.0
    assert simulator.window_duration == 64
    assert simulator.metadata["colored_noise"]["covariance_tolerance"] is None


def test_a_configuration_file_can_request_a_tolerance(tmp_path: Path) -> None:
    """The option reaches the simulator through a component definition."""
    config = NoiseConfig(
        detectors=["H1"],
        duration=1.0,
        sampling_frequency=4096.0,
        components=[{"simulator": "colored", "psd_file": O4, "covariance_tolerance": 2e-3}],
        output=OutputConfig(directory=tmp_path),
    )

    simulator = ColoredNoiseSimulator.from_component(config.components[0], config)

    assert simulator.window_duration == 64.0
    assert simulator.metadata["colored_noise"]["covariance_tolerance"]["requested"] == 2e-3


def test_the_documented_table_quotes_the_bundled_errors() -> None:
    """The user guide's table of errors is the bundled table, rounded to two figures."""
    guide = (Path(__file__).resolve().parents[1] / "docs" / "user_guide" / "noise_simulation.md").read_text(
        encoding="utf-8"
    )
    section = guide.split("### Supported curves", 1)[1].split("###", 1)[0]
    lines = [line for line in section.splitlines() if line.startswith("| `")]
    documented = {}
    for line in lines:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        documented[cells[0].strip("`")] = cells[1:]

    assert sorted(documented) == PRESETS
    for preset, cells in documented.items():
        rows = _rows(preset)[:5]
        assert [row["window_duration"] for row in rows] == [8.0, 16.0, 64.0, 256.0, 1024.0]
        expected = [
            f"{row['eps_total']:.1e}".replace("e-0", "e-") + ("\\*" if not row["converged"] else "") for row in rows
        ]
        assert cells == expected


def test_a_curve_tabulated_below_five_hertz_is_replaced_there_by_the_flat_value() -> None:
    """The ET curves start at 1 Hz; below 5 Hz the target is flat at S(5 Hz), far below the tabulated wall."""
    simulator = ColoredNoiseSimulator(
        psd_file="ET_D_psd", detectors=["E1"], sampling_frequency=4096.0, duration=1.0, covariance_tolerance=2e-2
    )
    table = np.loadtxt(_resolve_bundled_psd("ET_D_psd"))
    assert table[0, 0] < CONTINUATION["continuation_hz"]
    grid = simulator._frequency_grid
    wall = (grid >= 1.5) & (grid < 4.5)
    flat = np.interp(5.0, table[:, 0], table[:, 1])
    assert np.allclose(simulator._psd[wall], flat, rtol=1e-12, atol=0.0)
    tabulated = np.interp(grid[wall], table[:, 0], table[:, 1])
    assert np.all(tabulated > flat)
    assert tabulated[0] > 1e6 * flat
    assert table[0, 1] > 1e13 * flat
    recorded = simulator.metadata["colored_noise"]["covariance_tolerance"]["target"]
    assert recorded == {"first_tabulated_hz": float(table[0, 0]), **CONTINUATION}


def test_a_zero_outside_the_fitted_octave_is_a_psd_of_zero() -> None:
    """Zeros at 0 Hz, or beyond the fitted octave, are part of the target, not an error."""
    frequencies = np.array([0.0, 1.0, 2.0, 100.0, 1000.0, 2048.0])
    psd = np.array([0.0, 1e-44, 8e-45, 1e-46, 0.0, 0.0])

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
    )

    assert np.all(np.isfinite(target))
    assert target[grid == 1000.0] == 0.0
    assert target[grid == 100.0] == 1e-46


def test_a_coarse_table_is_continued_by_a_slope_it_determines() -> None:
    """Two points in the fitted octave fix the power law, and the continuation follows it exactly."""
    frequencies = np.array([10.0, 20.0, 1000.0])
    psd = 1e-44 * (frequencies / 10.0) ** -4.0

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
    )

    np.testing.assert_allclose(target[grid == 5.0], 1e-44 * 0.5**-4.0, rtol=1e-12, atol=0.0)


def test_the_power_law_is_fitted_to_exactly_the_points_the_table_check_guards() -> None:
    """A point just above the fitted range is ignored by the fit, as by the check; one on its edge is used by both."""
    frequencies = np.array([10.0, 15.0, 20.0, 20.5, 1000.0])
    psd = 1e-44 * (frequencies / 10.0) ** -4.0
    psd[3] = 0.0

    grid, target = continued_target(
        frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8
    )

    np.testing.assert_allclose(target[grid == 5.0], 1e-44 * 0.5**-4.0, rtol=1e-12, atol=0.0)
    psd[2] = 0.0
    with pytest.raises(ValueError, match="all positive"):
        continued_target(frequencies, psd, 4096.0, continuation_hz=5.0, slope_octave=2.0, continuation_points=8)
