"""Tests for evaluating the covariance error of the colored-noise crossfade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from gwmock_noise.cli.main import app
from gwmock_noise.data.covariance_tolerance import load_crossfade_errors
from gwmock_noise.diagnostics import crossfade_covariance
from gwmock_noise.diagnostics.crossfade_covariance import (
    crossfade_covariance_error,
    crossfade_target,
    simulator_settings,
)
from gwmock_noise.simulators import colored
from gwmock_noise.simulators._spectral import _resolve_bundled_psd

#: The cheapest bundled curves to evaluate, with the analysis band their table entries hold.
ET_CURVES = {"ET_D_psd": (5.0, 2048.0), "ET_10_full_cryo_psd": (5.0, 2048.0)}


def _released(preset: str, window_duration: float) -> dict:
    """Return the bundled table's row of a curve at 4096 Hz and one window."""
    rows = load_crossfade_errors()["spectra"][preset]["sampling_frequencies"]["4096"]["windows"]
    return next(row for row in rows if row["window_duration"] == window_duration)


@pytest.mark.parametrize("preset", sorted(ET_CURVES))
def test_reproduces_the_released_error(preset: str) -> None:
    """The evaluation reproduces the published error exactly, over the band the table records."""
    error = crossfade_covariance_error(preset, 4096.0, 8.0, ET_CURVES[preset])
    released = _released(preset, 8.0)
    assert error.error_band.total == released["eps_total"]
    assert error.converged is released["converged"] is True
    table_band = load_crossfade_errors()["spectra"][preset]["sampling_frequencies"]["4096"]["error_band_hz"]
    assert list(error.error_band.band_hz) == table_band
    assert error.analysis_band.band_hz == (5.0, 2048.0)


#: Every part of the 8 s errors at 4096 Hz in the data release (``results/exact/<curve>__specified__4096.json``
#: of doi:10.5281/zenodo.23243286), where the bundled table keeps only the total. The aLIGO curve exercises the
#: power-law continuation below its first point; the ET curve, tabulated from 1 Hz, the flat one.
RELEASED_PARTS = {
    "ET_D_psd": {
        "band": (5.0, 2048.0),
        "eps_total": 0.01769854080351865,
        "eps_stationary": 0.01746175566509891,
        "eps_cyclic": 0.002885383105620598,
        "eps_total_to_nyquist": 0.025007729138732792,
        "harmonics": 128,
    },
    "aLIGO_O4_high_projected_psd": {
        "band": (10.0, 2048.0),
        "eps_total": 0.05366739647722309,
        "eps_stationary": 0.024612055510936027,
        "eps_cyclic": 0.04769104914100822,
        "eps_total_to_nyquist": 0.05366739648990233,
        "harmonics": 1024,
    },
}


@pytest.mark.parametrize("preset", sorted(RELEASED_PARTS))
def test_every_part_matches_the_release(preset: str) -> None:
    """The stationary and cyclic parts, both bands and the harmonics summed equal the release's."""
    released = RELEASED_PARTS[preset]
    error = crossfade_covariance_error(preset, 4096.0, 8.0, released["band"])
    assert error.error_band.total == released["eps_total"]
    assert error.error_band.stationary == released["eps_stationary"]
    assert error.error_band.cyclic == released["eps_cyclic"]
    assert error.analysis_band.total == released["eps_total_to_nyquist"]
    assert error.harmonics == released["harmonics"]
    assert error.converged is True


def test_a_sum_stopped_at_the_cap_is_not_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cyclic sum cut at the cap omits part of the error, so it must not be reported as converged."""
    converged = crossfade_covariance_error("ET_D_psd", 4096.0, 8.0, (5.0, 2048.0))
    assert converged.harmonics > 4
    monkeypatch.setattr(crossfade_covariance, "MAXIMUM_HARMONICS", 4)
    capped = crossfade_covariance_error("ET_D_psd", 4096.0, 8.0, (5.0, 2048.0))
    assert capped.harmonics == 4
    assert capped.converged is False
    assert capped.error_band.cyclic < converged.error_band.cyclic
    assert capped.error_band.stationary == converged.error_band.stationary


def test_an_odd_window_is_refused() -> None:
    """With an odd number of samples the overlap is shorter than the stride, outside the derivation."""
    with pytest.raises(ValueError, match="even number"):
        crossfade_covariance_error("ET_D_psd", 4096.0, 8.0 + 1.0 / 4096.0, (5.0, 2048.0))


def test_the_result_names_the_window_the_simulator_used() -> None:
    """A duration that is not a whole number of samples is reported as the window actually evaluated."""
    error = crossfade_covariance_error("ET_D_psd", 4096.0, 409.7 / 4096.0, (5.0, 2048.0))
    assert error.window_duration == 410 / 4096.0


@pytest.mark.parametrize(
    ("band", "message"),
    [
        ((0.0, 2048.0), "0 < low < high"),
        ((10.0, 5.0), "0 < low < high"),
        ((5.0, 2049.0), "0 < low < high"),
        ((2045.0, 2048.0), "within 4.0 Hz of the Nyquist"),
        ((0.5, 2048.0), "taper"),
    ],
)
def test_an_invalid_band_is_refused(band: tuple[float, float], message: str) -> None:
    """A band outside ``(0, f_s / 2]``, or wholly within the Nyquist guard, is refused."""
    with pytest.raises(ValueError, match=message):
        crossfade_covariance_error("ET_D_psd", 4096.0, 8.0, band)


def test_a_table_that_stops_below_nyquist_is_evaluated_over_a_band_inside_it(tmp_path: Path) -> None:
    """The target stops with the table, without the ramp to zero at Nyquist a closing point would add."""
    table = np.loadtxt(_resolve_bundled_psd("ET_D_psd"))
    short = tmp_path / "short.txt"
    np.savetxt(short, table[table[:, 0] <= 1000.0])
    frequencies, _ = crossfade_target(short, 4096.0)
    assert frequencies[-1] == table[table[:, 0] <= 1000.0][-1, 0]
    error = crossfade_covariance_error(short, 2048.0, 4.0, (5.0, 990.0))
    assert np.isfinite(error.error_band.total)
    with pytest.raises(ValueError, match="extends beyond the PSD table"):
        crossfade_covariance_error(short, 2048.0, 4.0, (5.0, 1020.0))


@pytest.mark.parametrize(("value", "message"), [(0.0, "all positive"), (-1e-47, "non-negative")])
def test_a_table_the_continuation_cannot_use_is_refused(tmp_path: Path, value: float, message: str) -> None:
    """A zero in the fitted octave would make the target not-a-number, and a negative PSD is no PSD."""
    table = np.loadtxt(_resolve_bundled_psd("aLIGO_O4_high_projected_psd"))
    table[5, 1] = value
    path = tmp_path / "bad.txt"
    np.savetxt(path, table)
    assert table[5, 0] < 2.0 * table[0, 0]
    with pytest.raises(ValueError, match=message):
        crossfade_covariance_error(path, 4096.0, 8.0, (10.0, 2048.0))


@pytest.mark.parametrize("sampling_frequency", [0.0, -4096.0, float("nan"), float("inf")])
def test_a_sampling_frequency_that_is_not_positive_and_finite_is_refused(sampling_frequency: float) -> None:
    """The public helpers refuse a rate no simulator can have, rather than return a table for it."""
    with pytest.raises(ValueError, match="sampling_frequency"):
        crossfade_target("ET_D_psd", sampling_frequency)
    with pytest.raises(ValueError, match="sampling_frequency"):
        simulator_settings(sampling_frequency, 4.0)
    with pytest.raises(ValueError, match="sampling_frequency"):
        crossfade_covariance_error("ET_D_psd", sampling_frequency, 4.0, (5.0, 100.0))


def test_the_target_of_a_target_is_itself(tmp_path: Path) -> None:
    """The written target starts at 0 Hz, and continuing it again changes nothing."""
    frequencies, psd = crossfade_target("ET_D_psd", 1024.0)
    path = tmp_path / "target.txt"
    np.savetxt(path, np.column_stack([frequencies, psd]), fmt="%.17e")
    again = crossfade_target(path, 1024.0)
    assert np.array_equal(again[0], frequencies)
    assert np.array_equal(again[1], psd)


def test_the_evaluated_simulator_is_the_one_the_settings_describe(monkeypatch: pytest.MonkeyPatch) -> None:
    """The error is evaluated for the published target and settings, so passing them reproduces it."""
    built: list[dict] = []
    original = colored.ColoredNoiseSimulator

    def spy(**kwargs: object) -> colored.ColoredNoiseSimulator:
        built.append({**kwargs, "target": np.loadtxt(kwargs["psd_file"])})
        return original(**kwargs)

    monkeypatch.setattr(colored, "ColoredNoiseSimulator", spy)
    crossfade_covariance_error("ET_D_psd", 1024.0, 4.0, (5.0, 512.0))
    [kwargs] = built
    settings = simulator_settings(1024.0, 4.0)
    assert {name: kwargs[name] for name in settings} == settings
    frequencies, psd = crossfade_target("ET_D_psd", 1024.0)
    # Evaluated at another scale, by an exact power of two: the same configuration.
    scale = kwargs["target"][:, 1].max() / psd.max()
    assert np.frexp(scale)[0] == 0.5
    assert np.array_equal(kwargs["target"], np.column_stack([frequencies, scale * psd]))


def test_a_curve_given_as_a_file_matches_it_given_by_name(tmp_path: Path) -> None:
    """A PSD is read the same from a path as from its bundled name."""
    bundled = _resolve_bundled_psd("ET_D_psd")
    assert bundled is not None
    copy = tmp_path / "copy.txt"
    copy.write_bytes(bundled.read_bytes())
    by_path = crossfade_covariance_error(copy, 4096.0, 8.0, (5.0, 2048.0))
    assert by_path.error_band.total == _released("ET_D_psd", 8.0)["eps_total"]


def test_the_cli_writes_the_error_with_its_provenance(tmp_path: Path) -> None:
    """The command records the version, the settings and the PSD's SHA-256 with each window's error."""
    out = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            "ET_D_psd",
            "--sampling-frequency",
            "4096",
            "--analysis-band",
            "5",
            "2048",
            "--window-duration",
            "8",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "8 s: eps_total = 0.0177 (converged)" in result.output
    record = json.loads(out.read_text(encoding="utf-8"))
    table = load_crossfade_errors()
    assert record["psd"]["sha256"] == table["spectra"]["ET_D_psd"]["sha256"]
    assert record["target"] == table["target"]
    assert record["sampling_frequency"] == 4096.0
    assert record["analysis_band_hz"] == [5.0, 2048.0]
    assert record["gwmock_noise_version"]
    assert record["method"]["maximum_harmonics"] == crossfade_covariance.MAXIMUM_HARMONICS
    [window] = record["windows"]
    assert window["window_duration"] == 8.0
    assert window["eps_total"] == _released("ET_D_psd", 8.0)["eps_total"]
    assert window["error_band_hz"] == [5.0, 2044.0]
    assert window["converged"] is True
    target = tmp_path / "result_target.txt"
    assert record["target_file"] == {"file": target.name, "sha256": crossfade_covariance.sha256_of(target)}
    assert window["simulator"] == {"psd_file": target.name, **simulator_settings(4096.0, 8.0)}
    frequencies, psd = crossfade_target("ET_D_psd", 4096.0)
    assert np.array_equal(np.loadtxt(target), np.column_stack([frequencies, psd]))
    assert "ColoredNoiseSimulator(psd_file=" in result.output


def test_the_cli_creates_the_output_directory_and_evaluates_each_window_once(tmp_path: Path) -> None:
    """A missing output directory is created before the evaluation, and a repeated window is evaluated once."""
    out = tmp_path / "new" / "deeper" / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            "ET_D_psd",
            "--sampling-frequency",
            "1024",
            "--analysis-band",
            "5",
            "512",
            "--window-duration",
            "4",
            "--window-duration",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(out.read_text(encoding="utf-8"))
    assert [window["window_duration"] for window in record["windows"]] == [4.0]


def test_the_cli_refuses_an_unknown_psd(tmp_path: Path) -> None:
    """An argument that is neither a file nor a bundled curve is a usage error, and nothing is written."""
    out = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            "no_such_psd",
            "--sampling-frequency",
            "4096",
            "--analysis-band",
            "5",
            "2048",
            "--window-duration",
            "8",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_the_cli_reports_an_invalid_band_as_a_usage_error(tmp_path: Path) -> None:
    """A configuration the evaluation refuses is reported as a usage error, not a traceback, and writes nothing."""
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            "ET_D_psd",
            "--sampling-frequency",
            "4096",
            "--analysis-band",
            "5",
            "3000",
            "--window-duration",
            "8",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )
    assert result.exit_code == 2
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert list(tmp_path.iterdir()) == []


def test_the_cli_reports_a_table_it_cannot_continue_as_a_usage_error(tmp_path: Path) -> None:
    """A PSD table the continuation refuses is a usage error on the PSD argument, and nothing is written."""
    table = np.loadtxt(_resolve_bundled_psd("ET_D_psd"))
    table[5, 1] = -1e-47
    psd = tmp_path / "bad.txt"
    np.savetxt(psd, table)
    out = tmp_path / "out" / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            str(psd),
            "--sampling-frequency",
            "1024",
            "--analysis-band",
            "5",
            "512",
            "--window-duration",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 2
    assert "PSD" in result.output
    assert not out.parent.exists()


def test_a_continuation_that_overflows_is_refused(tmp_path: Path) -> None:
    """A valid table whose lowest octave is steep enough to overflow below its first point gives no target."""
    path = tmp_path / "steep.txt"
    np.savetxt(path, [[10.0, 1e300], [20.0, 1e-300], [100.0, 1e-46], [2048.0, 1e-47]])
    with pytest.raises(ValueError, match="overflows"):
        crossfade_target(path, 4096.0)
    with pytest.raises(ValueError, match="overflows"):
        crossfade_covariance_error(path, 4096.0, 8.0, (10.0, 2048.0))


@pytest.mark.parametrize("factor", [1e-160, 1e10, 1e300])
def test_the_error_does_not_depend_on_the_psd_scale(tmp_path: Path, factor: float) -> None:
    """The error is relative to the target, so rescaling the PSD, however far, leaves it unchanged."""
    table = np.loadtxt(_resolve_bundled_psd("ET_D_psd"))
    table[:, 1] /= table[:, 1].max()
    table[:, 1] *= factor
    scaled = tmp_path / "scaled.txt"
    np.savetxt(scaled, table, fmt="%.17e")
    reference = crossfade_covariance_error("ET_D_psd", 1024.0, 4.0, (5.0, 512.0))
    error = crossfade_covariance_error(scaled, 1024.0, 4.0, (5.0, 512.0))
    assert np.isclose(error.error_band.total, reference.error_band.total, rtol=1e-10, atol=0.0)
    assert np.isclose(error.analysis_band.cyclic, reference.analysis_band.cyclic, rtol=1e-10, atol=0.0)


def test_a_constant_psd_far_from_unit_scale_gives_its_error(tmp_path: Path) -> None:
    """At scales whose squares underflow or overflow, the error is that of the same PSD at scale one."""
    errors = []
    for level in (1.0, 1e-160, 1e306, 1e-310):
        psd = tmp_path / f"flat_{level:g}.txt"
        np.savetxt(psd, [[10.0, level], [20.0, level], [100.0, level], [512.0, level]])
        errors.append(crossfade_covariance_error(psd, 1024.0, 4.0, (10.0, 500.0)).error_band.total)
    assert errors[0] > 1e-9
    # The error is itself relative, so its rounding floor is absolute: a flat PSD's tiny error differs in its last
    # digits between scales whose mantissas differ, by far less than this, while an underflow would zero it.
    assert np.allclose(errors[1:], errors[0], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("decades", [50, 160])
def test_a_band_far_below_the_rest_of_the_table_is_evaluated(tmp_path: Path, decades: int) -> None:
    """A band far below the table's peak, and at a tiny scale, gives the error of the same PSD at scale one."""
    errors = []
    for band_level in (1e-240, 1.0):
        psd = tmp_path / f"wall_{band_level:g}.txt"
        wall = band_level * 10.0**decades
        np.savetxt(psd, [[10.0, wall], [20.0, wall], [100.0, band_level], [512.0, band_level]])
        errors.append(crossfade_covariance_error(psd, 1024.0, 4.0, (150.0, 500.0)).error_band.total)
    assert np.isfinite(errors[0])
    assert np.isclose(errors[0], errors[1], rtol=1e-8, atol=0.0)


@pytest.mark.parametrize(("peak", "band_level"), [(1e90, 1e-90), (1e300, 1e-300)])
def test_a_table_too_wide_to_represent_is_refused_and_nothing_is_written(
    tmp_path: Path, peak: float, band_level: float
) -> None:
    """Too many decades between the band and the peak, whether they overflow or underflow, are refused, not nan."""
    psd = tmp_path / "wide.txt"
    np.savetxt(psd, [[10.0, peak], [20.0, peak], [100.0, band_level], [512.0, band_level]])
    with pytest.raises(ValueError, match="too many orders of magnitude"):
        crossfade_covariance_error(psd, 1024.0, 4.0, (150.0, 500.0))
    out = tmp_path / "out" / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            str(psd),
            "--sampling-frequency",
            "1024",
            "--analysis-band",
            "150",
            "500",
            "--window-duration",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 2, result.output
    assert list(out.parent.iterdir()) == []


def test_a_band_where_the_psd_is_zero_is_refused_with_the_cause(tmp_path: Path) -> None:
    """The refusal says the PSD is zero over the band, not only that a norm is undefined."""
    psd = tmp_path / "gap.txt"
    np.savetxt(psd, [[0.0, 1e-46], [100.0, 1e-46], [101.0, 0.0], [512.0, 0.0]])
    with pytest.raises(ValueError, match="PSD is zero throughout the band"):
        crossfade_covariance_error(psd, 1024.0, 4.0, (200.0, 500.0))
