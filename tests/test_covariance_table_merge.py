"""Tests for merging an evaluated covariance error into the bundled table.

A merge extends the table for the curves it covers, and records where each row comes from. It
must refuse a result that describes another configuration, never replace a converged row, and
replace an unconverged row only with a converged one. The table must not depend on the order in
which evaluations are merged, so that ``--merge`` writes the table ``--check`` rebuilds.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gwmock_noise.cli.main import app
from gwmock_noise.data.covariance_tolerance import describe_source, load_crossfade_errors
from gwmock_noise.data.covariance_tolerance._merge import (
    RELEASE_SOURCE,
    EvaluationMismatchError,
    add_evaluation,
    build_table,
    contributes,
    merge_evaluation,
    method_constants,
    source_id,
)
from gwmock_noise.simulators import ColoredNoiseSimulator
from gwmock_noise.simulators import _covariance_tolerance as covariance_tolerance

ET_D = "ET_D_psd"
O4 = "aLIGO_O4_high_projected_psd"


def _table() -> dict:
    """Return a fresh copy of the bundled table."""
    return copy.deepcopy(load_crossfade_errors())


def _record(preset: str = ET_D, sampling_frequency: float = 8192.0, windows: list | None = None) -> dict:
    """Return a result in the shape ``evaluate-covariance-error`` writes, for a bundled curve."""
    table = load_crossfade_errors()
    band = table["spectra"][preset]["analysis_band_hz"]
    error_band = covariance_tolerance.error_band((band[0], band[1]), sampling_frequency)
    if windows is None:
        windows = [(8.0, 0.02, True), (16.0, 0.01, True)]
    return {
        "gwmock_noise_version": "0.19.0",
        "psd": {"argument": preset, "file": f"{preset}.txt", "sha256": table["spectra"][preset]["sha256"]},
        "sampling_frequency": sampling_frequency,
        "analysis_band_hz": list(band),
        "target": table["target"],
        "method": method_constants(),
        "windows": [
            {"window_duration": duration, "eps_total": error, "converged": converged, "error_band_hz": list(error_band)}
            for duration, error, converged in windows
        ],
    }


def _merge(table: dict, record: dict) -> str:
    """Merge a record as if read from a file, and return its source id."""
    data = json.dumps(record).encode("utf-8")
    sha256 = hashlib.sha256(data).hexdigest()
    merge_evaluation(table, record, file="evaluations/result.json", sha256=sha256)
    return f"evaluation-{sha256[:12]}"


def _rows(table: dict, preset: str, rate: str) -> dict[float, dict]:
    """Return one curve's rows at one rate, by window."""
    return {row["window_duration"]: row for row in table["spectra"][preset]["sampling_frequencies"][rate]["windows"]}


def test_windows_at_a_new_rate_are_added_with_their_source() -> None:
    """The release's 8192 Hz entry gains the evaluated short windows, in order, each naming the evaluation."""
    table = _table()
    source = _merge(table, _record())

    rows = _rows(table, ET_D, "8192")
    assert list(rows) == [8.0, 16.0, 2048.0, 4096.0]
    assert rows[8.0] == {"window_duration": 8.0, "eps_total": 0.02, "converged": True, "source": source}
    assert rows[2048.0]["source"] == RELEASE_SOURCE
    assert table["sources"][source] == {
        "kind": "evaluation",
        "file": "evaluations/result.json",
        "sha256": table["sources"][source]["sha256"],
        "gwmock_noise_version": "0.19.0",
    }


def test_a_rate_the_table_lacks_gets_the_curve_band() -> None:
    """A rate with no entry is created, measured over the curve's band cut 4 Hz below Nyquist."""
    table = _table()
    _merge(table, _record(sampling_frequency=16384.0))
    assert table["spectra"][ET_D]["sampling_frequencies"]["16384"]["error_band_hz"] == [5.0, 2048.0]


def test_a_converged_evaluation_replaces_an_unconverged_row() -> None:
    """The release's unconverged 2048 s row at 4096 Hz is replaced by a converged evaluation of that window."""
    table = _table()
    assert not _rows(table, ET_D, "4096")[2048.0]["converged"]
    source = _merge(table, _record(sampling_frequency=4096.0, windows=[(2048.0, 0.003, True)]))
    row = _rows(table, ET_D, "4096")[2048.0]
    assert row == {"window_duration": 2048.0, "eps_total": 0.003, "converged": True, "source": source}


def test_an_unconverged_evaluation_does_not_replace_a_row() -> None:
    """An unconverged evaluation of an evaluated window keeps the release's row, and contributes nothing."""
    table = _table()
    before = copy.deepcopy(table)
    record = _record(sampling_frequency=4096.0, windows=[(2048.0, 0.003, False)])
    assert not contributes(table, record)
    _merge(table, record)
    assert table == before


def test_a_converged_row_is_never_replaced_and_must_be_agreed_with() -> None:
    """A converged evaluation of a converged window must agree with it; agreeing, it adds nothing."""
    table = _table()
    released = _rows(table, ET_D, "4096")[8.0]["eps_total"]
    with pytest.raises(EvaluationMismatchError, match="must agree") as raised:
        _merge(table, _record(sampling_frequency=4096.0, windows=[(8.0, released * (1 + 1e-6), True)]))
    assert "doi:10.5281/zenodo.23243286" in str(raised.value)
    agreeing = _record(sampling_frequency=4096.0, windows=[(8.0, released * (1 + 1e-12), True)])
    assert not contributes(table, agreeing)
    _merge(table, agreeing)
    assert _rows(table, ET_D, "4096")[8.0] == {
        "window_duration": 8.0,
        "eps_total": released,
        "converged": True,
        "source": RELEASE_SOURCE,
    }
    assert set(table["sources"]) == {RELEASE_SOURCE}


def test_a_refused_merge_leaves_the_table_unchanged() -> None:
    """A result refused for one window changes no window, even one it would have added."""
    table = _table()
    before = copy.deepcopy(table)
    released = _rows(table, ET_D, "4096")[8.0]["eps_total"]
    with pytest.raises(EvaluationMismatchError):
        _merge(table, _record(sampling_frequency=4096.0, windows=[(4.0, 0.05, True), (8.0, 2 * released, True)]))
    assert table == before


def _changed(field: str, value: object) -> dict:
    """Return a result with one top-level field replaced."""
    record = _record()
    record[field] = value
    return record


@pytest.mark.parametrize(
    ("record", "message"),
    [
        (_changed("psd", {"argument": "x", "file": "x.txt", "sha256": "0" * 64}), "none of the bundled curves"),
        (_changed("gwmock_noise_version", "0.18.4-post.3+8f8b4bb"), "not a plain release version"),
        (_changed("gwmock_noise_version", "0.19.0.dev1"), "not a plain release version"),
        (_changed("gwmock_noise_version", "0.19"), "not a plain release version"),
        (_changed("gwmock_noise_version", "1"), "not a plain release version"),
        (_changed("target", {"continuation_hz": 4.0, "slope_octave": 2.0, "continuation_points": 256}), "target"),
        (_changed("method", {**method_constants(), "maximum_harmonics": 65536}), "method constants"),
        (_changed("analysis_band_hz", [10.0, 2048.0]), "analysis band"),
        (_changed("sampling_frequency", 8192.5), "not a positive integer"),
        (_changed("windows", []), "no window"),
        (_record(windows=[(8.0, float("nan"), True)]), "not positive and finite"),
        (_record(windows=[(8.0, 0.0, True)]), "not positive and finite"),
    ],
)
def test_a_result_for_another_configuration_is_refused(record: dict, message: str) -> None:
    """Each part of the configuration the table describes must match, or the merge is refused."""
    table = _table()
    before = copy.deepcopy(table)
    with pytest.raises(EvaluationMismatchError, match=message):
        _merge(table, record)
    assert table == before


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("converged", "false", "converged"),
        ("converged", 1, "converged"),
        ("eps_total", "0.003", "eps_total"),
        ("eps_total", True, "eps_total"),
        ("window_duration", "2048", "window_duration"),
    ],
)
def test_a_window_field_of_the_wrong_type_is_refused(field: str, value: object, message: str) -> None:
    """A field of the wrong JSON type is refused rather than coerced: the string "false" is not a convergence."""
    table = _table()
    before = copy.deepcopy(table)
    record = _record(sampling_frequency=4096.0, windows=[(2048.0, 0.003, True)])
    record["windows"][0][field] = value
    with pytest.raises(EvaluationMismatchError, match=message):
        _merge(table, record)
    assert table == before


def test_a_window_measured_over_another_band_is_refused() -> None:
    """The error band of every window must be the curve's band at that rate."""
    record = _record()
    record["windows"][0]["error_band_hz"] = [5.0, 4092.0]
    with pytest.raises(EvaluationMismatchError, match="measured over"):
        _merge(_table(), record)


def test_the_selector_cites_an_evaluated_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """A window chosen from an evaluation is reported with that evaluation as its source."""
    table = _table()
    source = _merge(table, _record(sampling_frequency=4096.0, windows=[(4.0, 0.5, True)]))
    monkeypatch.setattr(covariance_tolerance, "load_crossfade_errors", lambda: table)

    simulator = ColoredNoiseSimulator(psd_file=ET_D, sampling_frequency=4096.0, covariance_tolerance=0.6)

    recorded = simulator.metadata["colored_noise"]["covariance_tolerance"]
    assert simulator.window_duration == 4.0
    assert recorded["evaluated_error"] == 0.5
    assert recorded["source"] == describe_source(table, source)
    assert recorded["source"].startswith("gwmock-noise 0.19.0 evaluation evaluations/result.json (SHA-256 ")


def test_a_result_the_command_wrote_merges(tmp_path: Path) -> None:
    """The command's own output is accepted once it is from a release: the two agree on the configuration."""
    out = tmp_path / "result.json"
    result = CliRunner().invoke(
        app,
        [
            "evaluate-covariance-error",
            ET_D,
            "--sampling-frequency",
            "4096",
            "--analysis-band",
            "5",
            "2048",
            "--window-duration",
            "4",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    record = json.loads(out.read_text(encoding="utf-8"))
    table = _table()
    with pytest.raises(EvaluationMismatchError, match="not a plain release version"):
        _merge(table, record)
    record["gwmock_noise_version"] = "0.19.0"
    source = _merge(table, record)
    row = _rows(table, ET_D, "4096")[4.0]
    assert row["source"] == source
    assert row["eps_total"] == record["windows"][0]["eps_total"]
    assert row["converged"] is True


def _write(directory: Path, record: dict) -> Path:
    """Write a result as the command would, under a name of its own."""
    data = (json.dumps(record, indent=4, sort_keys=True) + "\n").encode("utf-8")
    path = directory / f"{hashlib.sha256(data).hexdigest()[:12]}.json"
    path.write_bytes(data)
    return path


#: Three evaluations of ET_D at 4096 Hz that share windows: two agree on 2 s, and the third
#: converges where another did not.
OVERLAPPING = [
    [(2.0, 0.1, True), (32.0, 0.001, True)],
    [(2.0, 0.1 * (1 + 1e-12), True), (4.0, 0.05, True)],
    [(4.0, 0.05, False), (32.0, 0.001, True), (2048.0, 0.003, True)],
]


def test_the_table_does_not_depend_on_the_order_of_merging(tmp_path: Path) -> None:
    """Every order of merging the same evaluations gives the same table, rows and sources alike."""
    records = [_record(sampling_frequency=4096.0, windows=windows) for windows in OVERLAPPING]
    tables = []
    for order in itertools.permutations(records):
        table = _table()
        for record in order:
            _merge(table, record)
        tables.append(table)
    assert all(table == tables[0] for table in tables)
    rows = _rows(tables[0], ET_D, "4096")
    assert rows[2048.0]["converged"] is True
    assert rows[4.0]["converged"] is True
    assert rows[2.0]["source"] == min(
        source_id(hashlib.sha256(json.dumps(r).encode()).hexdigest()) for r in records[:2]
    )


def test_adding_an_evaluation_writes_the_table_the_rebuild_gives(tmp_path: Path) -> None:
    """The table ``--merge`` writes is the one ``--check`` rebuilds, whatever the new file's name sorts as."""
    release = _table()
    directory = tmp_path / "evaluations"
    directory.mkdir()
    for windows in OVERLAPPING:
        source = tmp_path / "incoming.json"
        source.write_bytes((json.dumps(_record(sampling_frequency=4096.0, windows=windows)) + "\n").encode())
        added, table = add_evaluation(release, directory, source)
        assert added.read_bytes() == source.read_bytes()
        assert table == build_table(release, directory)
    assert len(list(directory.glob("*.json"))) == 3


def test_a_result_that_adds_nothing_is_not_added(tmp_path: Path) -> None:
    """A new result that only repeats the table is refused, and nothing is written."""
    directory = tmp_path / "evaluations"
    released = _rows(_table(), ET_D, "4096")[8.0]["eps_total"]
    result = _write(tmp_path, _record(sampling_frequency=4096.0, windows=[(8.0, released, True)]))
    with pytest.raises(EvaluationMismatchError, match="adds nothing"):
        add_evaluation(_table(), directory, result)
    assert not directory.exists()


def test_a_committed_evaluation_that_adds_nothing_does_not_stop_the_rebuild(tmp_path: Path) -> None:
    """A committed result whose windows another one covers is merged, not refused, when the table is rebuilt."""
    _write(tmp_path, _record(sampling_frequency=4096.0, windows=[(2.0, 0.1, True)]))
    _write(tmp_path, _record(sampling_frequency=4096.0, windows=[(2.0, 0.1, True), (32.0, 0.001, True)]))
    table = build_table(_table(), tmp_path)
    assert set(_rows(table, ET_D, "4096")) >= {2.0, 32.0}
    cited = {row["source"] for row in _rows(table, ET_D, "4096").values()}
    assert {source for source in table["sources"] if source != RELEASE_SOURCE} == cited - {RELEASE_SOURCE}


def test_a_committed_evaluation_that_no_longer_fits_names_its_file(tmp_path: Path) -> None:
    """When a committed result stops describing the table, such as after a change of method, the error names it."""
    record = _record(sampling_frequency=4096.0, windows=[(2.0, 0.1, True)])
    record["method"] = {**method_constants(), "maximum_harmonics": 1}
    path = _write(tmp_path, record)
    with pytest.raises(EvaluationMismatchError, match=path.name):
        build_table(_table(), tmp_path)


def test_a_plain_release_version_is_accepted() -> None:
    """The version check accepts the form this project tags releases with."""
    for version in ("0.19.0", "1.0.0", "10.20.30"):
        record = _record()
        record["gwmock_noise_version"] = version
        _merge(_table(), record)


def test_a_curve_shared_by_two_presets_is_taken_from_the_argument(monkeypatch: pytest.MonkeyPatch) -> None:
    """If two presets had the same bytes, the evaluation's argument would choose, and an unnamed one is refused."""
    table = _table()
    table["spectra"]["copy_of_ET_D"] = copy.deepcopy(table["spectra"][ET_D])
    record = _record()
    _merge(table, record)
    assert len(_rows(table, ET_D, "8192")) == 4
    record = _record()
    record["psd"]["argument"] = "/somewhere/else.txt"
    with pytest.raises(EvaluationMismatchError, match="evaluate it by name"):
        _merge(table, record)


def test_a_table_in_another_layout_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table that is not schema version 2 is refused with a reason, not read as if it were."""
    from gwmock_noise.data import covariance_tolerance as package

    monkeypatch.setattr(package.json, "loads", lambda _: {"source": {}, "spectra": {}})
    with pytest.raises(ValueError, match="schema version None"):
        load_crossfade_errors()


def test_the_build_script_imports() -> None:
    """The build script loads, so a change to the package it imports from cannot break it unseen."""
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "scripts" / "build_covariance_tolerance_table.py"
    if not script.exists():
        pytest.skip("the build script is not part of an installed package")
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode == 0, result.stderr
    assert "--merge" in result.stdout
