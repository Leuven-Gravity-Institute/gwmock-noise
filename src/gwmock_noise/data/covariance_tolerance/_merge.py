"""Merge evaluated covariance errors into the bundled table, with their provenance.

The bundled table starts as a copy of the data release and can be extended by results of
``gwmock-noise evaluate-covariance-error`` for the curves it already covers: windows the
release did not evaluate, sampling frequencies it did not evaluate, and windows whose release
evaluation did not converge. Every row names its source in ``table["sources"]``, and an
evaluation is committed verbatim under ``evaluations/``. The table is a function of the
release and those files: :func:`build_table` merges them, in any order, into the same table,
and ``scripts/build_covariance_tolerance_table.py --check`` rebuilds it with this function.

An evaluation is merged only if it describes the configuration the table does: the bundled
curve, byte for byte; the table's continued target; the method constants of this package;
the curve's analysis band, and the error band derived from it; and a release of gwmock-noise.
Where several sources give a window, the table keeps one by a fixed preference, so the order
of merging does not matter: a converged error before an unconverged one, then the release
before an evaluation, then the evaluation whose id sorts first. Converged errors of one window
must agree, which checks each source against the others. A new evaluation is added only if it
contributes: a window the table lacks, or a converged error where the table's is unconverged.
``table["sources"]`` lists every committed evaluation, cited by a row or overtaken by another,
with its SHA-256, so ``--check`` notices a change to any committed file.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path
from typing import Any

from gwmock_noise.data.covariance_tolerance import describe_source
from gwmock_noise.diagnostics import crossfade_covariance
from gwmock_noise.simulators._covariance_tolerance import error_band

#: The id of the data release in ``table["sources"]``.
RELEASE_SOURCE = "zenodo-23243286"

#: Relative tolerance within which two converged errors of one window must agree. Two
#: evaluations by the same method differ only by rounding, which depends on the numpy and
#: scipy versions (up to about 1e-12 relative).
AGREEMENT_RTOL = 1e-10

#: A plain release version, as this project tags them (``0.19.0``). A development or local
#: build carries more (``0.18.4-post.3+8f8b4bb``). The pattern checks the form only: it cannot
#: tell a release from a build that reports a release's version.
_RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")


class EvaluationMismatchError(ValueError):
    """An evaluation does not describe the configuration the bundled table does, or cannot join it."""


def method_constants() -> dict[str, Any]:
    """Return this package's constants of the evaluation method, as a result records them."""
    return {
        "nyquist_guard_hz": crossfade_covariance.NYQUIST_GUARD_HZ,
        "oversample": crossfade_covariance.OVERSAMPLE,
        "convergence_tolerance": crossfade_covariance.CONVERGENCE_TOLERANCE,
        "first_convergence_check": crossfade_covariance.FIRST_CONVERGENCE_CHECK,
        "maximum_harmonics": crossfade_covariance.MAXIMUM_HARMONICS,
    }


def source_id(sha256: str) -> str:
    """Return the id an evaluation file has in ``table["sources"]``."""
    return f"evaluation-{sha256[:12]}"


def evaluated_preset(table: dict[str, Any], record: dict[str, Any]) -> str:
    """Return the bundled curve an evaluation was made for, identified by its SHA-256.

    Raises:
        EvaluationMismatchError: If no bundled curve has the evaluated PSD's bytes, or if
            several do and the evaluation's argument does not name one of them.
    """
    digest = record["psd"]["sha256"]
    presets = sorted(preset for preset, spectrum in table["spectra"].items() if spectrum["sha256"] == digest)
    if not presets:
        raise EvaluationMismatchError(
            f"the evaluated PSD (SHA-256 {digest}) is none of the bundled curves the table covers, "
            f"{sorted(table['spectra'])}; a new curve is not added by a merge."
        )
    if len(presets) == 1:
        return presets[0]
    named = Path(str(record["psd"]["argument"])).stem
    if named in presets:
        return named
    raise EvaluationMismatchError(f"the evaluated PSD is the curve of each of {presets}; evaluate it by name.")


def _check_window(window: dict[str, Any], expected_band: list[float]) -> None:
    """Check one window of an evaluation: its field types, its error band and its error."""
    # Checked by JSON type, not coerced: bool("false") is True, and would pass an unconverged evaluation as
    # converged.
    if type(window["converged"]) is not bool:
        raise EvaluationMismatchError(f"a window's converged is {window['converged']!r}, not true or false.")
    for field in ("window_duration", "eps_total"):
        if isinstance(window[field], bool) or not isinstance(window[field], int | float):
            raise EvaluationMismatchError(f"a window's {field} is {window[field]!r}, not a number.")
    if not (math.isfinite(window["window_duration"]) and window["window_duration"] > 0):
        raise EvaluationMismatchError(
            f"a window's window_duration {window['window_duration']!r} s is not positive and finite."
        )
    if [float(edge) for edge in window["error_band_hz"]] != expected_band:
        raise EvaluationMismatchError(
            f"the {window['window_duration']!r} s window was measured over {window['error_band_hz']} Hz, "
            f"not {expected_band} Hz."
        )
    if not (math.isfinite(window["eps_total"]) and window["eps_total"] > 0):
        raise EvaluationMismatchError(
            f"the {window['window_duration']!r} s window's error {window['eps_total']!r} is not positive and finite."
        )


def _check_configuration(table: dict[str, Any], record: dict[str, Any], preset: str) -> str:
    """Check that an evaluation describes the table's configuration, and return its rate's key."""
    version = str(record["gwmock_noise_version"])
    if not _RELEASE_VERSION.fullmatch(version):
        raise EvaluationMismatchError(
            f"the evaluation was made by gwmock-noise {version}, not a plain release version such as 0.19.0; "
            f"this looks like a development or local build. Evaluate with a release."
        )
    if record["target"] != table["target"]:
        raise EvaluationMismatchError(
            f"the evaluation's target {record['target']} is not the table's {table['target']}."
        )
    if record["method"] != method_constants():
        raise EvaluationMismatchError(
            f"the evaluation's method constants {record['method']} are not this package's {method_constants()}; "
            f"an evaluation made with other constants describes another method. Evaluate again with this version."
        )
    band = table["spectra"][preset]["analysis_band_hz"]
    if [float(edge) for edge in record["analysis_band_hz"]] != band:
        raise EvaluationMismatchError(
            f"the evaluation's analysis band {record['analysis_band_hz']} Hz is not {preset!r}'s, {band} Hz."
        )
    sampling_frequency = float(record["sampling_frequency"])
    if not sampling_frequency.is_integer() or sampling_frequency <= 0:
        raise EvaluationMismatchError(f"the sampling frequency {sampling_frequency!r} Hz is not a positive integer.")
    expected_band = list(error_band((band[0], band[1]), sampling_frequency))
    rate = table["spectra"][preset]["sampling_frequencies"].get(str(round(sampling_frequency)))
    if rate is not None and rate["error_band_hz"] != expected_band:
        raise EvaluationMismatchError(
            f"the table's error band at {sampling_frequency!r} Hz is {rate['error_band_hz']}, not {expected_band}."
        )
    if not record["windows"]:
        raise EvaluationMismatchError("the evaluation holds no window.")
    for window in record["windows"]:
        _check_window(window, expected_band)
    return str(round(sampling_frequency))


def _preference(row: dict[str, Any]) -> tuple[bool, bool, str]:
    """Return the key by which the table keeps one row of a window: the smallest is kept."""
    return (not row["converged"], row["source"] != RELEASE_SOURCE, row["source"])


def merge_evaluation(table: dict[str, Any], record: dict[str, Any], *, file: str, sha256: str) -> None:
    """Merge one ``evaluate-covariance-error`` result into ``table``, in place.

    For each window the table keeps the preferred of its row and the evaluation's (see the
    module docstring), so merging several evaluations gives the same table in any order.

    Args:
        table: The table, as ``scripts/build_covariance_tolerance_table.py`` builds it.
        record: The result the command wrote, as parsed from its JSON.
        file: The result's path under the table's directory, recorded as its source.
        sha256: The SHA-256 of the result file's bytes.

    Raises:
        EvaluationMismatchError: If the result does not describe the table's configuration, or if a
            converged window it shares with the table does not agree with the table's value.
    """
    preset = evaluated_preset(table, record)
    key = _check_configuration(table, record, preset)
    band = table["spectra"][preset]["analysis_band_hz"]
    rates = table["spectra"][preset]["sampling_frequencies"]
    rate = rates.get(key, {"error_band_hz": list(error_band((band[0], band[1]), float(key))), "windows": []})
    source = source_id(sha256)
    rows = {row["window_duration"]: row for row in rate["windows"]}
    for window in record["windows"]:
        duration = float(window["window_duration"])
        new = {
            "window_duration": duration,
            "eps_total": float(window["eps_total"]),
            "converged": window["converged"],
            "source": source,
        }
        old = rows.get(duration)
        if (
            old is not None
            and old["converged"]
            and new["converged"]
            and not math.isclose(new["eps_total"], old["eps_total"], rel_tol=AGREEMENT_RTOL, abs_tol=0.0)
        ):
            raise EvaluationMismatchError(
                f"{preset!r} at {key} Hz and {duration!r} s: the evaluation gives {new['eps_total']!r}, but the "
                f"table holds {old['eps_total']!r} from {describe_source(table, old['source'])}; converged "
                f"errors of one window must agree to {AGREEMENT_RTOL}."
            )
        if old is None or _preference(new) < _preference(old):
            rows[duration] = new
    # Nothing is written until every window is accepted, so a refused merge leaves the table as it was.
    rates[key] = {**rate, "windows": [rows[duration] for duration in sorted(rows)]}
    table["sources"][source] = {
        "kind": "evaluation",
        "file": file,
        "sha256": sha256,
        "gwmock_noise_version": str(record["gwmock_noise_version"]),
    }


def contributes(table: dict[str, Any], record: dict[str, Any]) -> bool:
    """Return whether an evaluation adds to the table: a window it lacks, or a converged error for an unconverged one.

    Raises:
        EvaluationMismatchError: If the result does not describe the table's configuration.
    """
    preset = evaluated_preset(table, record)
    key = _check_configuration(table, record, preset)
    rate = table["spectra"][preset]["sampling_frequencies"].get(key, {"windows": []})
    rows = {row["window_duration"]: row for row in rate["windows"]}
    return any(
        float(window["window_duration"]) not in rows
        or (window["converged"] and not rows[float(window["window_duration"])]["converged"])
        for window in record["windows"]
    )


def build_table(release_table: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Return the release's table with every evaluation committed in ``directory`` merged.

    Besides each merge's check against the row the table keeps, every converged error of a
    window, from the release and from each evaluation, must agree with every other: closeness
    is not transitive, so checking each against the kept row alone would accept or refuse a
    set depending on the order of merging.

    Raises:
        EvaluationMismatchError: If a committed evaluation cannot be merged, or if the converged
            errors of a window do not all agree; the message names the files.
    """
    table = copy.deepcopy(release_table)
    converged: dict[tuple[str, str, float], list[tuple[float, str]]] = {}
    for preset, spectrum in release_table["spectra"].items():
        for key, rate in spectrum["sampling_frequencies"].items():
            for row in rate["windows"]:
                if row["converged"]:
                    converged.setdefault((preset, key, row["window_duration"]), []).append(
                        (row["eps_total"], describe_source(release_table, row["source"]))
                    )
    for path in sorted(directory.glob("*.json")):
        data = path.read_bytes()
        record = json.loads(data)
        try:
            merge_evaluation(
                table, record, file=f"{directory.name}/{path.name}", sha256=hashlib.sha256(data).hexdigest()
            )
        except EvaluationMismatchError as exc:
            raise EvaluationMismatchError(f"{directory.name}/{path.name}: {exc} Remove or replace that file.") from exc
        preset = evaluated_preset(table, record)
        key = str(round(float(record["sampling_frequency"])))
        for window in record["windows"]:
            if window["converged"]:
                converged.setdefault((preset, key, float(window["window_duration"])), []).append(
                    (float(window["eps_total"]), f"{directory.name}/{path.name}")
                )
    for (preset, key, duration), values in sorted(converged.items()):
        (low, low_source), (high, high_source) = min(values), max(values)
        if not math.isclose(low, high, rel_tol=AGREEMENT_RTOL, abs_tol=0.0):
            raise EvaluationMismatchError(
                f"{preset!r} at {key} Hz and {duration!r} s: converged errors of one window must agree to "
                f"{AGREEMENT_RTOL}, but range from {low!r} ({low_source}) to {high!r} ({high_source}). Remove or "
                f"replace the file that disagrees."
            )
    return table


def add_evaluation(release_table: dict[str, Any], directory: Path, result: Path) -> tuple[Path, dict[str, Any]]:
    """Copy a new result into ``directory`` and return its path and the table rebuilt with it.

    The result is checked against the table the committed evaluations give, must contribute
    to it, and must rebuild with them; only then is it copied byte for byte. The table returned
    is :func:`build_table`'s, the one ``--check`` rebuilds.

    Raises:
        EvaluationMismatchError: If the result does not describe the table's configuration, adds
            nothing to it, or disagrees with a converged error in it.
    """
    data = result.read_bytes()
    record = json.loads(data)
    sha256 = hashlib.sha256(data).hexdigest()
    current = build_table(release_table, directory)
    preset = evaluated_preset(current, record)
    if not contributes(current, record):
        raise EvaluationMismatchError(
            f"the evaluation adds nothing to {preset!r} at {round(float(record['sampling_frequency']))} Hz: every "
            f"window it holds is already in the table, converged or no less converged."
        )
    name = f"{preset}__{round(float(record['sampling_frequency']))}__{sha256[:12]}.json"
    # The table is rebuilt with the new file in a staging copy first, so a result the rebuild refuses is never
    # written where it would make every later --check fail.
    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / directory.name
        staged.mkdir()
        for path in directory.glob("*.json"):
            (staged / path.name).write_bytes(path.read_bytes())
        (staged / name).write_bytes(data)
        table = build_table(release_table, staged)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(data)
    return directory / name, table
