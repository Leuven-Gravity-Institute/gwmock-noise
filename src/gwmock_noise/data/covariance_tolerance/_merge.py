"""Merge an evaluated covariance error into the bundled table, with its provenance.

The bundled table starts as a copy of the data release and can be extended by results of
``gwmock-noise evaluate-covariance-error`` for the curves it already covers: windows the
release did not evaluate, sampling frequencies it did not evaluate, and windows whose release
evaluation did not converge. Every row names its source in ``table["sources"]``, and an
evaluation is committed verbatim under ``evaluations/``, so the table is a function of the
release archive and those files, and ``scripts/build_covariance_tolerance_table.py --check``
rebuilds it from them.

An evaluation is merged only if it describes the configuration the table does: the bundled
curve, byte for byte; the table's continued target; the method constants of this package;
the curve's analysis band, and the error band derived from it; and a released version of
gwmock-noise. A converged row is never replaced: a converged evaluation of the same window
must agree with it, which checks one against the other. An unconverged row is replaced only
by a converged evaluation of the same window.
"""

from __future__ import annotations

import math
import re
from typing import Any

from gwmock_noise.data.covariance_tolerance import describe_source
from gwmock_noise.diagnostics import crossfade_covariance
from gwmock_noise.simulators._covariance_tolerance import error_band

#: The version of the table's layout.
SCHEMA_VERSION = 2

#: The id of the data release in ``table["sources"]``.
RELEASE_SOURCE = "zenodo-23243286"

#: Relative tolerance within which two converged evaluations of one window must agree. Two
#: evaluations by the same method differ only by rounding, which depends on the numpy and
#: scipy versions (up to about 1e-12 relative).
AGREEMENT_RTOL = 1e-10

#: A final release, such as ``0.19.0``; development builds carry a ``post``, ``dev`` or local part.
_RELEASED_VERSION = re.compile(r"\d+(\.\d+)*")


class EvaluationMismatchError(ValueError):
    """An evaluation does not describe the configuration the bundled table does."""


def method_constants() -> dict[str, Any]:
    """Return this package's constants of the evaluation method, as a result records them."""
    return {
        "nyquist_guard_hz": crossfade_covariance.NYQUIST_GUARD_HZ,
        "oversample": crossfade_covariance.OVERSAMPLE,
        "convergence_tolerance": crossfade_covariance.CONVERGENCE_TOLERANCE,
        "first_convergence_check": crossfade_covariance.FIRST_CONVERGENCE_CHECK,
        "maximum_harmonics": crossfade_covariance.MAXIMUM_HARMONICS,
    }


def evaluated_preset(table: dict[str, Any], record: dict[str, Any]) -> str:
    """Return the bundled curve an evaluation was made for, identified by its SHA-256."""
    digest = record["psd"]["sha256"]
    presets = [preset for preset, spectrum in table["spectra"].items() if spectrum["sha256"] == digest]
    if not presets:
        raise EvaluationMismatchError(
            f"the evaluated PSD (SHA-256 {digest}) is none of the bundled curves the table covers, "
            f"{sorted(table['spectra'])}; a new curve is not added by a merge."
        )
    return presets[0]


def _check_configuration(table: dict[str, Any], record: dict[str, Any], preset: str) -> str:
    """Check that an evaluation describes the table's configuration, and return its rate's key."""
    version = str(record["gwmock_noise_version"])
    if not _RELEASED_VERSION.fullmatch(version):
        raise EvaluationMismatchError(
            f"the evaluation was made by gwmock-noise {version}, not a released version; evaluate with a release."
        )
    if record["target"] != table["target"]:
        raise EvaluationMismatchError(
            f"the evaluation's target {record['target']} is not the table's {table['target']}."
        )
    if record["method"] != method_constants():
        raise EvaluationMismatchError(
            f"the evaluation's method constants {record['method']} are not this package's {method_constants()}."
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
        if [float(edge) for edge in window["error_band_hz"]] != expected_band:
            raise EvaluationMismatchError(
                f"the {window['window_duration']!r} s window was measured over {window['error_band_hz']} Hz, "
                f"not {expected_band} Hz."
            )
        if not (math.isfinite(window["eps_total"]) and window["eps_total"] > 0):
            raise EvaluationMismatchError(
                f"the {window['window_duration']!r} s window's error {window['eps_total']!r} is not positive and finite."
            )
    return str(round(sampling_frequency))


def merge_evaluation(table: dict[str, Any], record: dict[str, Any], *, file: str, sha256: str) -> None:
    """Merge one ``evaluate-covariance-error`` result into ``table``, in place.

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
    source = f"evaluation-{sha256[:12]}"
    rows = {row["window_duration"]: row for row in rate["windows"]}
    used = False
    for window in record["windows"]:
        duration = float(window["window_duration"])
        new = {
            "window_duration": duration,
            "eps_total": float(window["eps_total"]),
            "converged": bool(window["converged"]),
            "source": source,
        }
        old = rows.get(duration)
        if old is None or (new["converged"] and not old["converged"]):
            rows[duration] = new
            used = True
        elif (
            old["converged"]
            and new["converged"]
            and not math.isclose(new["eps_total"], old["eps_total"], rel_tol=AGREEMENT_RTOL, abs_tol=0.0)
        ):
            raise EvaluationMismatchError(
                f"{preset!r} at {key} Hz and {duration!r} s: the evaluation gives {new['eps_total']!r}, but the "
                f"table holds {old['eps_total']!r} from {describe_source(table, old['source'])}; converged "
                f"evaluations of one window must agree to {AGREEMENT_RTOL}."
            )
    if not used:
        raise EvaluationMismatchError(
            f"the evaluation adds nothing to {preset!r} at {key} Hz: every window it holds is already in the table, "
            f"converged or no less converged."
        )
    # Nothing is written until every window is accepted, so a refused merge leaves the table as it was.
    rates[key] = {**rate, "windows": [rows[duration] for duration in sorted(rows)]}
    table["sources"][source] = {
        "kind": "evaluation",
        "file": file,
        "sha256": sha256,
        "gwmock_noise_version": str(record["gwmock_noise_version"]),
    }
