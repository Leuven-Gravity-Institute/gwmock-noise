"""Build the crossfade covariance-error table from the published data release and merged evaluations.

``ColoredNoiseSimulator(covariance_tolerance=...)`` chooses its crossfade window from a
small table bundled at ``src/gwmock_noise/data/covariance_tolerance/crossfade_errors.json``.
Its numbers are copied from the data release of

    I. C. F. Wong, "Choosing a generator for streaming Gaussian noise in
    gravitational-wave mock data: accuracy, cost and reproducibility",
    arXiv:2611.01624; data: doi:10.5281/zenodo.23243286

or taken from ``gwmock-noise evaluate-covariance-error`` results committed under
``evaluations/`` next to it, and every row names which. This script is how the table is made
from those sources, so it can be checked against them rather than trusted; ``--merge`` adds a
result after the checks of :mod:`gwmock_noise.data.covariance_tolerance._merge`.

From the release archive ``results.tar.gz`` it reads the evaluated covariance errors
``results/exact/<curve>__specified__<rate>.json`` of the four detector curves that
gwmock-noise bundles, and keeps the rows of this package's crossfade (``crossfade_gwmock``)
at the half-window overlap the simulator uses. Each row keeps its window, its total
covariance error, and whether the evaluation of its cyclic part converged; an unconverged
value omits part of the cyclic error and is therefore not an upper bound.

The errors were evaluated against the paper's continued target (held flat below 5 Hz at its
value there, which replaces the ET curves' tabulated 1-5 Hz; between 5 Hz and a first
tabulated point above it, as on the aLIGO curves, the power law of the lowest octave) over a
fixed band per curve. The table records those settings next to the numbers,
together with the SHA-256 of each bundled curve, so that a changed curve is detected. A
bundled curve is recorded only if it is byte for byte the curve the release evaluated, which
the archive also carries.

Usage::

    uv run python scripts/build_covariance_tolerance_table.py --archive results.tar.gz --check
    uv run python scripts/build_covariance_tolerance_table.py --archive results.tar.gz --write
    uv run python scripts/build_covariance_tolerance_table.py --archive results.tar.gz --merge result.json

``--check`` compares the table built from the release and the committed evaluations with
the bundled one, value by value, and exits non-zero on any difference. ``--merge`` checks a
result, copies it byte for byte into ``evaluations/``, and rewrites the table. ``--write`` rewrites the bundled table; the pre-commit
prettier hook then reformats it, which changes its layout but not its values. Without
``--archive`` the archive is downloaded from the Zenodo record. Its MD5 is checked against
the published value either way.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

from gwmock_noise.data.covariance_tolerance import SCHEMA_VERSION
from gwmock_noise.data.covariance_tolerance._merge import (
    RELEASE_SOURCE,
    EvaluationMismatchError,
    add_evaluation,
    build_table,
)
from gwmock_noise.simulators._covariance_tolerance import error_band

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the bundled table lives.
TABLE_PATH = REPO_ROOT / "src" / "gwmock_noise" / "data" / "covariance_tolerance" / "crossfade_errors.json"

#: Evaluations merged into the table, committed verbatim.
EVALUATIONS_DIR = TABLE_PATH.parent / "evaluations"

#: The bundled PSD presets.
PSD_DIR = REPO_ROOT / "src" / "gwmock_noise" / "data" / "psd"

#: The published data release.
DOI = "10.5281/zenodo.23243286"
ARCHIVE_URL = "https://zenodo.org/records/23243286/files/results.tar.gz?download=1"
ARCHIVE_MD5 = "d24fab45af50addeb46ea8c03cb5a1ba"

#: The release's curve names for the four bundled presets, and each curve's analysis band.
#: The bands are those of the paper: 10 Hz for the aLIGO curves and 5 Hz for the ET curves,
#: to 2048 Hz.
CURVES: dict[str, tuple[str, tuple[float, float]]] = {
    "aLIGO_O4_high_projected_psd": ("aligo_o4high_smooth", (10.0, 2048.0)),
    "aLIGO_O3_actual_H1_psd": ("aligo_o3h1_lines", (10.0, 2048.0)),
    "ET_D_psd": ("et_d_smooth", (5.0, 2048.0)),
    "ET_10_full_cryo_psd": ("et10_full_cryo_lines", (5.0, 2048.0)),
}

#: The sampling frequencies the release evaluates.
SAMPLING_FREQUENCIES = (4096, 8192)

#: The continued target the errors were evaluated against.
TARGET = {"continuation_hz": 5.0, "slope_octave": 2.0, "continuation_points": 256}

#: The release's name for this package's crossfade.
CROSSFADE_ROW = "crossfade_gwmock"

#: A grid identifier at the half-window overlap, such as ``64s_50`` or ``64s_50_8192``.
HALF_OVERLAP_GRID = re.compile(r"^(?P<window>\d+)s_50(?:_\d+)?$")


def md5_of(path: Path) -> str:
    """Return the MD5 of a file."""
    digest = hashlib.md5()  # noqa: S324 - the checksum Zenodo publishes.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_archive(target: Path) -> None:
    """Download the release archive."""
    with urllib.request.urlopen(ARCHIVE_URL, timeout=600) as response, target.open("wb") as handle:
        handle.write(response.read())


def read_exact(archive: Path, curve: str, sampling_frequency: int) -> dict[str, Any]:
    """Return one evaluated-error file of the release.

    Raises:
        KeyError: If the archive has no such file.
    """
    name = f"results/exact/{curve}__specified__{sampling_frequency}.json"
    with tarfile.open(archive) as bundle:
        member = bundle.extractfile(name)
        if member is None:
            raise KeyError(name)
        return json.load(member)


def read_curve(archive: Path, curve: str) -> bytes:
    """Return the bytes of a curve the release evaluated.

    Raises:
        KeyError: If the archive has no such file.
    """
    name = f"psd/{curve}.txt"
    with tarfile.open(archive) as bundle:
        member = bundle.extractfile(name)
        if member is None:
            raise KeyError(name)
        return member.read()


def crossfade_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the crossfade rows of one file at the half-window overlap, by window.

    Raises:
        ValueError: If a row failed or a window appears twice.
    """
    rows = []
    for row in record["results"]["exact_errors"]:
        match = HALF_OVERLAP_GRID.match(row["parameters"].get("grid", ""))
        if row["name"] != CROSSFADE_ROW or match is None:
            continue
        if row.get("failure") is not None:
            raise ValueError(f"{row['parameters']['grid']}: the release records a failure, {row['failure']!r}.")
        rows.append(
            {
                "window_duration": float(match["window"]),
                "eps_total": float(row["eps_total"]),
                "converged": bool(row["diagnostic"]["converged"]),
            }
        )
    rows.sort(key=lambda entry: entry["window_duration"])
    windows = [entry["window_duration"] for entry in rows]
    if len(windows) != len(set(windows)):
        raise ValueError(f"a window appears twice: {windows}.")
    return rows


def build(archive: Path) -> dict[str, Any]:
    """Return the table built from the release archive.

    Raises:
        ValueError: If the archive is not the published one, if its files disagree on the
            generator they evaluated, or if a bundled curve is not the one the release evaluated.
    """
    actual = md5_of(archive)
    if actual != ARCHIVE_MD5:
        raise ValueError(f"{archive} has MD5 {actual}, not the published {ARCHIVE_MD5}.")
    spectra: dict[str, Any] = {}
    generator_commits: set[str] = set()
    analysis_commits: set[str] = set()
    for preset, (curve, band) in CURVES.items():
        rates = {}
        for sampling_frequency in SAMPLING_FREQUENCIES:
            record = read_exact(archive, curve, sampling_frequency)
            generator_commits.add(record["generator_commit"])
            analysis_commits.add(record["analysis_commit"])
            rates[str(sampling_frequency)] = {
                "error_band_hz": list(error_band(band, sampling_frequency)),
                "windows": [{**row, "source": RELEASE_SOURCE} for row in crossfade_rows(record)],
            }
        # The table pins the bundled file, so the bundled file must be the curve the release evaluated.
        bundled = (PSD_DIR / f"{preset}.txt").read_bytes()
        if bundled != read_curve(archive, curve):
            raise ValueError(f"the bundled {preset}.txt differs from psd/{curve}.txt, the curve the release evaluated.")
        spectra[preset] = {
            "analysis_band_hz": list(band),
            "release_curve": curve,
            "sha256": hashlib.sha256(bundled).hexdigest(),
            "sampling_frequencies": rates,
        }
    if len(generator_commits) != 1:
        raise ValueError(f"the files evaluated different generators: {sorted(generator_commits)}.")
    return {
        "schema_version": SCHEMA_VERSION,
        "sources": {
            RELEASE_SOURCE: {
                "kind": "release",
                "doi": DOI,
                "archive": "results.tar.gz",
                "archive_md5": ARCHIVE_MD5,
                "generator_commit": generator_commits.pop(),
                "analysis_commits": sorted(analysis_commits),
            }
        },
        "target": TARGET,
        "spectra": spectra,
    }


def write_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step, so an interruption leaves the old table or the new one."""
    partial = path.with_name(f"{path.name}.partial")
    try:
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def write_table(text: str, added: Path | None) -> int:
    """Write the table; if that fails, remove the result ``--merge`` just added, and report the failure.

    A result copied into the evaluations without the table that cites it would make ``--check`` fail.
    """
    try:
        write_atomically(TABLE_PATH, text)
    except BaseException as exc:
        if added is not None:
            added.unlink(missing_ok=True)
        if not isinstance(exc, OSError):
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {TABLE_PATH.relative_to(REPO_ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Build the table, print it, and write it when asked."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", type=Path, help="The release's results.tar.gz; downloaded when omitted.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help=f"Rewrite {TABLE_PATH.relative_to(REPO_ROOT)}.")
    action.add_argument("--check", action="store_true", help="Compare with the bundled table; exit 1 if it differs.")
    action.add_argument(
        "--merge",
        type=Path,
        metavar="RESULT",
        help="Add an evaluate-covariance-error result to the evaluations and rewrite the table.",
    )
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as directory:
        archive = args.archive
        if archive is None:
            archive = Path(directory) / "results.tar.gz"
            fetch_archive(archive)
        release = build(archive)
    added = None
    try:
        if args.merge is not None:
            added, table = add_evaluation(release, EVALUATIONS_DIR, args.merge)
            print(f"added {added.relative_to(REPO_ROOT)}")
            args.write = True
        else:
            table = build_table(release, EVALUATIONS_DIR)
    except EvaluationMismatchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (ValueError, KeyError, TypeError) as exc:
        # A file that is not what evaluate-covariance-error writes: not JSON, or JSON of another shape.
        # Committed evaluations are reported by build_table with their names, so this is the --merge input.
        print(
            f"error: {args.merge or 'the table'}: not an evaluate-covariance-error result "
            f"({type(exc).__name__}: {exc}).",
            file=sys.stderr,
        )
        return 1
    text = json.dumps(table, indent=4, sort_keys=True) + "\n"
    if args.check:
        bundled = json.loads(TABLE_PATH.read_text(encoding="utf-8"))
        if bundled != table:
            print(
                f"{TABLE_PATH.relative_to(REPO_ROOT)} differs from the table built from the release and the evaluations."
            )
            return 1
        print(f"{TABLE_PATH.relative_to(REPO_ROOT)} matches the release and the committed evaluations.")
    elif args.write:
        return write_table(text, added)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
