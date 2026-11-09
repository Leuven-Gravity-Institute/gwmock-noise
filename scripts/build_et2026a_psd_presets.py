"""Rebuild the bundled ET2026a PSD presets from the published Einstein Telescope curves.

The ET2026a sensitivity curves are published in the public repository
``pub/et-sensitivity-curves`` on the Einstein Telescope GitLab. Each arm length has a low-frequency
(LF) and a high-frequency (HF) interferometer, each with its own ``data/total.txt``. This script
fetches the four files at a pinned commit, checks each against its SHA-256, and writes one preset
per arm length:

- column 3 of each file, the curve's default sensitivity (with the high-frequency correction for a
  signal 15 degrees off zenith), is an amplitude spectral density and is squared;
- the LF and HF power spectral densities, on the same grid, are combined as the optimal combination
  of two co-located detectors with independent noise, ``S = 1 / (1 / S_LF + 1 / S_HF)``;
- the result is written in the bundled presets' format, two columns, ``%.18e``.

``src/gwmock_noise/data/psd/PROVENANCE.md`` records the source, the procedure and the checks, and
``tests/test_psd_presets.py`` checks the bundled files against this script's output.

Usage::

    uv run python scripts/build_et2026a_psd_presets.py            # check the bundled files
    uv run python scripts/build_et2026a_psd_presets.py --write    # rewrite them
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the presets are bundled.
PSD_DIRECTORY = REPO_ROOT / "src" / "gwmock_noise" / "data" / "psd"

#: The published repository and the commit the presets are built from (2026-03-30).
SOURCE_REPOSITORY = "https://gitlab.et-gw.eu/pub/et-sensitivity-curves"
SOURCE_COMMIT = "66177ccf4626c77fee243ef21367ac934b2c0ccc"

#: SHA-256 of each source file at that commit.
SOURCE_FILES = {
    "10_km/LF/data/total.txt": "2e4477e0d2dc30228a7cbc7ede804355b5b307486859a82ed9c6d4cd9bd16375",
    "10_km/HF/data/total.txt": "35bedf22facefd325148ef787caebb582348b91c1e37fd2b586de8510e7be355",
    "15_km/LF/data/total.txt": "ba58ade04eb74e9c2f148a16abf6d57da377c0b817d973d0f784a2b7ffa39854",
    "15_km/HF/data/total.txt": "b6f003e0224aaee67b62bb2aa788c87b3a93b100bf94a38f5019bc76932ba838",
}

#: Preset name per arm length.
PRESETS = {"10_km": "ET2026a_10km_psd", "15_km": "ET2026a_15km_psd"}

#: The column of each source file the preset is built from (zero-based): the default sensitivity.
SENSITIVITY_COLUMN = 2


def fetch(path: str, cache_dir: Path | None) -> bytes:
    """Return one source file's bytes, checked against its SHA-256.

    Args:
        path: The file's path in the source repository.
        cache_dir: Where to keep downloaded files, or ``None`` to download every time.

    Returns:
        The file's bytes.

    Raises:
        ValueError: If the bytes do not have the recorded SHA-256.
    """
    cached = None if cache_dir is None else cache_dir / SOURCE_COMMIT / path
    if cached is not None and cached.exists():
        payload = cached.read_bytes()
    else:
        url = f"{SOURCE_REPOSITORY}/-/raw/{SOURCE_COMMIT}/{path}"
        print(f"  downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url, timeout=600) as response:  # noqa: S310 - a fixed https URL.
            payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SOURCE_FILES[path]:
        raise ValueError(f"{path} has SHA-256 {digest}, not the recorded {SOURCE_FILES[path]}.")
    if cached is not None and not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(payload)
    return payload


def combine(low: np.ndarray, high: np.ndarray) -> np.ndarray:
    """Combine an LF and an HF source table into one two-column PSD table.

    Args:
        low: The LF file's columns.
        high: The HF file's columns.

    Returns:
        Frequency and the combined one-sided PSD.

    Raises:
        ValueError: If the two files are not on the same frequency grid.
    """
    if not np.array_equal(low[:, 0], high[:, 0]):
        raise ValueError("the LF and HF curves are not on the same frequency grid.")
    psd_low = low[:, SENSITIVITY_COLUMN] ** 2
    psd_high = high[:, SENSITIVITY_COLUMN] ** 2
    return np.column_stack((low[:, 0], 1.0 / (1.0 / psd_low + 1.0 / psd_high)))


def render(table: np.ndarray) -> str:
    """The table in the bundled presets' format."""
    buffer = io.StringIO()
    np.savetxt(buffer, table, fmt="%.18e")
    return buffer.getvalue()


def build(cache_dir: Path | None = None) -> dict[str, str]:
    """Build every preset.

    Args:
        cache_dir: Where to keep downloaded source files.

    Returns:
        Each preset's file content, keyed by preset name.
    """
    built = {}
    for length, name in PRESETS.items():
        low, high = (
            np.loadtxt(io.BytesIO(fetch(f"{length}/{band}/data/total.txt", cache_dir))) for band in ("LF", "HF")
        )
        built[name] = render(combine(low, high))
    return built


def main(argv: list[str] | None = None) -> int:
    """Build the presets and check them against, or write them over, the bundled files."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "gwmock-noise" / "et-sensitivity-curves",
        help="Where the downloaded source files are kept.",
    )
    parser.add_argument("--write", action="store_true", help="Rewrite the bundled presets.")
    arguments = parser.parse_args(argv)
    differing = []
    for name, content in build(arguments.cache_dir).items():
        path = PSD_DIRECTORY / f"{name}.txt"
        if arguments.write:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(REPO_ROOT)}")
        elif not path.exists() or path.read_text(encoding="utf-8") != content:
            differing.append(name)
    if differing:
        print(f"bundled presets differ from the build: {differing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
