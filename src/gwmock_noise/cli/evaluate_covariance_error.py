"""CLI entry point for evaluating the covariance error of the colored-noise crossfade."""

from __future__ import annotations

# Imported on first use from Python 3.15 (PEP 810), and as usual before it: only the command body
# needs them, so ``--help`` and building the command-line app do not pay for them.
__lazy_modules__ = [
    "numpy",
    "gwmock_noise.diagnostics",
    "gwmock_noise.diagnostics.crossfade_covariance",
    "gwmock_noise.simulators._spectral",
]

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from gwmock_noise.diagnostics import crossfade_covariance
from gwmock_noise.simulators._spectral import _resolve_bundled_psd
from gwmock_noise.version import __version__


def _psd_path(psd: str) -> Path:
    """Return the file a PSD argument names: a path, or the name of a bundled curve."""
    given = Path(psd)
    if given.is_file():
        return given
    # Only a bare name selects a bundled curve; one with a directory part is a path, so a mistyped path is reported
    # instead of falling back to the bundled curve its file name matches.
    if given.name != psd:
        raise typer.BadParameter(f"{psd!r} is not a file.", param_hint="PSD")
    bundled = _resolve_bundled_psd(psd)
    if bundled is None:
        raise typer.BadParameter(f"{psd!r} is neither a file nor a bundled PSD.", param_hint="PSD")
    return Path(bundled)


def _write_atomically(path: Path, text: str) -> None:
    """Replace ``path`` with ``text`` in one step, so an interruption leaves the old file or the new one."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def evaluate_covariance_error(
    psd: Annotated[str, typer.Argument(help="A two-column PSD table, or the name of a bundled curve.")],
    sampling_frequency: Annotated[float, typer.Option("--sampling-frequency", help="Sampling frequency in Hz.")],
    analysis_band: Annotated[
        tuple[float, float], typer.Option("--analysis-band", help="Low and high edge of the analysis band in Hz.")
    ],
    window_durations: Annotated[
        list[float],
        typer.Option("--window-duration", help="A crossfade window in seconds; repeat for several."),
    ],
    out: Annotated[Path, typer.Option("--out", dir_okay=False, help="Where to write the JSON result.")],
) -> None:
    """Evaluate the covariance error of the colored-noise simulator's crossfade for a PSD.

    The PSD is continued as for the errors ColoredNoiseSimulator(covariance_tolerance=...)
    reads (held flat below 5 Hz, and continued by a power law between 5 Hz and a first point
    above it) and the simulator run from zero to the Nyquist frequency, so the result is
    comparable with them. The error describes that configuration, not
    the PSD run with the simulator's defaults: the continued target is written next to the
    result, and the result records the ColoredNoiseSimulator settings that, with that target,
    generate the noise it describes. It also records the gwmock-noise version and the PSD's
    SHA-256, and is rewritten after each window, so a long run keeps the windows it has
    finished. Long windows are expensive: at 4096 Hz a 64 s window takes minutes, a 256 s
    window hours, and windows of 1024 s and more need gigabytes of memory.
    """
    path = _psd_path(psd)
    target_out = out.with_name(f"{out.stem}_target.txt")
    for written in (out, target_out):
        if written.resolve() == path.resolve():
            raise typer.BadParameter(f"writing {written} would overwrite the PSD it evaluates.", param_hint="--out")
    with tempfile.TemporaryDirectory() as directory:
        # Every window, the hash and the target come from one snapshot of the PSD, so a file changed during a long
        # run cannot make them describe different curves.
        snapshot = Path(directory) / path.name
        snapshot.write_bytes(path.read_bytes())
        _evaluate(
            psd,
            snapshot,
            sampling_frequency=sampling_frequency,
            analysis_band=analysis_band,
            window_durations=window_durations,
            out=out,
            target_out=target_out,
        )


def _evaluate(  # noqa: PLR0913 - the command's arguments, passed through.
    psd: str,
    path: Path,
    *,
    sampling_frequency: float,
    analysis_band: tuple[float, float],
    window_durations: list[float],
    out: Path,
    target_out: Path,
) -> None:
    """Evaluate each window for the PSD at ``path`` and write the result after each one."""
    try:
        target_frequencies, target_psd = crossfade_covariance.crossfade_target(path, sampling_frequency)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="PSD") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    target = crossfade_covariance.load_crossfade_errors()["target"]
    result: dict[str, Any] = {
        "gwmock_noise_version": __version__,
        "psd": {"argument": psd, "file": path.name, "sha256": crossfade_covariance.sha256_of(path)},
        "sampling_frequency": sampling_frequency,
        "analysis_band_hz": list(analysis_band),
        "target": target,
        "method": {
            "nyquist_guard_hz": crossfade_covariance.NYQUIST_GUARD_HZ,
            "oversample": crossfade_covariance.OVERSAMPLE,
            "convergence_tolerance": crossfade_covariance.CONVERGENCE_TOLERANCE,
            "first_convergence_check": crossfade_covariance.FIRST_CONVERGENCE_CHECK,
            "maximum_harmonics": crossfade_covariance.MAXIMUM_HARMONICS,
        },
        "windows": [],
    }
    for window_duration in sorted(set(window_durations)):
        start = time.perf_counter()
        try:
            error = crossfade_covariance.crossfade_covariance_error(
                path, sampling_frequency, window_duration, analysis_band
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        result["windows"].append(
            {
                "window_duration": error.window_duration,
                "eps_total": error.error_band.total,
                "eps_stationary": error.error_band.stationary,
                "eps_cyclic": error.error_band.cyclic,
                "error_band_hz": list(error.error_band.band_hz),
                "eps_total_analysis_band": error.analysis_band.total,
                "harmonics": error.harmonics,
                "converged": error.converged,
                "wall_seconds": time.perf_counter() - start,
                "simulator": {
                    "psd_file": target_out.name,
                    **crossfade_covariance.simulator_settings(sampling_frequency, error.window_duration),
                },
            }
        )
        if "target_file" not in result:
            # Written once the first window is accepted, so a refused configuration leaves nothing behind.
            np.savetxt(target_out, np.column_stack([target_frequencies, target_psd]), fmt="%.17e")
            result["target_file"] = {"file": target_out.name, "sha256": crossfade_covariance.sha256_of(target_out)}
        _write_atomically(out, json.dumps(result, indent=4, sort_keys=True, allow_nan=False) + "\n")
        state = "converged" if error.converged else "NOT converged: not an upper bound"
        typer.echo(f"{error.window_duration:g} s: eps_total = {error.error_band.total:.4g} ({state})")
    settings = crossfade_covariance.simulator_settings(sampling_frequency, window_duration=0.0)
    arguments = ", ".join(f"{name}={value!r}" for name, value in settings.items() if name != "window_duration")
    typer.echo(
        f"Wrote {out} and the target it evaluated, {target_out}. To generate the noise an error describes, use "
        f"ColoredNoiseSimulator(psd_file={str(target_out)!r}, {arguments}, window_duration=<its window>)."
    )
