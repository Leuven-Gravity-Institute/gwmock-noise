"""CLI entry point for building gengli blip-population tables."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from gwmock_noise.glitches.gengli import write_blip_population_file


def build_blip_glitch_table(
    gravity_spy_csv: Annotated[
        Path,
        typer.Option("--gravity-spy-csv", exists=True, dir_okay=False, readable=True),
    ],
    out: Annotated[Path, typer.Option("--out", dir_okay=False)],
    threshold: Annotated[float, typer.Option("--threshold", min=0.0, max=1.0)] = 0.9,
    blip_column: Annotated[str, typer.Option("--blip-column")] = "Blip",
    snr_column: Annotated[str, typer.Option("--snr-column")] = "snr",
) -> None:
    """Build a gengli blip-population table from a GravitySpy CSV export."""
    with gravity_spy_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if blip_column not in fieldnames:
            raise typer.BadParameter(f"CSV is missing the '{blip_column}' column.", param_hint="--blip-column")
        if snr_column not in fieldnames:
            raise typer.BadParameter(f"CSV is missing the '{snr_column}' column.", param_hint="--snr-column")

        snr_samples: list[float] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                blip_score = float(row[blip_column])
                snr = float(row[snr_column])
            except (TypeError, ValueError) as exc:
                raise typer.BadParameter(
                    f"Row {row_number} contains a non-numeric '{blip_column}' or '{snr_column}' value."
                ) from exc

            if blip_score >= threshold and np.isfinite(snr) and snr > 0.0:
                snr_samples.append(snr)

    if not snr_samples:
        raise typer.BadParameter("No blip glitches matched the requested threshold.")

    write_blip_population_file(
        out,
        snr_samples=np.asarray(snr_samples, dtype=float),
        metadata={
            "source": "gravity_spy_csv",
            "threshold": threshold,
            "blip_column": blip_column,
            "snr_column": snr_column,
        },
    )
    typer.echo(f"Wrote {len(snr_samples)} blip SNR samples to {out}")
