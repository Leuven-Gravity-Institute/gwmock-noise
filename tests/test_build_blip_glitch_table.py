"""Tests for the blip-population CLI builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from gwmock_noise.cli.main import app
from gwmock_noise.glitches.gengli import read_blip_population_file

runner = CliRunner()


def test_build_blip_glitch_table_writes_hdf5_population(tmp_path: Path) -> None:
    """The CLI filters the CSV and emits the expected population schema."""
    csv_path = tmp_path / "gravity_spy.csv"
    out_path = tmp_path / "glitches.h5"
    csv_path.write_text("event_id,Blip,snr\n1,0.95,8.0\n2,0.50,6.0\n3,0.91,12.0\n", encoding="utf-8")

    result = runner.invoke(app, ["build-blip-glitch-table", "--gravity-spy-csv", str(csv_path), "--out", str(out_path)])

    assert result.exit_code == 0
    np.testing.assert_allclose(read_blip_population_file(out_path), np.array([8.0, 12.0]))


def test_build_blip_glitch_table_rejects_missing_required_columns(tmp_path: Path) -> None:
    """The CLI reports missing CSV columns explicitly."""
    csv_path = tmp_path / "gravity_spy.csv"
    out_path = tmp_path / "glitches.h5"
    csv_path.write_text("event_id,score\n1,0.95\n", encoding="utf-8")

    result = runner.invoke(app, ["build-blip-glitch-table", "--gravity-spy-csv", str(csv_path), "--out", str(out_path)])

    assert result.exit_code != 0
    assert "missing the 'Blip' column" in result.output
