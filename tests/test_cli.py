"""Tests for the CLI simulate command."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from gwsim_noise.cli.main import app

runner = CliRunner()


def test_simulate_command(tmp_path: Path) -> None:
    """Simulate command loads config and runs simulator."""
    out_dir = tmp_path / "output"
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"""
detectors: [H1]
duration: 2.0
output:
  directory: {out_dir}
  prefix: run
"""
    )
    result = runner.invoke(app, ["simulate", str(config_file)])
    assert result.exit_code == 0
    assert out_dir.exists()
    assert (out_dir / "run_H1.json").exists()


def test_simulate_command_missing_file() -> None:
    """Simulate command fails gracefully for missing config file."""
    result = runner.invoke(app, ["simulate", "/nonexistent/config.yaml"])
    assert result.exit_code != 0
