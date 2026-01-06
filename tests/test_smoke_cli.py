"""Smoke tests for CLI."""

import subprocess
import sys


def test_cli_version():
    """Test that CLI version command works."""
    result = subprocess.run(
        [sys.executable, "-m", "bioopt_bench", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "bioopt_bench" in result.stdout


def test_cli_list():
    """Test that CLI list command works."""
    result = subprocess.run(
        [sys.executable, "-m", "bioopt_bench", "list"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "ALGORITHMS" in result.stdout
    assert "ga" in result.stdout
    assert "pso" in result.stdout


def test_cli_list_json():
    """Test that CLI list --json command works."""
    result = subprocess.run(
        [sys.executable, "-m", "bioopt_bench", "list", "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert '"algorithms"' in result.stdout


def test_cli_run_help():
    """Test that CLI run help works."""
    result = subprocess.run(
        [sys.executable, "-m", "bioopt_bench", "run", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--task" in result.stdout
    assert "--algo" in result.stdout
