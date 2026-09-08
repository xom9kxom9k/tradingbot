"""Smoke tests for the CLI wiring."""

from __future__ import annotations

from typer.testing import CliRunner

from tradingbot import __version__
from tradingbot.cli import app

runner = CliRunner()


def test_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in ("data", "backtest", "optimize", "walkforward", "live", "telegram"):
        assert group in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
