"""Smoke tests for the CLI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from tradingbot import __version__
from tradingbot.cli import app

runner = CliRunner()


def test_help_lists_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for group in (
        "data",
        "backtest",
        "optimize",
        "walkforward",
        "montecarlo",
        "live",
        "telegram",
        "dashboard",
    ):
        assert group in result.stdout


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def config_file(tmp_path: Path) -> Path:
    """A minimal config pointing every output at ``tmp_path``."""
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "exchange": {"symbols": ["BTC/USDT"], "timeframe": "4h"},
                "data": {"cache_dir": str(tmp_path / "cache")},
                "backtest": {"runs_dir": str(tmp_path / "runs")},
                "logging": {
                    "file": str(tmp_path / "log.txt"),
                    "json_file": str(tmp_path / "log.jsonl"),
                },
                "live": {"state_db": str(tmp_path / "state.db")},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_backtest_run_without_data_explains_itself(tmp_path: Path) -> None:
    result = runner.invoke(app, ["backtest", "run", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 1
    assert "data download" in result.output


def test_backtest_metrics_without_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["backtest", "metrics", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 1
    assert "no runs found" in result.output


def test_backtest_metrics_of_an_unknown_run(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["backtest", "metrics", "--run-id", "nope", "--config", str(config_file(tmp_path))],
    )
    assert result.exit_code == 1
    assert "not found" in result.output


def test_backtest_report_without_runs(tmp_path: Path) -> None:
    result = runner.invoke(app, ["backtest", "report", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 1
    assert "no runs found" in result.output


def test_optimize_grid_help() -> None:
    result = runner.invoke(app, ["optimize", "grid", "--help"])
    assert result.exit_code == 0
    assert "plateau" in result.stdout.lower() or "grid" in result.stdout.lower()


def test_walkforward_run_help() -> None:
    result = runner.invoke(app, ["walkforward", "run", "--help"])
    assert result.exit_code == 0
    assert "--is-months" in result.stdout


def test_montecarlo_run_help() -> None:
    result = runner.invoke(app, ["montecarlo", "run", "--help"])
    assert result.exit_code == 0
    assert "--run-id" in result.stdout


def test_optimize_grid_without_data_explains_itself(tmp_path: Path) -> None:
    result = runner.invoke(app, ["optimize", "grid", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 1
    assert "data download" in result.output


def test_dashboard_help() -> None:
    result = runner.invoke(app, ["dashboard", "--help"])
    assert result.exit_code == 0
    assert "Streamlit" in result.stdout or "dashboard" in result.stdout.lower()


def test_live_run_help() -> None:
    result = runner.invoke(app, ["live", "run", "--help"])
    assert result.exit_code == 0
    assert "--mode" in result.stdout


def test_live_status_on_a_fresh_database(tmp_path: Path) -> None:
    result = runner.invoke(app, ["live", "status", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 0
    assert "mode" in result.stdout
    assert "running" in result.stdout


def test_live_run_rejects_unknown_mode(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["live", "run", "--mode", "real", "--config", str(config_file(tmp_path))],
    )
    assert result.exit_code == 1
    assert "paper" in result.output


def test_telegram_test_help() -> None:
    result = runner.invoke(app, ["telegram", "test", "--help"])
    assert result.exit_code == 0
    assert "test" in result.stdout.lower()


def test_telegram_test_without_secrets_explains_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tradingbot.config.models import TelegramSecrets

    monkeypatch.setattr(
        "tradingbot.config.loader.load_secrets",
        lambda environ=None: TelegramSecrets(bot_token=None, chat_ids=[], enabled=True),
    )
    result = runner.invoke(app, ["telegram", "test", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 1
    assert "TELEGRAM_BOT_TOKEN" in result.output


def test_backtest_list_is_empty_before_any_run(tmp_path: Path) -> None:
    result = runner.invoke(app, ["backtest", "list", "--config", str(config_file(tmp_path))])
    assert result.exit_code == 0
    assert "no runs stored yet" in result.stdout
