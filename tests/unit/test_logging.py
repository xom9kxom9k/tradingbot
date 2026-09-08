"""Log sink configuration."""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from tradingbot.config.models import LoggingConfig
from tradingbot.core.logging import setup_logging


def _configure(tmp_path: Path, level: str = "INFO") -> LoggingConfig:
    config = LoggingConfig(
        level=level,  # type: ignore[arg-type]
        file=tmp_path / "logs" / "app.log",
        json_file=tmp_path / "logs" / "app.jsonl",
    )
    setup_logging(config, run_id="test-run", console=False)
    return config


def test_creates_log_directories_and_writes_both_sinks(tmp_path: Path) -> None:
    config = _configure(tmp_path)
    logger.bind(symbol="BTC/USDT").info("hello")
    logger.complete()
    logger.remove()

    assert config.file.exists()
    assert "hello" in config.file.read_text(encoding="utf-8")
    assert "BTC/USDT" in config.file.read_text(encoding="utf-8")


def test_json_sink_contains_structured_context(tmp_path: Path) -> None:
    config = _configure(tmp_path)
    logger.bind(symbol="ETH/USDT", bar_ts="2024-01-01T00:00:00Z").warning("gap detected")
    logger.complete()
    logger.remove()

    records = [json.loads(line) for line in config.json_file.read_text().splitlines() if line]
    assert len(records) == 1
    record = records[0]
    assert record["record"]["level"]["name"] == "WARNING"
    assert record["record"]["extra"]["run_id"] == "test-run"
    assert record["record"]["extra"]["symbol"] == "ETH/USDT"
    assert record["record"]["message"] == "gap detected"


def test_level_filters_out_lower_severity(tmp_path: Path) -> None:
    config = _configure(tmp_path, level="WARNING")
    logger.info("ignored")
    logger.error("kept")
    logger.complete()
    logger.remove()

    contents = config.file.read_text(encoding="utf-8")
    assert "ignored" not in contents
    assert "kept" in contents


def test_setup_replaces_previous_sinks(tmp_path: Path) -> None:
    first = _configure(tmp_path / "first")
    second = _configure(tmp_path / "second")
    logger.info("only in second")
    logger.complete()
    logger.remove()

    assert "only in second" not in first.file.read_text(encoding="utf-8")
    assert "only in second" in second.file.read_text(encoding="utf-8")
