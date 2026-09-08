"""Logging setup built on loguru.

Three sinks are configured: a colourised console stream for humans, a rotating
plain-text file, and a rotating JSON-lines file for machine processing. Every
record carries the ``run_id``, ``symbol`` and ``bar_ts`` context fields so log
lines can be traced back to a specific run and candle.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from tradingbot.config.models import LoggingConfig

CONTEXT_DEFAULTS: dict[str, Any] = {"run_id": "-", "symbol": "-", "bar_ts": "-"}

CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>{extra[run_id]}</cyan>|<cyan>{extra[symbol]}</cyan> "
    "<magenta>{name}:{function}:{line}</magenta> - <level>{message}</level>"
)

FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{extra[run_id]} | {extra[symbol]} | {extra[bar_ts]} | "
    "{name}:{function}:{line} - {message}"
)


def setup_logging(
    config: LoggingConfig,
    *,
    run_id: str | None = None,
    console: bool = True,
) -> None:
    """Install the application log sinks, replacing any previous configuration.

    Args:
        config: Levels, file paths and rotation policy.
        run_id: Value bound to the ``run_id`` context field of every record.
        console: Set to ``False`` for processes where stderr must stay clean.
    """
    logger.remove()
    logger.configure(extra={**CONTEXT_DEFAULTS, "run_id": run_id or "-"})

    if console:
        logger.add(
            sys.stderr,
            level=config.level,
            format=CONSOLE_FORMAT,
            colorize=True,
            backtrace=False,
            diagnose=False,
        )

    _ensure_parent(config.file)
    logger.add(
        config.file,
        level=config.level,
        format=FILE_FORMAT,
        rotation=config.rotation,
        retention=config.retention,
        encoding="utf-8",
        enqueue=True,
    )

    _ensure_parent(config.json_file)
    logger.add(
        config.json_file,
        level=config.level,
        rotation=config.rotation,
        retention=config.retention,
        serialize=True,
        encoding="utf-8",
        enqueue=True,
    )


def _ensure_parent(path: Path) -> None:
    """Create the directory holding ``path`` if it does not exist yet."""
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = ["CONSOLE_FORMAT", "FILE_FORMAT", "logger", "setup_logging"]
