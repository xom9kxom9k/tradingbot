"""Event-driven backtesting: broker, portfolio, engine and run storage."""

from __future__ import annotations

from tradingbot.backtest.broker import Fill, PaperBroker
from tradingbot.backtest.engine import BacktestEngine, BacktestResult
from tradingbot.backtest.portfolio import OpenTrade, Portfolio
from tradingbot.backtest.runner import (
    BacktestRunner,
    StoredRun,
    build_strategy,
    latest_run_id,
    list_runs,
    load_run,
    run_summary,
    save_run,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestRunner",
    "Fill",
    "OpenTrade",
    "PaperBroker",
    "Portfolio",
    "StoredRun",
    "build_strategy",
    "latest_run_id",
    "list_runs",
    "load_run",
    "run_summary",
    "save_run",
]
