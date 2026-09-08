"""Throughput check for the definition of done: 5 symbols x 5 years under 30s."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest.runner import BacktestRunner
from tradingbot.config.models import AppConfig, BacktestConfig, DataConfig, ExchangeConfig

BARS_IN_FIVE_YEARS = 5 * 365 * 6  # 4h candles
SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"]


def series(bars: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, bars)))
    spread = np.abs(rng.normal(0.0, 0.005, bars))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2020-01-01", periods=bars, freq="4h", tz="UTC", name="ts")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + spread),
            "low": np.minimum(open_, close) * (1 - spread),
            "close": close,
            "volume": rng.uniform(100.0, 500.0, bars),
        },
        index=index,
    )


@pytest.mark.slow
def test_five_symbols_five_years_finish_quickly(tmp_path: Path) -> None:
    data = {symbol: series(BARS_IN_FIVE_YEARS, seed) for seed, symbol in enumerate(SYMBOLS)}
    config = AppConfig(
        exchange=ExchangeConfig(symbols=SYMBOLS, timeframe="4h"),
        data=DataConfig(cache_dir=tmp_path / "cache"),
        backtest=BacktestConfig(runs_dir=tmp_path / "runs"),
    )

    started = time.perf_counter()
    stored = BacktestRunner(config).run(data=data, save=False)
    elapsed = time.perf_counter() - started

    assert elapsed < 30.0, f"backtest took {elapsed:.1f}s"
    assert len(stored.equity) == BARS_IN_FIVE_YEARS
    assert not stored.trades.empty
