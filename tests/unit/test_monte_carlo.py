"""Monte Carlo shuffled-trade paths (tech.md 9.4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_backtest_runner import SYMBOL, make_config, trending_frame
from tradingbot.analytics.monte_carlo import (
    attach_monte_carlo,
    run_monte_carlo,
    run_monte_carlo_on_run,
)
from tradingbot.backtest.runner import BacktestRunner
from tradingbot.config.models import MonteCarloConfig, StrategyConfig
from tradingbot.core.exceptions import BacktestError
from tradingbot.reporting.html_report import extras_for


def _journal(pnls: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"pnl_net": pnls})


class TestMonteCarlo:
    def test_percentiles_and_ruin_on_a_known_bag(self) -> None:
        result = run_monte_carlo(
            _journal([10.0, -5.0, 20.0, -8.0]),
            initial_capital=100.0,
            iterations=200,
            seed=1,
            ruin_equity_pct=0.0,
            dd_threshold_pct=0.20,
        )
        assert result.paths.shape == (5, 200)
        finals = result.stats["final_equity"]
        # Permuting additive PnL never changes the terminal equity.
        assert finals["p5"] == pytest.approx(117.0)
        assert finals["p95"] == pytest.approx(117.0)
        assert set(finals) == {"p5", "p25", "p50", "p75", "p95"}
        assert 0.0 <= result.stats["p_dd_gt_threshold"] <= 1.0
        assert 0.0 <= result.stats["p_ruin"] <= 1.0

    def test_same_seed_is_byte_identical(self) -> None:
        trades = _journal([1.0, -2.0, 3.0, -1.5, 4.0])
        first = run_monte_carlo(trades, initial_capital=50.0, iterations=64, seed=99)
        second = run_monte_carlo(trades, initial_capital=50.0, iterations=64, seed=99)
        np.testing.assert_array_equal(first.paths.to_numpy(), second.paths.to_numpy())

    def test_empty_journal_is_rejected(self) -> None:
        with pytest.raises(BacktestError, match="pnl_net"):
            run_monte_carlo(pd.DataFrame(), initial_capital=100.0, iterations=10)

    def test_attaches_to_a_stored_run(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            strategy=StrategyConfig(
                name="donchian_trend",
                params={
                    "entry_channel": 8,
                    "exit_channel": 4,
                    "trend_fast": 8,
                    "trend_slow": 16,
                    "atr_period": 5,
                    "adx_period": 5,
                    "adx_min": 10.0,
                    "cooldown_bars": 1,
                },
            ),
            montecarlo=MonteCarloConfig(iterations=50),
        )
        stored = BacktestRunner(config).run(data={SYMBOL: trending_frame(bars=400)})
        if stored.trades.empty:
            pytest.skip("synthetic series produced no trades")
        result = run_monte_carlo_on_run(stored, config, iterations=50)
        attach_monte_carlo(stored, result)
        assert (stored.path / "montecarlo.parquet").is_file()
        extras = extras_for(stored)
        assert extras.montecarlo_paths is not None
        assert extras.montecarlo_paths.shape[1] == 50
