"""Monte Carlo robustness: shuffle trade order, not prices (section 9.4).

The historical trade PnLs are treated as a bag of independent outcomes. We
draw many permutations of that bag, rebuild an equity path from each, and
read off the distribution of final capital and max drawdown. That answers
"how much of the result depends on the luck of the sequence?" rather than
re-simulating the market.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from tradingbot.backtest.runner import StoredRun
from tradingbot.config.models import AppConfig
from tradingbot.core.exceptions import BacktestError

PATHS_FILE = "montecarlo.parquet"
SUMMARY_FILE = "montecarlo.json"
PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(slots=True)
class MonteCarloResult:
    """Shuffled-trade paths plus the headline distribution."""

    paths: pd.DataFrame
    stats: dict[str, Any]
    iterations: int
    n_trades: int

    def to_json(self) -> str:
        """Pretty-printed summary for ``montecarlo.json``."""
        return json.dumps(self.stats, indent=2, default=str)


def run_monte_carlo(
    trades: pd.DataFrame,
    *,
    initial_capital: float,
    iterations: int = 1000,
    seed: int = 42,
    ruin_equity_pct: float = 0.0,
    dd_threshold_pct: float = 0.20,
) -> MonteCarloResult:
    """Bootstrap-permute ``pnl_net`` and rebuild equity paths.

    Args:
        trades: Journal with a ``pnl_net`` column. Empty journals are rejected.
        initial_capital: Starting equity of every path.
        iterations: Number of shuffled sequences (section 9.4 asks for ≥ 1000).
        seed: RNG seed so two runs of the same journal match.
        ruin_equity_pct: Fraction of starting capital treated as ruin (0 = bust).
        dd_threshold_pct: Drawdown level whose exceedance probability we report.
    """
    if trades.empty or "pnl_net" not in trades:
        raise BacktestError("monte carlo needs a non-empty trade journal with pnl_net")
    if iterations < 1:
        raise BacktestError("monte carlo iterations must be at least 1")

    pnls = trades["pnl_net"].to_numpy(dtype=float)
    n_trades = int(pnls.size)
    rng = np.random.default_rng(seed)
    paths = np.empty((n_trades + 1, iterations), dtype=float)
    paths[0] = initial_capital
    for i in range(iterations):
        paths[1:, i] = initial_capital + np.cumsum(rng.permutation(pnls))

    finals = paths[-1]
    drawdowns = np.array([_max_drawdown(paths[:, i]) for i in range(iterations)])
    ruin_level = initial_capital * ruin_equity_pct
    p_ruin = float(np.mean(np.any(paths <= ruin_level, axis=0)))
    p_dd = float(np.mean(drawdowns > dd_threshold_pct))

    stats: dict[str, Any] = {
        "iterations": iterations,
        "n_trades": n_trades,
        "seed": seed,
        "initial_capital": initial_capital,
        "ruin_equity_pct": ruin_equity_pct,
        "dd_threshold_pct": dd_threshold_pct,
        "final_equity": _percentile_block(finals),
        "max_drawdown": _percentile_block(drawdowns),
        "p_dd_gt_threshold": p_dd,
        "p_ruin": p_ruin,
    }
    logger.info(
        "monte carlo {n} iters: median {median:,.2f}, P(DD>{dd:.0%})={pdd:.1%}, P(ruin)={pr:.1%}",
        n=iterations,
        median=float(np.percentile(finals, 50)),
        dd=dd_threshold_pct,
        pdd=p_dd,
        pr=p_ruin,
    )
    frame = pd.DataFrame(paths, columns=[f"s{i}" for i in range(iterations)])
    return MonteCarloResult(paths=frame, stats=stats, iterations=iterations, n_trades=n_trades)


def run_monte_carlo_on_run(
    stored: StoredRun,
    config: AppConfig,
    *,
    iterations: int | None = None,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run the shuffle against a stored backtest and keep the original seed by default."""
    settings = config.montecarlo
    return run_monte_carlo(
        stored.trades,
        initial_capital=stored.initial_capital or config.backtest.initial_capital,
        iterations=iterations if iterations is not None else settings.iterations,
        seed=config.backtest.seed if seed is None else seed,
        ruin_equity_pct=settings.ruin_equity_pct,
        dd_threshold_pct=settings.dd_threshold_pct,
    )


def attach_monte_carlo(stored: StoredRun, result: MonteCarloResult) -> Path:
    """Write the fan-chart paths and the percentile summary next to an existing run."""
    stored.path.mkdir(parents=True, exist_ok=True)
    result.paths.to_parquet(stored.path / PATHS_FILE, engine="pyarrow")
    target = stored.path / SUMMARY_FILE
    target.write_text(result.to_json(), encoding="utf-8")
    return target


def _max_drawdown(path: np.ndarray) -> float:
    """Deepest fractional drawdown of one simulated equity path."""
    peak = np.maximum.accumulate(path)
    safe = np.where(peak > 0.0, peak, np.nan)
    dd = (peak - path) / safe
    value = np.nanmax(dd)
    return float(value) if np.isfinite(value) else 0.0


def _percentile_block(values: np.ndarray) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(values, p)) for p in PERCENTILES}
