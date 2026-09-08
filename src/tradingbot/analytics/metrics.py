"""Performance and risk metrics (tech.md section 9.1).

Every formula lives in its own small function so it can be checked against a
hand-computed answer. The composite :func:`compute_metrics` only wires them
together and never does arithmetic of its own.

Conventions used throughout:

* returns are fractions, not percentages, until they are written out;
* drawdowns are reported as positive numbers (a 20% drawdown is ``0.20``);
* crypto trades every day, so annualisation uses 365 days rather than 252.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from tradingbot.config.models import CostsConfig
from tradingbot.core.enums import Side

DAYS_PER_YEAR = 365.0
TIMEFRAME_MINUTES: dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "6h": 360,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


def periods_per_year(timeframe: str) -> float:
    """How many candles of ``timeframe`` fit into a year of 24/7 trading."""
    minutes = TIMEFRAME_MINUTES.get(timeframe)
    if minutes is None:
        raise ValueError(f"unknown timeframe: {timeframe}")
    return DAYS_PER_YEAR * 24 * 60 / minutes


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #


def total_return(equity: pd.Series) -> float:
    """Overall growth of the account as a fraction of its starting value."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def elapsed_years(index: pd.DatetimeIndex) -> float:
    """Length of the sample in years, measured from the timestamps."""
    if len(index) < 2:
        return 0.0
    span = index[-1] - index[0]
    return float(span.total_seconds() / (DAYS_PER_YEAR * 24 * 3600))


def cagr(equity: pd.Series) -> float:
    """Compound annual growth rate implied by the equity curve."""
    years = elapsed_years(pd.DatetimeIndex(equity.index))
    if years <= 0 or len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def buy_and_hold_return(close: pd.Series) -> float:
    """Return of simply holding the instrument over the same window."""
    if len(close) < 2 or close.iloc[0] <= 0:
        return 0.0
    return float(close.iloc[-1] / close.iloc[0] - 1.0)


def period_returns(equity: pd.Series, freq: str) -> pd.Series:
    """Returns resampled to ``freq`` (``ME`` for monthly, ``YE`` for yearly)."""
    if equity.empty:
        return pd.Series(dtype=float)
    last = equity.resample(freq).last().dropna()
    if last.empty:
        return pd.Series(dtype=float)
    # The first period is measured from the starting equity, not from itself.
    previous = last.shift(1)
    previous.iloc[0] = equity.iloc[0]
    return (last / previous - 1.0).dropna()


# --------------------------------------------------------------------------- #
# Risk
# --------------------------------------------------------------------------- #


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Fractional distance below the running peak, at every point in time."""
    if equity.empty:
        return pd.Series(dtype=float)
    peak = equity.cummax()
    return (peak - equity) / peak.replace(0.0, np.nan)


@dataclass(frozen=True, slots=True)
class DrawdownEpisode:
    """One excursion below a previous equity peak."""

    peak_ts: datetime
    trough_ts: datetime
    recovery_ts: datetime | None
    depth: float

    @property
    def recovered(self) -> bool:
        """True when equity climbed back to the old peak."""
        return self.recovery_ts is not None

    def duration_days(self, end: datetime) -> float:
        """Peak to recovery, or peak to ``end`` while still under water."""
        finish = self.recovery_ts or end
        return (finish - self.peak_ts).total_seconds() / 86400.0

    def recovery_days(self, end: datetime) -> float:
        """Trough to recovery, or trough to ``end`` while still under water."""
        finish = self.recovery_ts or end
        return (finish - self.trough_ts).total_seconds() / 86400.0


def drawdown_episodes(equity: pd.Series) -> list[DrawdownEpisode]:
    """Split the equity curve into its underwater periods."""
    if equity.empty:
        return []

    index = pd.DatetimeIndex(equity.index)
    values = equity.to_numpy(dtype=float)

    episodes: list[DrawdownEpisode] = []
    peak_value = float(values[0])
    peak_ts = index[0]
    trough_value = peak_value
    trough_ts = peak_ts
    under_water = False

    for moment, value in zip(index, values, strict=True):
        if value >= peak_value:
            if under_water:
                episodes.append(
                    DrawdownEpisode(
                        peak_ts=peak_ts,
                        trough_ts=trough_ts,
                        recovery_ts=moment,
                        depth=(peak_value - trough_value) / peak_value,
                    )
                )
                under_water = False
            peak_value = value
            peak_ts = moment
            trough_value = value
            trough_ts = moment
            continue

        if not under_water or value < trough_value:
            trough_value = value
            trough_ts = moment
        under_water = True

    if under_water:
        episodes.append(
            DrawdownEpisode(
                peak_ts=peak_ts,
                trough_ts=trough_ts,
                recovery_ts=None,
                depth=(peak_value - trough_value) / peak_value,
            )
        )
    return episodes


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Deepest drawdown as a fraction and in account currency."""
    if equity.empty:
        return 0.0, 0.0
    peak = equity.cummax()
    absolute = (peak - equity).max()
    fractional = drawdown_series(equity).max()
    return float(fractional if pd.notna(fractional) else 0.0), float(absolute)


def average_drawdown(equity: pd.Series) -> float:
    """Mean depth of the individual underwater episodes."""
    episodes = drawdown_episodes(equity)
    if not episodes:
        return 0.0
    return float(np.mean([episode.depth for episode in episodes]))


def annualized_volatility(returns: pd.Series, per_year: float) -> float:
    """Standard deviation of returns scaled to a year."""
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * math.sqrt(per_year))


def value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """Historical VaR, reported as a positive loss fraction."""
    if returns.empty:
        return 0.0
    quantile = float(np.quantile(returns.to_numpy(), 1.0 - level))
    return max(0.0, -quantile)


def conditional_var(returns: pd.Series, level: float = 0.95) -> float:
    """Average loss in the worst ``1 - level`` tail, as a positive fraction."""
    if returns.empty:
        return 0.0
    cutoff = float(np.quantile(returns.to_numpy(), 1.0 - level))
    tail = returns[returns <= cutoff]
    if tail.empty:
        return max(0.0, -cutoff)
    return max(0.0, -float(tail.mean()))


def ulcer_index(equity: pd.Series) -> float:
    """Root mean square drawdown, in percentage points."""
    drawdowns = drawdown_series(equity).dropna()
    if drawdowns.empty:
        return 0.0
    return float(np.sqrt(np.mean((drawdowns.to_numpy() * 100.0) ** 2)))


# --------------------------------------------------------------------------- #
# Risk-adjusted ratios
# --------------------------------------------------------------------------- #


def sharpe_ratio(returns: pd.Series, per_year: float, risk_free: float = 0.0) -> float:
    """Annualised excess return per unit of total volatility."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / per_year
    deviation = excess.std(ddof=1)
    # A perfectly constant series has a standard deviation of a few 1e-18 rather
    # than exactly zero, which would otherwise report a Sharpe in the billions.
    if not np.isfinite(deviation) or deviation < 1e-12:
        return 0.0
    return float(excess.mean() / deviation * math.sqrt(per_year))


def sortino_ratio(returns: pd.Series, per_year: float, risk_free: float = 0.0) -> float:
    """Like Sharpe, but only downside deviation counts as risk."""
    if len(returns) < 2:
        return 0.0
    excess = returns - risk_free / per_year
    downside = np.minimum(excess.to_numpy(), 0.0)
    deviation = math.sqrt(float(np.mean(downside**2)))
    if deviation < 1e-12:
        return 0.0
    return float(excess.mean() / deviation * math.sqrt(per_year))


def calmar_ratio(annual_growth: float, max_dd: float) -> float:
    """Annual growth divided by the deepest drawdown."""
    if max_dd <= 0:
        return 0.0
    return annual_growth / max_dd


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Probability-weighted gains over losses relative to ``threshold``."""
    if returns.empty:
        return 0.0
    excess = returns.to_numpy() - threshold
    gains = float(np.sum(excess[excess > 0]))
    losses = -float(np.sum(excess[excess < 0]))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


# --------------------------------------------------------------------------- #
# Trade statistics
# --------------------------------------------------------------------------- #


def longest_streak(flags: Sequence[bool]) -> int:
    """Longest run of consecutive ``True`` values."""
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def profit_factor(pnl: pd.Series) -> float:
    """Gross profit divided by gross loss."""
    if pnl.empty:
        return 0.0
    profit = float(pnl[pnl > 0].sum())
    loss = -float(pnl[pnl < 0].sum())
    if loss == 0:
        return float("inf") if profit > 0 else 0.0
    return profit / loss


def exposure_pct(equity: pd.DataFrame) -> float:
    """Share of bars during which at least one position was open."""
    if equity.empty or "open_positions" not in equity:
        return 0.0
    return float((equity["open_positions"] > 0).mean())


@dataclass(frozen=True, slots=True)
class TradeStats:
    """Everything derivable from the trade journal alone."""

    trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    expectancy_usdt: float = 0.0
    avg_win_r: float = 0.0
    avg_loss_r: float = 0.0
    max_win_streak: int = 0
    max_loss_streak: int = 0
    avg_bars_held: float = 0.0
    avg_hours_held: float = 0.0
    avg_mae_r: float = 0.0
    avg_mfe_r: float = 0.0
    best_trade_r: float = 0.0
    worst_trade_r: float = 0.0
    exit_reasons: dict[str, int] = field(default_factory=dict)


def trade_stats(trades: pd.DataFrame) -> TradeStats:
    """Summarise the trade journal."""
    if trades.empty:
        return TradeStats()

    pnl = trades["pnl_net"]
    pnl_r = trades["pnl_r"]
    wins = pnl > 0
    losses = pnl < 0
    held = trades["exit_ts"] - trades["entry_ts"]

    return TradeStats(
        trades=len(trades),
        long_trades=int((trades["side"] == Side.LONG.value).sum()),
        short_trades=int((trades["side"] == Side.SHORT.value).sum()),
        win_rate=float(wins.mean()),
        profit_factor=profit_factor(pnl),
        expectancy_r=float(pnl_r.mean()),
        expectancy_usdt=float(pnl.mean()),
        avg_win_r=float(pnl_r[wins].mean()) if wins.any() else 0.0,
        avg_loss_r=float(pnl_r[losses].mean()) if losses.any() else 0.0,
        max_win_streak=longest_streak(wins.tolist()),
        max_loss_streak=longest_streak(losses.tolist()),
        avg_bars_held=float(trades["bars_held"].mean()),
        avg_hours_held=float(held.dt.total_seconds().mean() / 3600.0),
        avg_mae_r=float(trades["mae"].mean()),
        avg_mfe_r=float(trades["mfe"].mean()),
        best_trade_r=float(pnl_r.max()),
        worst_trade_r=float(pnl_r.min()),
        exit_reasons={str(k): int(v) for k, v in trades["exit_reason"].value_counts().items()},
    )


def aggregate_trades(trades: pd.DataFrame, by: str) -> list[dict[str, Any]]:
    """Per-group trade statistics, for example by ``symbol`` or by ``side``."""
    if trades.empty or by not in trades:
        return []
    rows: list[dict[str, Any]] = []
    for key, group in trades.groupby(by, sort=True):
        stats = trade_stats(group)
        rows.append({by: str(key), "pnl_net": float(group["pnl_net"].sum()), **asdict(stats)})
    return rows


def monthly_trade_pnl(trades: pd.DataFrame) -> dict[str, float]:
    """Net PnL grouped by calendar month of the exit."""
    if trades.empty:
        return {}
    grouped = trades.groupby(trades["exit_ts"].dt.strftime("%Y-%m"))["pnl_net"].sum()
    return {str(period): float(value) for period, value in grouped.items()}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CostSummary:
    """What the account paid to be in the market."""

    fees: float = 0.0
    funding: float = 0.0
    slippage_estimate: float = 0.0
    total: float = 0.0
    share_of_gross_profit: float = 0.0


def cost_summary(
    trades: pd.DataFrame,
    costs: CostsConfig | None = None,
    slippage: float | None = None,
) -> CostSummary:
    """Aggregate trading costs and weigh them against gross profit.

    The engine measures slippage exactly and passes it in. When it is unavailable
    — for a journal loaded from disk, say — it is reconstructed from the
    configured model and the traded notional; the ATR model cannot be
    reconstructed because the concession depended on volatility at the moment of
    each fill, so it is reported as zero rather than guessed.

    The denominator is gross profit from winning trades, not net gross PnL, so
    the share stays meaningful for a strategy that costs turned into a loser.
    """
    if trades.empty:
        return CostSummary()

    fees = float(trades["fees"].sum())
    funding = float(trades["funding"].sum())

    if slippage is None:
        rate = 0.0
        if costs is not None:
            if costs.slippage_model == "fixed_bps":
                rate = costs.slippage_bps / 10_000.0
            elif costs.slippage_model == "percent":
                rate = costs.slippage_percent
        notional = float(((trades["entry_price"] + trades["exit_price"]) * trades["size"]).sum())
        slippage = notional * rate

    gross_profit = float(trades.loc[trades["pnl_gross"] > 0, "pnl_gross"].sum())
    total = fees + funding + slippage
    return CostSummary(
        fees=fees,
        funding=funding,
        slippage_estimate=slippage,
        total=total,
        share_of_gross_profit=total / gross_profit if gross_profit > 0 else 0.0,
    )


# --------------------------------------------------------------------------- #
# Composite
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """The full section 9.1 set, ready to be written to ``metrics.json``."""

    returns: dict[str, Any]
    risk: dict[str, Any]
    ratios: dict[str, Any]
    trading: dict[str, Any]
    costs: dict[str, Any]
    monthly_returns: dict[str, float]
    annual_returns: dict[str, float]
    by_symbol: list[dict[str, Any]]
    by_side: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Plain nested mapping suitable for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PerformanceMetrics:
        """Rebuild a metrics set that was read back from ``metrics.json``."""
        return cls(
            returns=dict(payload.get("returns", {})),
            risk=dict(payload.get("risk", {})),
            ratios=dict(payload.get("ratios", {})),
            trading=dict(payload.get("trading", {})),
            costs=dict(payload.get("costs", {})),
            monthly_returns=dict(payload.get("monthly_returns", {})),
            annual_returns=dict(payload.get("annual_returns", {})),
            by_symbol=list(payload.get("by_symbol", [])),
            by_side=list(payload.get("by_side", [])),
        )

    def headline(self) -> dict[str, Any]:
        """The handful of numbers worth putting on a summary card."""
        return {
            "total_return_pct": self.returns["total_return_pct"],
            "cagr_pct": self.returns["cagr_pct"],
            "max_drawdown_pct": self.risk["max_drawdown_pct"],
            "sharpe": self.ratios["sharpe"],
            "profit_factor": self.trading["profit_factor"],
            "win_rate_pct": self.trading["win_rate_pct"],
            "trades": self.trading["trades"],
        }


def compute_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    timeframe: str,
    risk_free_rate: float = 0.0,
    benchmark: Mapping[str, pd.Series] | None = None,
    costs: CostsConfig | None = None,
    slippage: float | None = None,
) -> PerformanceMetrics:
    """Compute every metric of section 9.1 for one run.

    Args:
        equity: Equity curve with at least an ``equity`` column, indexed by time.
        trades: Trade journal in the section 8.4 schema.
        timeframe: Candle size, used to annualise.
        risk_free_rate: Annual risk-free rate used by Sharpe and Sortino.
        benchmark: Close series per symbol, for the buy-and-hold comparison.
        costs: Cost model, used to estimate slippage when it was not measured.
        slippage: Slippage actually paid, as measured by the engine.
    """
    curve = equity["equity"] if "equity" in equity else pd.Series(dtype=float)
    returns = curve.pct_change().dropna() if len(curve) > 1 else pd.Series(dtype=float)
    per_year = periods_per_year(timeframe)

    growth = total_return(curve)
    annual_growth = cagr(curve)
    max_dd_pct, max_dd_abs = max_drawdown(curve)

    hold = {symbol: buy_and_hold_return(series) for symbol, series in (benchmark or {}).items()}
    hold_portfolio = float(np.mean(list(hold.values()))) if hold else 0.0

    episodes = drawdown_episodes(curve)
    deepest = max(episodes, key=lambda item: item.depth, default=None)
    end = curve.index[-1] if len(curve) else None

    stats = trade_stats(trades)
    summary = cost_summary(trades, costs, slippage)

    return PerformanceMetrics(
        returns={
            "total_return_pct": 100.0 * growth,
            "cagr_pct": 100.0 * annual_growth,
            "buy_hold_pct": {symbol: 100.0 * value for symbol, value in hold.items()},
            "buy_hold_portfolio_pct": 100.0 * hold_portfolio,
            "alpha_pct": 100.0 * (growth - hold_portfolio),
            "final_equity": float(curve.iloc[-1]) if len(curve) else 0.0,
            "bars": len(curve),
            "years": elapsed_years(pd.DatetimeIndex(curve.index)),
        },
        risk={
            "max_drawdown_pct": 100.0 * max_dd_pct,
            "max_drawdown_abs": max_dd_abs,
            "max_drawdown_duration_days": (
                deepest.duration_days(end) if deepest and end is not None else 0.0
            ),
            "max_drawdown_recovery_days": (
                deepest.recovery_days(end) if deepest and end is not None else 0.0
            ),
            "max_drawdown_recovered": bool(deepest.recovered) if deepest else True,
            "avg_drawdown_pct": 100.0 * average_drawdown(curve),
            "annualized_volatility_pct": 100.0 * annualized_volatility(returns, per_year),
            "var_95_pct": 100.0 * value_at_risk(returns),
            "cvar_95_pct": 100.0 * conditional_var(returns),
            "ulcer_index": ulcer_index(curve),
        },
        ratios={
            "sharpe": sharpe_ratio(returns, per_year, risk_free_rate),
            "sortino": sortino_ratio(returns, per_year, risk_free_rate),
            "calmar": calmar_ratio(annual_growth, max_dd_pct),
            "omega": omega_ratio(returns),
        },
        trading={
            **asdict(stats),
            "win_rate_pct": 100.0 * stats.win_rate,
            "exposure_pct": 100.0 * exposure_pct(equity),
            "net_pnl": float(trades["pnl_net"].sum()) if not trades.empty else 0.0,
        },
        costs=asdict(summary),
        monthly_returns={
            str(period): 100.0 * float(value)
            for period, value in period_returns(curve, "ME").items()
        },
        annual_returns=_by_year(period_returns(curve, "YE")),
        by_symbol=aggregate_trades(trades, "symbol"),
        by_side=aggregate_trades(trades, "side"),
    )


def _by_year(yearly: pd.Series) -> dict[str, float]:
    """Label yearly returns by calendar year and convert them to percent."""
    index = pd.DatetimeIndex(yearly.index)
    return {
        str(moment.year): 100.0 * float(value)
        for moment, value in zip(index, yearly.to_numpy(dtype=float), strict=True)
    }


__all__ = [
    "CostSummary",
    "DrawdownEpisode",
    "PerformanceMetrics",
    "TradeStats",
    "aggregate_trades",
    "annualized_volatility",
    "average_drawdown",
    "buy_and_hold_return",
    "cagr",
    "calmar_ratio",
    "compute_metrics",
    "conditional_var",
    "cost_summary",
    "drawdown_episodes",
    "drawdown_series",
    "elapsed_years",
    "exposure_pct",
    "longest_streak",
    "max_drawdown",
    "monthly_trade_pnl",
    "omega_ratio",
    "period_returns",
    "periods_per_year",
    "profit_factor",
    "sharpe_ratio",
    "sortino_ratio",
    "total_return",
    "trade_stats",
    "ulcer_index",
    "value_at_risk",
]
