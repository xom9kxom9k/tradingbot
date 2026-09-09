"""Trade-table filters and session statistics used by the dashboard.

Kept free of Streamlit so the same functions can be unit-tested and reused by
CLI exporters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from tradingbot.core.enums import Side

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True, slots=True)
class TradeFilters:
    """Sidebar selection applied to the trade journal."""

    symbols: tuple[str, ...] | None = None
    sides: tuple[str, ...] | None = None
    start: date | None = None
    end: date | None = None
    outcome: str = "all"

    def is_active(self) -> bool:
        """True when at least one constraint is narrower than 'everything'."""
        return any(
            (
                self.symbols is not None,
                self.sides is not None,
                self.start is not None,
                self.end is not None,
                self.outcome != "all",
            )
        )


def apply_filters(trades: pd.DataFrame, filters: TradeFilters) -> pd.DataFrame:
    """Return the subset of ``trades`` that matches ``filters``.

    The period is applied to the exit timestamp: a trade belongs to the day it
    stopped being open, which is also how monthly PnL is attributed.
    """
    if trades.empty:
        return trades
    selected = trades
    if filters.symbols is not None:
        selected = selected[selected["symbol"].isin(filters.symbols)]
    if filters.sides is not None:
        selected = selected[selected["side"].isin(filters.sides)]
    if "exit_ts" in selected:
        exits = pd.to_datetime(selected["exit_ts"], utc=True)
        if filters.start is not None:
            start = pd.Timestamp(filters.start, tz="UTC")
            selected = selected[exits >= start]
            exits = pd.to_datetime(selected["exit_ts"], utc=True)
        if filters.end is not None:
            end = pd.Timestamp(filters.end, tz="UTC") + pd.Timedelta(days=1)
            selected = selected[exits < end]
    if filters.outcome == "win":
        selected = selected[selected["pnl_net"] > 0]
    elif filters.outcome == "loss":
        selected = selected[selected["pnl_net"] < 0]
    return selected


def available_symbols(trades: pd.DataFrame) -> list[str]:
    """Sorted unique symbols in the journal."""
    if trades.empty or "symbol" not in trades:
        return []
    return sorted(trades["symbol"].astype(str).unique())


def available_sides(trades: pd.DataFrame) -> list[str]:
    """Sides present in the journal, in a stable order."""
    if trades.empty or "side" not in trades:
        return [Side.LONG.value, Side.SHORT.value]
    present = set(trades["side"].astype(str))
    return [side for side in (Side.LONG.value, Side.SHORT.value) if side in present]


def hour_of_day_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Mean R and trade count by UTC hour of the entry."""
    if trades.empty:
        return pd.DataFrame(columns=["hour", "trades", "expectancy_r", "pnl_net"])
    stamps = pd.to_datetime(trades["entry_ts"], utc=True)
    grouped = trades.assign(hour=stamps.dt.hour).groupby("hour", sort=True)
    frame = grouped.agg(
        trades=("pnl_net", "size"), expectancy_r=("pnl_r", "mean"), pnl_net=("pnl_net", "sum")
    )
    return (
        frame.reindex(range(24))
        .fillna({"trades": 0, "expectancy_r": 0.0, "pnl_net": 0.0})
        .reset_index()
    )


def day_of_week_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """Mean R and trade count by weekday of the entry, Monday first."""
    if trades.empty:
        return pd.DataFrame(columns=["weekday", "trades", "expectancy_r", "pnl_net"])
    stamps = pd.to_datetime(trades["entry_ts"], utc=True)
    grouped = trades.assign(weekday=stamps.dt.day_name()).groupby("weekday")
    frame = grouped.agg(
        trades=("pnl_net", "size"), expectancy_r=("pnl_r", "mean"), pnl_net=("pnl_net", "sum")
    )
    return (
        frame.reindex(WEEKDAYS)
        .fillna({"trades": 0, "expectancy_r": 0.0, "pnl_net": 0.0})
        .reset_index()
    )
