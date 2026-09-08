"""Performance analytics: metrics, aggregations and console summaries."""

from __future__ import annotations

from tradingbot.analytics.metrics import (
    CostSummary,
    DrawdownEpisode,
    PerformanceMetrics,
    TradeStats,
    aggregate_trades,
    compute_metrics,
    drawdown_series,
    max_drawdown,
    periods_per_year,
    trade_stats,
)
from tradingbot.analytics.summary import format_annual_returns, format_metrics

__all__ = [
    "CostSummary",
    "DrawdownEpisode",
    "PerformanceMetrics",
    "TradeStats",
    "aggregate_trades",
    "compute_metrics",
    "drawdown_series",
    "format_annual_returns",
    "format_metrics",
    "max_drawdown",
    "periods_per_year",
    "trade_stats",
]
