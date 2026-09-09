"""Interactive exploration of stored backtest runs."""

from __future__ import annotations

from tradingbot.dashboard.compare import comparison_equity, headline_table, normalised_equity
from tradingbot.dashboard.filters import TradeFilters, apply_filters

__all__ = [
    "TradeFilters",
    "apply_filters",
    "comparison_equity",
    "headline_table",
    "normalised_equity",
]
