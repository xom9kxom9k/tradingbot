"""Trading strategies and the registry that selects them by name."""

from __future__ import annotations

from tradingbot.strategy.base import Strategy
from tradingbot.strategy.donchian_trend import DonchianTrendParams, DonchianTrendStrategy
from tradingbot.strategy.registry import (
    available,
    create_strategy,
    get_strategy_class,
    register,
)

__all__ = [
    "DonchianTrendParams",
    "DonchianTrendStrategy",
    "Strategy",
    "available",
    "create_strategy",
    "get_strategy_class",
    "register",
]
