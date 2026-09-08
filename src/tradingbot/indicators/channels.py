"""Donchian channel.

The channel is shifted one bar forward on purpose: at bar ``t`` it describes the
extremes of bars ``t-n .. t-1``. Without that shift the current bar would be
part of its own channel and could never break out of it, which would make every
breakout rule silently unfalsifiable.
"""

from __future__ import annotations

import pandas as pd

from tradingbot.core.exceptions import StrategyError


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> pd.DataFrame:
    """Upper, lower and mid lines of the Donchian channel.

    Args:
        high: Series of bar highs.
        low: Series of bar lows.
        period: Lookback length, excluding the current bar.

    Returns:
        A frame with ``dc_upper``, ``dc_lower`` and ``dc_mid`` columns.
    """
    if period < 1:
        raise StrategyError(f"donchian period must be >= 1, got {period}")

    upper = high.rolling(window=period, min_periods=period).max().shift(1)
    lower = low.rolling(window=period, min_periods=period).min().shift(1)
    return pd.DataFrame(
        {
            "dc_upper": upper.rename("dc_upper"),
            "dc_lower": lower.rename("dc_lower"),
            "dc_mid": ((upper + lower) / 2.0).rename("dc_mid"),
        }
    )


def donchian_width_pct(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Channel height relative to price: a rough measure of range expansion."""
    channel = donchian(high, low, period)
    return ((channel["dc_upper"] - channel["dc_lower"]) / close).rename(f"dc_width_pct_{period}")
