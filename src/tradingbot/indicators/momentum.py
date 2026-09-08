"""Momentum indicators: RSI, directional movement (ADX) and rate of change."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.core.exceptions import StrategyError
from tradingbot.indicators.trend import rma
from tradingbot.indicators.volatility import true_range


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative strength index using Wilder smoothing.

    A stretch with no losses yields 100 rather than infinity.
    """
    if period < 1:
        raise StrategyError(f"rsi period must be >= 1, got {period}")
    change = series.diff()
    gains = change.clip(lower=0.0)
    losses = (-change).clip(lower=0.0)
    average_gain = rma(gains, period)
    average_loss = rma(losses, period)

    strength = average_gain / average_loss.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + strength)
    result = result.where(average_loss != 0.0, 100.0)
    result = result.where(~((average_loss == 0.0) & (average_gain == 0.0)), 50.0)
    return result.where(average_gain.notna()).rename(f"rsi_{period}")


def directional_movement(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """+DI, -DI, DX and ADX, computed exactly as Wilder defined them.

    Returns:
        A frame with columns ``plus_di``, ``minus_di``, ``dx`` and ``adx``.
    """
    if period < 1:
        raise StrategyError(f"adx period must be >= 1, got {period}")

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
        dtype="float64",
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=high.index,
        dtype="float64",
    )

    smoothed_range = rma(true_range(high, low, close), period)
    # A perfectly flat stretch has zero true range: call that "no directional
    # movement" rather than letting 0/0 poison the rest of the series.
    divisor = smoothed_range.replace(0.0, np.nan)
    flat = smoothed_range == 0.0
    plus_di = (100.0 * rma(plus_dm, period) / divisor).where(~flat, 0.0)
    minus_di = (100.0 * rma(minus_dm, period) / divisor).where(~flat, 0.0)
    plus_di = plus_di.where(smoothed_range.notna())
    minus_di = minus_di.where(smoothed_range.notna())

    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0.0, np.nan)
    dx = dx.where(di_sum != 0.0, 0.0).where(plus_di.notna())

    return pd.DataFrame(
        {
            "plus_di": plus_di.rename("plus_di"),
            "minus_di": minus_di.rename("minus_di"),
            "dx": dx.rename("dx"),
            "adx": rma(dx, period).rename("adx"),
        }
    )


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average directional index: trend strength regardless of direction."""
    return directional_movement(high, low, close, period)["adx"].rename(f"adx_{period}")


def roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of change over ``period`` bars, in percent."""
    if period < 1:
        raise StrategyError(f"roc period must be >= 1, got {period}")
    return ((series / series.shift(period) - 1.0) * 100.0).rename(f"roc_{period}")
