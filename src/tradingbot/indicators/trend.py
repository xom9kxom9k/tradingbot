"""Trend indicators: simple and exponential moving averages.

Every function here returns a series aligned to the input index, with ``NaN``
during the warmup period. Nothing looks forward: the value at bar ``t`` depends
only on bars ``<= t``, which is verified by the "stability of the past" test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingbot.core.exceptions import StrategyError


def _validate_period(period: int, name: str) -> None:
    """Reject periods that make an indicator meaningless."""
    if period < 1:
        raise StrategyError(f"{name} period must be >= 1, got {period}")


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average over ``period`` bars."""
    _validate_period(period, "sma")
    return series.rolling(window=period, min_periods=period).mean().rename(f"sma_{period}")


def _seeded_smoothing(series: pd.Series, period: int, alpha: float) -> pd.Series:
    """Recursive smoothing seeded with the mean of the first ``period`` valid values.

    Leading ``NaN`` values are skipped rather than treated as data. That matters
    for chained indicators such as ADX, where the input already carries a warmup
    hole: seeding on the first non-null value alone would bias the whole series.
    """
    values = series.astype("float64")
    valid_positions = np.flatnonzero(values.notna().to_numpy())
    if len(valid_positions) < period:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    seed_position = int(valid_positions[period - 1])
    seeded = values.copy()
    seeded.iloc[:seed_position] = np.nan
    seeded.iloc[seed_position] = float(values.iloc[valid_positions[:period]].mean())
    result = seeded.ewm(alpha=alpha, adjust=False, ignore_na=False).mean()
    result.iloc[:seed_position] = np.nan
    return result


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average seeded with the SMA of the first ``period`` bars.

    Uses ``alpha = 2 / (period + 1)`` as specified in tech.md section 7.2. The
    SMA seed is what makes the result reproducible: a plain recursive EMA from
    bar zero would depend on how much history happened to be loaded.
    """
    _validate_period(period, "ema")
    return _seeded_smoothing(series, period, 2.0 / (period + 1.0)).rename(f"ema_{period}")


def rma(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothed average, ``alpha = 1 / period``.

    Wilder's own indicators (ATR, ADX, RSI) all use this smoothing rather than a
    standard EMA, so keeping it separate avoids silently mixing the two.
    """
    _validate_period(period, "rma")
    return _seeded_smoothing(series, period, 1.0 / period).rename(f"rma_{period}")


def slope(series: pd.Series, lookback: int = 1) -> pd.Series:
    """Average change per bar over ``lookback`` bars."""
    _validate_period(lookback, "slope")
    return ((series - series.shift(lookback)) / lookback).rename(f"slope_{lookback}")


def slope_pct(series: pd.Series, lookback: int = 1) -> pd.Series:
    """Average change per bar, expressed as a fraction of the current value."""
    return (slope(series, lookback) / series.abs()).rename(f"slope_pct_{lookback}")
