"""Volatility indicators: true range, ATR, rolling deviation and Keltner bands."""

from __future__ import annotations

import pandas as pd

from tradingbot.core.exceptions import StrategyError
from tradingbot.indicators.trend import ema, rma


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range.

    The first bar has no previous close, so its true range degenerates to the
    plain high-low span.
    """
    previous_close = close.shift(1)
    spans = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return spans.max(axis=1).rename("true_range")


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average true range, smoothed the Wilder way (tech.md section 7.2)."""
    return rma(true_range(high, low, close), period).rename(f"atr_{period}")


def atr_pct(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """ATR as a fraction of price, the scale-free volatility filter of section 7.5."""
    return (atr(high, low, close, period) / close).rename(f"atr_pct_{period}")


def stdev(series: pd.Series, period: int = 20, ddof: int = 1) -> pd.Series:
    """Rolling standard deviation."""
    if period < 2:
        raise StrategyError(f"stdev period must be >= 2, got {period}")
    return (
        series.rolling(window=period, min_periods=period).std(ddof=ddof).rename(f"stdev_{period}")
    )


def keltner_channels(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner channel: an EMA centre line with ATR-scaled bands."""
    middle = ema(close, period)
    band = atr(high, low, close, atr_period) * multiplier
    return pd.DataFrame(
        {
            "kc_middle": middle,
            "kc_upper": middle + band,
            "kc_lower": middle - band,
        }
    )
