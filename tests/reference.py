"""Naive reference implementations of the indicators.

These are deliberately slow, loop-based transcriptions of the formulas in
tech.md section 7.2. They exist so the vectorised production code is checked
against an independent implementation rather than against itself.
"""

from __future__ import annotations

from math import isnan, nan

Series = list[float]


def sma(values: Series, period: int) -> Series:
    out = [nan] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = sum(values[i - period + 1 : i + 1]) / period
    return out


def _wilder_like(values: Series, period: int, alpha: float) -> Series:
    """Recursive smoothing seeded with the mean of the first ``period`` valid values."""
    out = [nan] * len(values)
    valid = [i for i, value in enumerate(values) if not isnan(value)]
    if len(valid) < period:
        return out
    seed_position = valid[period - 1]
    out[seed_position] = sum(values[i] for i in valid[:period]) / period
    for i in range(seed_position + 1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def ema(values: Series, period: int) -> Series:
    return _wilder_like(values, period, 2.0 / (period + 1.0))


def rma(values: Series, period: int) -> Series:
    return _wilder_like(values, period, 1.0 / period)


def true_range(high: Series, low: Series, close: Series) -> Series:
    out = [high[0] - low[0]]
    for i in range(1, len(high)):
        previous = close[i - 1]
        out.append(max(high[i] - low[i], abs(high[i] - previous), abs(low[i] - previous)))
    return out


def atr(high: Series, low: Series, close: Series, period: int) -> Series:
    return rma(true_range(high, low, close), period)


def donchian(high: Series, low: Series, period: int) -> tuple[Series, Series, Series]:
    upper = [nan] * len(high)
    lower = [nan] * len(high)
    mid = [nan] * len(high)
    for i in range(period, len(high)):
        window_high = high[i - period : i]
        window_low = low[i - period : i]
        upper[i] = max(window_high)
        lower[i] = min(window_low)
        mid[i] = (upper[i] + lower[i]) / 2.0
    return upper, lower, mid


def directional_movement(
    high: Series, low: Series, close: Series, period: int
) -> tuple[Series, Series, Series]:
    """Return ``(+DI, -DI, ADX)``."""
    size = len(high)
    plus_dm = [0.0] * size
    minus_dm = [0.0] * size
    for i in range(1, size):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0

    smoothed_tr = rma(true_range(high, low, close), period)
    smoothed_plus = rma(plus_dm, period)
    smoothed_minus = rma(minus_dm, period)

    plus_di = [nan] * size
    minus_di = [nan] * size
    dx = [nan] * size
    for i in range(size):
        if isnan(smoothed_tr[i]):
            continue
        if smoothed_tr[i] == 0:
            plus_di[i] = minus_di[i] = dx[i] = 0.0
            continue
        plus_di[i] = 100.0 * smoothed_plus[i] / smoothed_tr[i]
        minus_di[i] = 100.0 * smoothed_minus[i] / smoothed_tr[i]
        total = plus_di[i] + minus_di[i]
        dx[i] = 0.0 if total == 0 else 100.0 * abs(plus_di[i] - minus_di[i]) / total

    return plus_di, minus_di, rma(dx, period)


def rsi(values: Series, period: int) -> Series:
    # The first bar has no preceding change, hence NaN rather than a zero gain.
    gains: Series = [nan] + [0.0] * (len(values) - 1)
    losses: Series = [nan] + [0.0] * (len(values) - 1)
    for i in range(1, len(values)):
        change = values[i] - values[i - 1]
        gains[i] = max(change, 0.0)
        losses[i] = max(-change, 0.0)

    average_gain = rma(gains, period)
    average_loss = rma(losses, period)
    out = [nan] * len(values)
    for i in range(len(values)):
        if isnan(average_gain[i]):
            continue
        if average_loss[i] == 0:
            out[i] = 50.0 if average_gain[i] == 0 else 100.0
        else:
            out[i] = 100.0 - 100.0 / (1.0 + average_gain[i] / average_loss[i])
    return out
