"""Regenerate the deterministic test fixtures.

Run with ``python scripts/make_fixtures.py``. Two files are produced:

``tests/fixtures/ohlcv_200.csv``
    200 synthetic 4h candles built from a fixed seed.
``tests/fixtures/indicators_expected.csv``
    Golden indicator values computed by the *naive* reference implementation in
    ``tests/reference.py``, never by the production code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

import reference  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
BARS = 200
SEED = 20260908


def build_candles(bars: int = BARS, seed: int = SEED) -> pd.DataFrame:
    """Random walk with a mild trend, shaped into valid OHLCV candles."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(loc=0.15, scale=2.0, size=bars)
    closes = 100.0 + np.cumsum(steps)

    opens = np.empty(bars)
    opens[0] = 100.0
    opens[1:] = closes[:-1]

    wiggle = np.abs(rng.normal(loc=0.0, scale=1.2, size=bars))
    highs = np.maximum(opens, closes) + wiggle
    lows = np.minimum(opens, closes) - np.abs(rng.normal(0.0, 1.2, size=bars))
    volumes = np.abs(rng.normal(1_000.0, 200.0, size=bars))

    index = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC", name="ts")
    return pd.DataFrame(
        {
            "open": opens.round(4),
            "high": highs.round(4),
            "low": lows.round(4),
            "close": closes.round(4),
            "volume": volumes.round(4),
        },
        index=index,
    )


def build_expected(candles: pd.DataFrame) -> pd.DataFrame:
    """Compute every golden series with the loop-based reference code."""
    high = candles["high"].tolist()
    low = candles["low"].tolist()
    close = candles["close"].tolist()

    upper, lower, mid = reference.donchian(high, low, 20)
    plus_di, minus_di, adx = reference.directional_movement(high, low, close, 14)

    return pd.DataFrame(
        {
            "sma_20": reference.sma(close, 20),
            "ema_50": reference.ema(close, 50),
            "ema_200": reference.ema(close, 200),
            "true_range": reference.true_range(high, low, close),
            "atr_14": reference.atr(high, low, close, 14),
            "dc_upper": upper,
            "dc_lower": lower,
            "dc_mid": mid,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "adx": adx,
            "rsi_14": reference.rsi(close, 14),
        },
        index=candles.index,
    )


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    candles = build_candles()
    candles.to_csv(FIXTURES / "ohlcv_200.csv", float_format="%.6f")
    build_expected(candles).to_csv(FIXTURES / "indicators_expected.csv", float_format="%.10f")
    print(f"wrote {BARS} candles and golden indicator values to {FIXTURES}")


if __name__ == "__main__":
    main()
