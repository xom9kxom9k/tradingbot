"""Integration checks against the live Binance public API.

Excluded from CI by the ``integration`` marker; run them with ``make test-all``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from tradingbot.data.cache import ParquetCache
from tradingbot.data.ccxt_feed import CcxtDataFeed
from tradingbot.data.validators import validate_ohlcv

pytestmark = pytest.mark.integration


def test_downloads_recent_candles() -> None:
    since = datetime.now(tz=UTC) - timedelta(days=20)
    with CcxtDataFeed("binanceusdm") as feed:
        rows = feed.fetch_ohlcv("BTC/USDT", "4h", since, 100)

    assert len(rows) == 100
    assert all(row[2] >= row[3] for row in rows), "high must not be below low"


def test_market_metadata_has_real_limits() -> None:
    with CcxtDataFeed("binanceusdm") as feed:
        meta = feed.fetch_market("BTC/USDT")

    assert meta.lot_step > 0
    assert meta.price_tick > 0


def test_cached_download_passes_validation(tmp_path: Path) -> None:
    cache = ParquetCache(tmp_path, "binanceusdm")
    since = datetime.now(tz=UTC) - timedelta(days=30)
    with CcxtDataFeed("binanceusdm") as feed:
        result = cache.sync(feed, "BTC/USDT", "4h", since)

    report = validate_ohlcv(cache.read("BTC/USDT", "4h"), "BTC/USDT", "4h")
    assert result.total > 100
    assert report.is_valid, report.render()
