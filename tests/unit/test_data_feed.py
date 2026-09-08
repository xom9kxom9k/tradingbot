"""Feed pagination and the ccxt adapter, exercised without touching the network."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import ccxt
import pytest

from tradingbot.core.exceptions import FeedError
from tradingbot.core.models import MarketMeta
from tradingbot.data.ccxt_feed import CcxtDataFeed, _step
from tradingbot.data.feed import DataFeed, OhlcvRow, rows_to_frame, timeframe_delta

START = datetime(2024, 1, 1, tzinfo=UTC)
STEP = timedelta(hours=4)


def synthetic_rows(count: int, start: datetime = START, step: timedelta = STEP) -> list[OhlcvRow]:
    """Deterministic candles with a gently rising price."""
    rows: list[OhlcvRow] = []
    for i in range(count):
        ts = (start + i * step).timestamp() * 1000
        base = 100.0 + i
        rows.append([ts, base, base + 2.0, base - 1.0, base + 1.0, 10.0 + i])
    return rows


class RecordingFeed(DataFeed):
    """In-memory feed that hands out a fixed series one page at a time."""

    name = "recording"

    def __init__(self, rows: list[OhlcvRow]) -> None:
        self.rows = rows
        self.calls: list[tuple[datetime | None, int]] = []

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: datetime | None, limit: int
    ) -> list[OhlcvRow]:
        self.calls.append((since, limit))
        since_ms = since.timestamp() * 1000 if since else 0.0
        available = [row for row in self.rows if row[0] >= since_ms]
        return available[:limit]

    def fetch_market(self, symbol: str) -> MarketMeta:
        return MarketMeta.permissive(symbol)


class TestTimeframeDelta:
    def test_supported_values(self) -> None:
        assert timeframe_delta("15m") == timedelta(minutes=15)
        assert timeframe_delta("4h") == timedelta(hours=4)
        assert timeframe_delta("1d") == timedelta(days=1)

    def test_unsupported_value(self) -> None:
        with pytest.raises(FeedError, match="unsupported timeframe"):
            timeframe_delta("3m")


class TestRowsToFrame:
    def test_schema_and_index(self) -> None:
        frame = rows_to_frame(synthetic_rows(3))
        assert list(frame.columns) == ["open", "high", "low", "close", "volume", "is_gap"]
        assert frame.index.name == "ts"
        assert str(frame.index.tz) == "UTC"
        assert frame.index[0] == START

    def test_duplicates_collapse_to_the_last_value(self) -> None:
        rows = synthetic_rows(2)
        duplicate = list(rows[0])
        duplicate[4] = 999.0
        frame = rows_to_frame([*rows, duplicate])
        assert len(frame) == 2
        assert frame.iloc[0]["close"] == 999.0

    def test_unsorted_input_is_sorted(self) -> None:
        rows = synthetic_rows(3)
        frame = rows_to_frame([rows[2], rows[0], rows[1]])
        assert frame.index.is_monotonic_increasing

    def test_empty_input(self) -> None:
        assert rows_to_frame([]).empty


class TestFetchRange:
    def test_pages_until_exhausted(self) -> None:
        feed = RecordingFeed(synthetic_rows(250))
        frame = feed.fetch_range("BTC/USDT", "4h", START, START + 249 * STEP, limit=100)
        assert len(frame) == 250
        assert len(feed.calls) == 3
        assert feed.calls[1][0] == START + 100 * STEP

    def test_respects_the_end_boundary(self) -> None:
        feed = RecordingFeed(synthetic_rows(50))
        frame = feed.fetch_range("BTC/USDT", "4h", START, START + 9 * STEP, limit=100)
        assert len(frame) == 10
        assert frame.index[-1] == START + 9 * STEP

    def test_stops_on_empty_page(self) -> None:
        feed = RecordingFeed([])
        assert feed.fetch_range("BTC/USDT", "4h", START, START + STEP).empty

    def test_rejects_a_feed_that_goes_backwards(self) -> None:
        class BackwardsFeed(RecordingFeed):
            def fetch_ohlcv(
                self, symbol: str, timeframe: str, since: datetime | None, limit: int
            ) -> list[OhlcvRow]:
                return synthetic_rows(1, start=START - 10 * STEP)

        feed = BackwardsFeed(synthetic_rows(10))
        with pytest.raises(FeedError, match="before the requested cursor"):
            feed.fetch_range("BTC/USDT", "4h", START + 5 * STEP, START + 9 * STEP)

    def test_context_manager_closes(self) -> None:
        closed: list[bool] = []

        class ClosingFeed(RecordingFeed):
            def close(self) -> None:
                closed.append(True)

        with ClosingFeed([]) as feed:
            assert feed.name == "recording"
        assert closed == [True]


class FakeExchange:
    """Minimal stand-in for a ccxt exchange client."""

    def __init__(self, rows: list[OhlcvRow] | None = None, markets: dict[str, Any] | None = None):
        self._rows = rows if rows is not None else synthetic_rows(5)
        self.markets = markets or {}
        self.load_calls = 0
        self.closed = False

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None, limit: int
    ) -> list[list[float]]:
        return self._rows[:limit]

    def load_markets(self) -> dict[str, Any]:
        self.load_calls += 1
        return self.markets

    def close(self) -> None:
        self.closed = True


class TestCcxtFeed:
    def test_fetch_ohlcv_returns_floats(self) -> None:
        feed = CcxtDataFeed("binanceusdm", exchange=FakeExchange())
        rows = feed.fetch_ohlcv("BTC/USDT", "4h", START, 5)
        assert len(rows) == 5
        assert all(isinstance(value, float) for value in rows[0])

    def test_bad_symbol_becomes_feed_error(self) -> None:
        class Rejecting(FakeExchange):
            def fetch_ohlcv(
                self, symbol: str, timeframe: str, since: int | None, limit: int
            ) -> list[list[float]]:
                raise ccxt.BadSymbol("no such market")

        feed = CcxtDataFeed("binanceusdm", exchange=Rejecting())
        with pytest.raises(FeedError, match="not listed"):
            feed.fetch_ohlcv("FOO/USDT", "4h", START, 5)

    def test_transient_error_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        attempts = {"count": 0}

        class Flaky(FakeExchange):
            def fetch_ohlcv(
                self, symbol: str, timeframe: str, since: int | None, limit: int
            ) -> list[list[float]]:
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise ccxt.NetworkError("connection reset")
                return synthetic_rows(2)

        monkeypatch.setattr("tradingbot.data.ccxt_feed.wait_exponential", lambda **_: 0)
        feed = CcxtDataFeed("binanceusdm", exchange=Flaky())
        feed.fetch_ohlcv.retry.wait = lambda *_args, **_kwargs: 0  # type: ignore[attr-defined]
        assert len(feed.fetch_ohlcv("BTC/USDT", "4h", START, 2)) == 2
        assert attempts["count"] == 3

    def test_unknown_exchange(self) -> None:
        with pytest.raises(FeedError, match="unknown exchange"):
            CcxtDataFeed("definitely_not_an_exchange")

    def test_market_metadata_is_parsed(self) -> None:
        markets = {
            "BTC/USDT": {
                "precision": {"price": 2, "amount": 3},
                "limits": {"amount": {"min": 0.001}, "cost": {"min": 5.0}},
            }
        }
        feed = CcxtDataFeed("binanceusdm", exchange=FakeExchange(markets=markets))
        meta = feed.fetch_market("BTC/USDT")
        assert meta.price_tick == pytest.approx(0.01)
        assert meta.lot_step == pytest.approx(0.001)
        assert meta.min_size == pytest.approx(0.001)
        assert meta.min_notional == pytest.approx(5.0)

    def test_markets_are_loaded_once(self) -> None:
        exchange = FakeExchange(markets={"BTC/USDT": {}})
        feed = CcxtDataFeed("binanceusdm", exchange=exchange)
        feed.fetch_market("BTC/USDT")
        feed.fetch_market("BTC/USDT")
        assert exchange.load_calls == 1

    def test_unknown_symbol_falls_back_to_permissive_limits(self) -> None:
        feed = CcxtDataFeed("binanceusdm", exchange=FakeExchange(markets={}))
        meta = feed.fetch_market("DOGE/USDT")
        assert meta == MarketMeta.permissive("DOGE/USDT")

    def test_market_load_failure_is_not_fatal(self) -> None:
        class Broken(FakeExchange):
            def load_markets(self) -> dict[str, Any]:
                raise RuntimeError("offline")

        feed = CcxtDataFeed("binanceusdm", exchange=Broken())
        assert feed.fetch_market("BTC/USDT").lot_step == MarketMeta.permissive("BTC/USDT").lot_step

    def test_close_delegates_to_the_client(self) -> None:
        exchange = FakeExchange()
        CcxtDataFeed("binanceusdm", exchange=exchange).close()
        assert exchange.closed is True


class TestStepParsing:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(2, 0.01), (0, 1.0), (0.001, 0.001), (None, 0.5), (-1.0, 0.5)],
    )
    def test_precision_forms(self, value: object, expected: float) -> None:
        assert _step(value, default=0.5) == pytest.approx(expected)
