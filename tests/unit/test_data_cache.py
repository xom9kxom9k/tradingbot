"""Parquet cache behaviour, including incremental top-up."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tradingbot.core.exceptions import DataError
from tradingbot.data.cache import ParquetCache, sanitize_symbol
from tradingbot.data.feed import rows_to_frame

from .test_data_feed import STEP, RecordingFeed, synthetic_rows

START = datetime(2024, 1, 1, tzinfo=UTC)


@pytest.fixture
def cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(tmp_path / "ohlcv", "binanceusdm")


def frame_of(count: int) -> pd.DataFrame:
    return rows_to_frame(synthetic_rows(count))


class TestPaths:
    def test_sanitize_symbol(self) -> None:
        assert sanitize_symbol("BTC/USDT") == "BTCUSDT"
        assert sanitize_symbol("BTC/USDT:USDT") == "BTCUSDTUSDT"

    def test_path_layout(self, cache: ParquetCache) -> None:
        path = cache.path_for("BTC/USDT", "4h")
        assert path.parts[-3:] == ("binanceusdm", "BTCUSDT", "4h.parquet")


class TestReadWrite:
    def test_missing_file_returns_empty_frame(self, cache: ParquetCache) -> None:
        frame = cache.read("BTC/USDT", "4h")
        assert frame.empty
        assert list(frame.columns) == ["open", "high", "low", "close", "volume", "is_gap"]

    def test_roundtrip_preserves_index_and_dtypes(self, cache: ParquetCache) -> None:
        original = frame_of(10)
        cache.write("BTC/USDT", "4h", original)
        restored = cache.read("BTC/USDT", "4h")
        pd.testing.assert_frame_equal(original, restored)
        assert str(restored.index.tz) == "UTC"

    def test_naive_index_is_localised(self, cache: ParquetCache) -> None:
        frame = frame_of(3)
        frame.index = frame.index.tz_localize(None)
        cache.write("BTC/USDT", "4h", frame)
        assert str(cache.read("BTC/USDT", "4h").index.tz) == "UTC"

    def test_missing_column_is_rejected(self, cache: ParquetCache) -> None:
        with pytest.raises(DataError, match="missing columns"):
            cache.write("BTC/USDT", "4h", frame_of(3).drop(columns=["volume"]))

    def test_non_datetime_index_is_rejected(self, cache: ParquetCache) -> None:
        frame = frame_of(3).reset_index(drop=True)
        with pytest.raises(DataError, match="indexed by candle open time"):
            cache.write("BTC/USDT", "4h", frame)

    def test_corrupt_file_raises(self, cache: ParquetCache) -> None:
        path = cache.path_for("BTC/USDT", "4h")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not parquet", encoding="utf-8")
        with pytest.raises(DataError, match="could not read"):
            cache.read("BTC/USDT", "4h")

    def test_last_ts(self, cache: ParquetCache) -> None:
        assert cache.last_ts("BTC/USDT", "4h") is None
        cache.write("BTC/USDT", "4h", frame_of(5))
        assert cache.last_ts("BTC/USDT", "4h") == START + 4 * STEP


class TestMerge:
    def test_new_rows_are_appended(self, cache: ParquetCache) -> None:
        merged = cache.merge(frame_of(5), rows_to_frame(synthetic_rows(3, start=START + 5 * STEP)))
        assert len(merged) == 8
        assert merged.index.is_monotonic_increasing

    def test_overlapping_rows_take_the_fresh_value(self, cache: ParquetCache) -> None:
        cached = frame_of(3)
        fresh = frame_of(3)
        fresh.iloc[-1, fresh.columns.get_loc("close")] = 555.0
        merged = cache.merge(cached, fresh)
        assert len(merged) == 3
        assert merged.iloc[-1]["close"] == 555.0

    def test_empty_inputs(self, cache: ParquetCache) -> None:
        assert len(cache.merge(frame_of(0), frame_of(4))) == 4
        assert len(cache.merge(frame_of(4), frame_of(0))) == 4


class TestSync:
    def test_first_run_downloads_everything(self, cache: ParquetCache) -> None:
        feed = RecordingFeed(synthetic_rows(20))
        now = START + 20 * STEP
        result = cache.sync(feed, "BTC/USDT", "4h", START, now=now)
        assert result.total == 20
        assert result.added == 20
        assert result.path.is_file()

    def test_second_run_only_requests_the_tail(self, cache: ParquetCache) -> None:
        rows = synthetic_rows(30)
        first_feed = RecordingFeed(rows[:20])
        cache.sync(first_feed, "BTC/USDT", "4h", START, now=START + 20 * STEP)

        second_feed = RecordingFeed(rows)
        result = cache.sync(second_feed, "BTC/USDT", "4h", START, now=START + 30 * STEP)

        assert result.cached_before == 20
        assert result.total == 30
        assert result.added == 10
        assert second_feed.calls[0][0] == START + 20 * STEP

    def test_no_download_when_already_current(self, cache: ParquetCache) -> None:
        rows = synthetic_rows(10)
        cache.sync(RecordingFeed(rows), "BTC/USDT", "4h", START, now=START + 10 * STEP)
        feed = RecordingFeed(rows)
        result = cache.sync(feed, "BTC/USDT", "4h", START, now=START + 10 * STEP)
        assert feed.calls == []
        assert result.added == 0

    def test_forming_candle_is_never_cached(self, cache: ParquetCache) -> None:
        feed = RecordingFeed(synthetic_rows(10))
        now = START + 9 * STEP + timedelta(minutes=30)
        result = cache.sync(feed, "BTC/USDT", "4h", START, now=now)
        assert result.end == START + 8 * STEP

    def test_explicit_end_is_respected(self, cache: ParquetCache) -> None:
        feed = RecordingFeed(synthetic_rows(20))
        result = cache.sync(
            feed, "BTC/USDT", "4h", START, end=START + 4 * STEP, now=START + 20 * STEP
        )
        assert result.total == 5
        assert result.end == START + 4 * STEP


class TestLoadAndFingerprint:
    def test_window_slicing(self, cache: ParquetCache) -> None:
        cache.write("BTC/USDT", "4h", frame_of(20))
        window = cache.load("BTC/USDT", "4h", start=START + 5 * STEP, end=START + 9 * STEP)
        assert len(window) == 5
        assert window.index[0] == START + 5 * STEP

    def test_load_missing_symbol(self, cache: ParquetCache) -> None:
        assert cache.load("ETH/USDT", "4h").empty

    def test_fingerprint_is_stable_and_content_sensitive(self, cache: ParquetCache) -> None:
        cache.write("BTC/USDT", "4h", frame_of(10))
        first = cache.fingerprint("BTC/USDT", "4h")
        assert first == cache.fingerprint("BTC/USDT", "4h")

        changed = frame_of(10)
        changed.iloc[0, changed.columns.get_loc("close")] = 1.0
        cache.write("BTC/USDT", "4h", changed)
        assert cache.fingerprint("BTC/USDT", "4h") != first

    def test_fingerprint_of_empty_cache(self, cache: ParquetCache) -> None:
        assert cache.fingerprint("BTC/USDT", "4h") == "empty"
