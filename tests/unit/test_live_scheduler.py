"""Candle-close scheduler: grace period, alignment and catch-up."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from freezegun import freeze_time

from tradingbot.live.scheduler import (
    bar_close,
    closed_opens_since,
    current_bar_open,
    last_ready_bar_open,
    next_fetch_at,
    seconds_until_fetch,
)

GRACE = 10
TF = "4h"


def ts(hour: int, minute: int = 0, second: int = 0) -> datetime:
    return datetime(2024, 1, 15, hour, minute, second, tzinfo=UTC)


class TestAlignment:
    def test_4h_bars_sit_on_utc_boundaries(self) -> None:
        assert current_bar_open(ts(12, 0, 5), TF) == ts(12)
        assert current_bar_open(ts(11, 59, 59), TF) == ts(8)
        assert bar_close(ts(8), TF) == ts(12)

    def test_daily_bars_align_to_utc_midnight(self) -> None:
        noon = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)
        assert current_bar_open(noon, "1d") == datetime(2024, 1, 15, tzinfo=UTC)


class TestGrace:
    @freeze_time("2024-01-15 12:00:05", tz_offset=0)
    def test_just_closed_bar_is_not_ready_before_grace(self) -> None:
        now = datetime.now(tz=UTC)
        assert last_ready_bar_open(now, TF, GRACE) == ts(4)
        assert seconds_until_fetch(now, TF, GRACE) == pytest.approx(5.0)

    @freeze_time("2024-01-15 12:00:10", tz_offset=0)
    def test_bar_becomes_fetchable_after_grace(self) -> None:
        now = datetime.now(tz=UTC)
        assert last_ready_bar_open(now, TF, GRACE) == ts(8)
        assert next_fetch_at(now, TF, GRACE) == ts(16, 0, 10)

    @freeze_time("2024-01-15 16:00:10", tz_offset=0)
    def test_after_grace_the_wait_is_until_the_next_close(self) -> None:
        now = datetime.now(tz=UTC)
        assert last_ready_bar_open(now, TF, GRACE) == ts(12)
        assert seconds_until_fetch(now, TF, GRACE) == pytest.approx(4 * 3600)


class TestCatchUp:
    def test_first_boot_does_not_replay_history(self) -> None:
        assert closed_opens_since(None, ts(12, 0, 10), TF, GRACE) == []

    def test_skipped_bars_are_returned_oldest_first(self) -> None:
        due = closed_opens_since(ts(0), ts(12, 0, 10), TF, GRACE)
        assert due == [ts(4), ts(8)]

    def test_nothing_due_when_cursor_is_current(self) -> None:
        assert closed_opens_since(ts(8), ts(12, 0, 10), TF, GRACE) == []

    def test_naive_datetimes_are_treated_as_utc(self) -> None:
        naive = datetime(2024, 1, 15, 8, 0)
        due = closed_opens_since(naive, ts(16, 0, 10), TF, GRACE)
        assert due == [ts(12)]

    def test_one_step_after_the_cursor(self) -> None:
        due = closed_opens_since(ts(4), ts(12, 0, 10), TF, GRACE)
        assert due == [ts(8)]
        assert due[0] - ts(4) == timedelta(hours=4)
