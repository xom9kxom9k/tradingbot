"""Wait for candle close, with a grace period for exchange lag (section 8 / FR-6.1).

Binance 4h candles sit on UTC hour boundaries. We never ask for a bar at the
exact close: ``bar_close_grace_sec`` gives the venue time to publish the final
print. If the process woke up late, :func:`last_ready_bar_open` still returns
every bar that is already fetchable, so clock drift cannot skip a close.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from tradingbot.data.feed import timeframe_delta

UNIX_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def as_utc(moment: datetime) -> datetime:
    """Normalise *moment* to timezone-aware UTC."""
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def current_bar_open(now: datetime, timeframe: str) -> datetime:
    """Open time of the candle that is still forming at ``now``."""
    delta = timeframe_delta(timeframe)
    utc = as_utc(now).replace(microsecond=0)
    step = int(delta.total_seconds())
    elapsed = int((utc - UNIX_EPOCH).total_seconds())
    aligned = elapsed - (elapsed % step)
    return UNIX_EPOCH + timedelta(seconds=aligned)


def bar_close(open_ts: datetime, timeframe: str) -> datetime:
    """Exclusive end of the candle that opened at ``open_ts``."""
    return as_utc(open_ts) + timeframe_delta(timeframe)


def next_fetch_at(now: datetime, timeframe: str, grace_sec: int) -> datetime:
    """When the next not-yet-ready close becomes safe to download.

    Inside the grace window this is ``forming_open + grace``. Once that bar is
    fetchable, it is the close of the candle that is currently forming.
    """
    forming = current_bar_open(now, timeframe)
    just_closed_ready = forming + timedelta(seconds=grace_sec)
    if as_utc(now) < just_closed_ready:
        return just_closed_ready
    return bar_close(forming, timeframe) + timedelta(seconds=grace_sec)


def seconds_until_fetch(now: datetime, timeframe: str, grace_sec: int) -> float:
    """Seconds to sleep before the next close+grace."""
    due = next_fetch_at(now, timeframe, grace_sec)
    return max(0.0, (due - as_utc(now)).total_seconds())


def last_ready_bar_open(now: datetime, timeframe: str, grace_sec: int) -> datetime:
    """Open time of the most recent candle that is past close + grace.

    At ``12:00:05`` with a 10s grace the 08:00 4h bar is closed but not yet
    fetchable, so this returns 04:00. At ``12:00:10`` it returns 08:00.
    """
    delta = timeframe_delta(timeframe)
    forming = current_bar_open(now, timeframe)
    just_closed = forming - delta
    ready_at = forming + timedelta(seconds=grace_sec)
    if as_utc(now) >= ready_at:
        return just_closed
    return just_closed - delta


def closed_opens_since(
    last_processed: datetime | None,
    now: datetime,
    timeframe: str,
    grace_sec: int,
) -> list[datetime]:
    """Every fetchable bar open strictly after ``last_processed``, oldest first.

    Used to catch up after a restart or a clock jump. An empty list means
    there is nothing new to do.
    """
    ready = last_ready_bar_open(now, timeframe, grace_sec)
    delta = timeframe_delta(timeframe)
    if last_processed is None:
        return []
    cursor = as_utc(last_processed) + delta
    opens: list[datetime] = []
    while cursor <= ready:
        opens.append(cursor)
        cursor = cursor + delta
    return opens
