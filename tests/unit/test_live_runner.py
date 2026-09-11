"""Live runner: paper fills, idempotent signals and restart recovery."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tests.unit.test_backtest_runner import SYMBOL, make_config, trending_frame
from tradingbot.config.models import AppConfig, LiveConfig
from tradingbot.core.enums import SignalType
from tradingbot.live.runner import LiveRunner
from tradingbot.live.scheduler import bar_close
from tradingbot.live.state import LiveStore
from tradingbot.storage.repositories import bind_repos

GRACE = 10


def live_config(tmp_path: Path, *, mode: str = "paper") -> AppConfig:
    return make_config(
        tmp_path,
        live=LiveConfig(
            mode=mode,  # type: ignore[arg-type]
            state_db=tmp_path / "state.db",
            bar_close_grace_sec=GRACE,
            poll_interval_sec=1,
        ),
    )


def fetch_at(open_ts: datetime, timeframe: str = "4h") -> datetime:
    return bar_close(open_ts, timeframe) + timedelta(seconds=GRACE)


def seed_cursor(store: LiveStore, symbol: str, ts: datetime) -> None:
    with store.db.session_scope() as session:
        bind_repos(session).state.set_last_processed_bar(symbol, ts)


def signal_uids(store: LiveStore) -> list[str]:
    with store.db.session_scope() as session:
        return [row.signal_uid for row in bind_repos(session).signals.recent(limit=500)]


def to_utc(ts: object) -> datetime:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.to_pydatetime().astimezone(UTC)


class TestFirstBoot:
    def test_arm_cursors_skips_historical_bars(self, tmp_path: Path) -> None:
        frame = trending_frame(bars=80)
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        now = fetch_at(to_utc(frame.index[-1]))
        runner = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        runner.arm_cursors()
        assert runner.tick() == 0
        last = runner._last_processed(SYMBOL)
        assert last is not None
        with store.db.session_scope() as session:
            assert bind_repos(session).signals.count() == 0


class TestPaperLoop:
    def test_processes_closed_bars_into_sqlite(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        now = fetch_at(to_utc(frame.index[-1]))
        runner = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        handled = runner.tick()
        assert handled > 0
        status = store.read_status()
        assert status.last_processed[SYMBOL] == to_utc(frame.index[-1])
        assert status.health.bars_processed == handled
        with store.db.session_scope() as session:
            repos = bind_repos(session)
            assert repos.state.latest_equity() is not None

    def test_restart_does_not_duplicate_signals(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        now = fetch_at(to_utc(frame.index[-1]))
        first = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        first.tick()
        uids = signal_uids(store)
        second = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        assert second.tick() == 0
        assert signal_uids(store) == uids

    def test_pending_order_survives_restart_and_fills(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        clock = {"now": fetch_at(to_utc(frame.index[warmup]))}
        runner = LiveRunner(
            config,
            store=store,
            clock=lambda: clock["now"],
            history={SYMBOL: frame},
        )
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        pending_at: int | None = None
        for index in range(warmup, len(frame) - 1):
            clock["now"] = fetch_at(to_utc(frame.index[index]))
            runner.tick()
            state = runner.engine._states.get(SYMBOL)
            if state is not None and state.pending is not None:
                pending_at = index
                break
        assert pending_at is not None, "synthetic series never queued a next-bar order"
        restarted = LiveRunner(
            config,
            store=store,
            clock=lambda: clock["now"],
            history={SYMBOL: frame},
        )
        restarted.restore()
        restored = restarted.engine._states[SYMBOL].pending
        assert restored is not None
        assert restored.signal.signal_type is SignalType.ENTRY
        clock["now"] = fetch_at(to_utc(frame.index[pending_at + 1]))
        restarted.tick()
        assert SYMBOL in restarted.engine.portfolio.open_trades
        status = store.read_status()
        assert any(pos.symbol == SYMBOL for pos in status.open_positions)

    def test_open_position_survives_restart(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        now = fetch_at(to_utc(frame.index[-1]))
        LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame}).tick()
        if not store.read_status().open_positions:
            pytest.skip("trending fixture closed flat; pending-order test covers restore")
        restarted = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        restarted.restore()
        assert SYMBOL in restarted.engine.portfolio.open_trades
        original = store.read_status().open_positions[0]
        restored = restarted.engine.portfolio.open_trades[SYMBOL].position
        assert restored.size == pytest.approx(original.size)
        assert restored.entry_price == pytest.approx(original.entry_price)

    def test_signal_only_does_not_open_positions(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path, mode="signal_only")
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        now = fetch_at(to_utc(frame.index[-1]))
        runner = LiveRunner(config, store=store, clock=lambda: now, history={SYMBOL: frame})
        runner.tick()
        assert runner.engine.portfolio.open_trades == {}
        assert store.read_status().open_positions == []
        assert signal_uids(store)

    def test_paused_skips_new_entries_but_manages_open_book(self, tmp_path: Path) -> None:
        frame = trending_frame()
        config = live_config(tmp_path)
        store = LiveStore.from_config(config)
        warmup = LiveRunner(config, store=store, history={SYMBOL: frame}).strategy.warmup_period()
        clock = {"now": fetch_at(to_utc(frame.index[warmup]))}
        runner = LiveRunner(
            config,
            store=store,
            clock=lambda: clock["now"],
            history={SYMBOL: frame},
        )
        seed_cursor(store, SYMBOL, to_utc(frame.index[warmup - 1]))
        for index in range(warmup, len(frame) - 2):
            clock["now"] = fetch_at(to_utc(frame.index[index]))
            runner.tick()
            if runner.engine._states.get(SYMBOL) and runner.engine._states[SYMBOL].pending:
                break
        else:
            pytest.skip("no pending entry to freeze under /pause")
        with store.db.session_scope() as session:
            bind_repos(session).state.set_paused(True)
        uids = signal_uids(store)
        last = runner._last_processed(SYMBOL)
        assert last is not None
        nxt = next(i for i, ts in enumerate(frame.index) if to_utc(ts) > last)
        clock["now"] = fetch_at(to_utc(frame.index[nxt]))
        runner.tick()
        assert signal_uids(store) == uids


@pytest.mark.asyncio
async def test_run_stops_when_event_is_set(tmp_path: Path) -> None:
    config = live_config(tmp_path)
    store = LiveStore.from_config(config)
    stop = asyncio.Event()
    stop.set()
    await LiveRunner(config, store=store, history={SYMBOL: trending_frame(bars=40)}).run(stop=stop)
    assert store.read_status().health.running is False
