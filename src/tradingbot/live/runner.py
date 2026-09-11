"""Asynchronous live loop: wait for a close, process the bar, persist (FR-6.1)."""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

import pandas as pd
from loguru import logger

from tradingbot.backtest.engine import BacktestEngine, SymbolState, _optional_float
from tradingbot.backtest.runner import build_strategy
from tradingbot.config.models import AppConfig, LiveMode
from tradingbot.core.exceptions import TradingBotError
from tradingbot.core.models import Signal
from tradingbot.data.cache import ParquetCache
from tradingbot.data.ccxt_feed import CcxtDataFeed
from tradingbot.data.feed import DataFeed
from tradingbot.live.scheduler import (
    as_utc,
    closed_opens_since,
    last_ready_bar_open,
    seconds_until_fetch,
)
from tradingbot.live.state import (
    LiveHealth,
    LiveStore,
    restore_last_exits,
    restore_pending,
    restore_portfolio,
    restore_risk,
)
from tradingbot.storage.repositories import bind_repos

Clock = Callable[[], datetime]


class RecordingEngine(BacktestEngine):
    """Backtest engine that keeps the domain :class:`Signal` objects it logs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.emitted: list[tuple[Signal, bool, str]] = []

    def _log_signal(self, signal: Signal, approved: bool, reason: str) -> None:
        super()._log_signal(signal, approved, reason)
        self.emitted.append((signal, approved, reason))


class LiveRunner:
    """Paper / signal-only loop over the configured symbols.

    The same :class:`~tradingbot.strategy.base.Strategy` instance type as the
    backtest is used. Paper mode fills through :class:`PaperBroker`; signal-only
    records intents and never opens a virtual position.
    """

    def __init__(
        self,
        config: AppConfig,
        *,
        store: LiveStore | None = None,
        feed: DataFeed | None = None,
        clock: Clock | None = None,
        history: Mapping[str, pd.DataFrame] | None = None,
    ) -> None:
        self.config = config
        self.store = store or LiveStore.from_config(config)
        self.feed = feed
        self.clock: Clock = clock or (lambda: datetime.now(tz=UTC))
        self._history = dict(history or {})
        self.strategy = build_strategy(config)
        self.engine = RecordingEngine(config, self.strategy)
        self.health = LiveHealth(mode=config.live.mode)
        self._owns_feed = feed is None
        self._trade_cursor = 0
        self._signal_cursor = 0
        self._restored = False
        self._last_exit_ts: dict[str, datetime] = {}

    @property
    def mode(self) -> LiveMode:
        """Effective live mode."""
        return self.config.live.mode

    @property
    def paper(self) -> bool:
        """True when virtual fills and positions are tracked."""
        return self.mode == "paper"

    def now(self) -> datetime:
        """Clock used by the scheduler; injectable for tests."""
        return self.clock()

    def restore(self) -> None:
        """Rebuild engine book from SQLite. Safe to call more than once."""
        restore_portfolio(self.config, self.store, self.engine.portfolio)
        restore_risk(self.store, self.engine.risk)
        pending = restore_pending(self.store)
        last_exits = restore_last_exits(self.store)
        symbols = list(self.config.exchange.symbols)
        self.engine._states = {symbol: SymbolState() for symbol in symbols}
        for symbol, order in pending.items():
            state = self.engine._states.setdefault(symbol, SymbolState())
            state.pending = order
        self._last_exit_ts = last_exits
        self._trade_cursor = len(self.engine.portfolio.trades)
        self._signal_cursor = len(self.engine.emitted)
        previous = self.store.read_status().health
        self.health.bars_processed = previous.bars_processed
        self.health.last_error = previous.last_error
        self._restored = True
        logger.info(
            "live state restored: {n} open positions, {pending} pending orders",
            n=len(self.engine.portfolio.open_trades),
            pending=sum(1 for state in self.engine._states.values() if state.pending),
        )

    def arm_cursors(self) -> None:
        """On first boot, skip history and wait for the next close (no flood of signals)."""
        timeframe = self.config.exchange.timeframe
        grace = self.config.live.bar_close_grace_sec
        ready = last_ready_bar_open(self.now(), timeframe, grace)
        with self.store.db.session_scope() as session:
            state = bind_repos(session).state
            for symbol in self.config.exchange.symbols:
                if state.last_processed_bar(symbol) is None:
                    state.set_last_processed_bar(symbol, ready)
                    logger.info(
                        "{symbol}: first start, cursor armed at {ts} (waiting for the next bar)",
                        symbol=symbol,
                        ts=ready,
                    )

    def tick(self, history: Mapping[str, pd.DataFrame] | None = None) -> int:
        """Process every fetchable bar that is newer than the stored cursor.

        Returns:
            Number of (symbol, bar) pairs handled.
        """
        if not self._restored:
            self.restore()
        frames = self._resolve_history(history)
        timeframe = self.config.exchange.timeframe
        grace = self.config.live.bar_close_grace_sec
        now = self.now()
        handled = 0
        paused = self._is_paused()
        for symbol in self.config.exchange.symbols:
            frame = frames.get(symbol)
            if frame is None or frame.empty:
                continue
            last = self._last_processed(symbol)
            due = closed_opens_since(last, now, timeframe, grace)
            prepared = self.strategy.prepare(frame)
            warmup = self.strategy.warmup_period()
            for open_ts in due:
                index = _index_of(prepared.index, open_ts)
                if index is None:
                    logger.warning(
                        "{symbol}: closed bar {ts} not in history, skipping",
                        symbol=symbol,
                        ts=open_ts,
                    )
                    self._advance_cursor(symbol, open_ts)
                    continue
                self._bind_exit_index(symbol, prepared)
                self._process_one(symbol, prepared, index, warmup, paused=paused)
                handled += 1
        if handled:
            self._snapshot_runtime()
            self.health.last_cycle_at = now
            self.health.bars_processed += handled
            self.store.save_health(self.health)
        return handled

    async def run(self, stop: asyncio.Event | None = None) -> None:
        """Block until cancelled: wait for close+grace, then :meth:`tick`."""
        stop = stop or asyncio.Event()
        self._install_signals(stop)
        self.restore()
        self.arm_cursors()
        self.health.running = True
        self.health.started_at = self.now()
        self.health.pid = os.getpid()
        self.health.mode = self.mode
        self.store.save_health(self.health)
        with self.store.db.session_scope() as session:
            bind_repos(session).state.set_started_at(self.health.started_at)
        logger.info(
            "live runner started mode={mode} symbols={symbols} timeframe={tf}",
            mode=self.mode,
            symbols=",".join(self.config.exchange.symbols),
            tf=self.config.exchange.timeframe,
        )
        try:
            while not stop.is_set():
                try:
                    n = self.tick()
                    if n:
                        logger.info("processed {n} closed bars", n=n)
                except TradingBotError as exc:
                    logger.error("live tick failed: {exc}", exc=exc)
                    self.health.last_error = str(exc)
                    self.store.save_health(self.health)
                except Exception as exc:
                    logger.exception("unexpected live tick error")
                    self.health.last_error = str(exc)
                    self.store.save_health(self.health)
                delay = seconds_until_fetch(
                    self.now(),
                    self.config.exchange.timeframe,
                    self.config.live.bar_close_grace_sec,
                )
                wait = (
                    min(delay, float(self.config.live.poll_interval_sec))
                    if delay > 0
                    else float(self.config.live.poll_interval_sec)
                )
                if delay > 0:
                    logger.info("waiting {delay:.1f}s for the next bar close", delay=delay)
                if await self._sleep(wait, stop):
                    break
        finally:
            self.health.running = False
            self.store.save_health(self.health)
            self._snapshot_runtime()
            if self._owns_feed and self.feed is not None:
                self.feed.close()
            logger.info("live runner stopped")

    def _process_one(
        self,
        symbol: str,
        frame: pd.DataFrame,
        index: int,
        warmup: int,
        *,
        paused: bool,
    ) -> None:
        moment = as_utc(frame.index[index].to_pydatetime())
        state = self.engine._states.setdefault(symbol, SymbolState())
        row = frame.iloc[index]
        atr_value = _optional_float(row.get("atr")) if self.paper else None
        marks = {
            sym: rec.position.entry_price for sym, rec in self.engine.portfolio.open_trades.items()
        }
        before_trades = len(self.engine.portfolio.trades)
        before_signals = len(self.engine.emitted)

        if self.paper:
            self.engine._fill_pending(symbol, state, row, moment, index, atr_value)
            self.engine._check_intrabar_exits(symbol, row, moment, index, atr_value)
            self.engine._accrue_funding(symbol, row, moment, frame, index)
            marks[symbol] = float(row["close"])
            equity = self.engine.portfolio.equity(marks)
            self.engine.risk.update_equity(moment, equity)
            consult = (not paused) and index >= warmup
            if consult:
                try:
                    self.engine._consult_strategy(symbol, frame, index, moment, state, equity)
                except Exception as exc:
                    logger.error(
                        "{symbol} strategy error on {ts}: {exc}",
                        symbol=symbol,
                        ts=moment,
                        exc=exc,
                    )
                    self.health.last_error = str(exc)
            self.engine.portfolio.mark(moment, marks)
        else:
            marks[symbol] = float(row["close"])
            if (not paused) and index >= warmup:
                try:
                    self._signal_only(symbol, frame, index, moment, state)
                except Exception as exc:
                    logger.error(
                        "{symbol} strategy error on {ts}: {exc}",
                        symbol=symbol,
                        ts=moment,
                        exc=exc,
                    )
                    self.health.last_error = str(exc)

        new_trades = self.engine.portfolio.trades[before_trades:]
        new_signals = self.engine.emitted[before_signals:]
        if new_trades and symbol not in self.engine.portfolio.open_trades:
            self._last_exit_ts[symbol] = moment
        self.store.persist_bar(
            symbol=symbol,
            bar_ts=moment,
            portfolio=self.engine.portfolio,
            marks=marks,
            new_signals=new_signals,
            new_trades=new_trades,
        )
        self._signal_cursor = len(self.engine.emitted)
        self._trade_cursor = len(self.engine.portfolio.trades)
        self._snapshot_runtime()

    def _signal_only(
        self,
        symbol: str,
        frame: pd.DataFrame,
        index: int,
        moment: datetime,
        state: SymbolState,
    ) -> None:
        """Record strategy intents without filling them."""
        from tradingbot.core.models import BarContext

        ctx = BarContext(
            symbol=symbol,
            timeframe=self.config.exchange.timeframe,
            index=index,
            history=frame.iloc[: index + 1],
            equity=self.config.backtest.initial_capital,
            position=None,
            bars_since_exit=_bars_since(frame, index, self._last_exit_ts.get(symbol)),
        )
        signal = self.strategy.on_bar(ctx)
        if signal is None:
            return
        self.engine._log_signal(signal, True, "")
        state.pending = None

    def _resolve_history(
        self, override: Mapping[str, pd.DataFrame] | None
    ) -> dict[str, pd.DataFrame]:
        if override is not None:
            return dict(override)
        if self._history:
            return dict(self._history)
        return self._load_from_cache()

    def _load_from_cache(self) -> dict[str, pd.DataFrame]:
        cache = ParquetCache(self.config.data.cache_dir, self.config.exchange.name)
        timeframe = self.config.exchange.timeframe
        feed = self.feed
        if feed is None:
            feed = CcxtDataFeed(self.config.exchange.name)
            self.feed = feed
        frames: dict[str, pd.DataFrame] = {}
        for symbol in self.config.exchange.symbols:
            try:
                cache.sync(feed, symbol, timeframe, self.config.data.start, self.config.data.end)
            except TradingBotError as exc:
                logger.warning("{symbol}: cache sync failed: {exc}", symbol=symbol, exc=exc)
            frame = cache.load(symbol, timeframe, self.config.data.start, self.config.data.end)
            if not frame.empty:
                frames[symbol] = frame
        return frames

    def _last_processed(self, symbol: str) -> datetime | None:
        with self.store.db.session_scope() as session:
            raw = bind_repos(session).state.last_processed_bar(symbol)
        return as_utc(raw) if raw is not None else None

    def _advance_cursor(self, symbol: str, ts: datetime) -> None:
        with self.store.db.session_scope() as session:
            bind_repos(session).state.set_last_processed_bar(symbol, ts)

    def _is_paused(self) -> bool:
        with self.store.db.session_scope() as session:
            return bind_repos(session).state.is_paused()

    def _bind_exit_index(self, symbol: str, frame: pd.DataFrame) -> None:
        ts = self._last_exit_ts.get(symbol)
        state = self.engine._states.setdefault(symbol, SymbolState())
        loc = _index_of(frame.index, ts) if ts is not None else None
        if loc is not None:
            state.last_exit_bar = loc

    def _snapshot_runtime(self) -> None:
        pending = {
            symbol: state.pending
            for symbol, state in self.engine._states.items()
            if state.pending is not None
        }
        self.store.save_runtime(
            portfolio=self.engine.portfolio,
            pending=pending,
            last_exit_ts=self._last_exit_ts,
            risk=self.engine.risk,
        )

    def _install_signals(self, stop: asyncio.Event) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                continue

    async def _sleep(self, seconds: float, stop: asyncio.Event) -> bool:
        """Sleep ``seconds``, returning True if *stop* was set."""
        if seconds <= 0:
            return stop.is_set()
        try:
            await asyncio.wait_for(stop.wait(), timeout=seconds)
            return True
        except TimeoutError:
            return stop.is_set()


def _index_of(index: pd.Index, ts: datetime) -> int | None:
    indexer = index.get_indexer(pd.DatetimeIndex([pd.Timestamp(ts)]))
    loc = int(indexer[0])
    return loc if loc >= 0 else None


def _bars_since(frame: pd.DataFrame, index: int, last_exit: datetime | None) -> int | None:
    if last_exit is None:
        return None
    exit_i = _index_of(frame.index, last_exit)
    if exit_i is None:
        return None
    return index - exit_i
