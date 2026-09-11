"""Telegram formatters, safety wrapper, queue and command handlers."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.unit.test_backtest_runner import make_config
from tests.unit.test_storage import TS, make_signal, make_trade
from tradingbot.config.models import LiveConfig, TelegramSecrets
from tradingbot.core.enums import ExitReason
from tradingbot.core.exceptions import NotificationError
from tradingbot.core.models import Signal, TakeProfit, Trade
from tradingbot.live.state import LiveStore
from tradingbot.notify.base import NullNotifier, SafeNotifier
from tradingbot.notify.bot import allowed_chat
from tradingbot.notify.commands import CommandService, parse_last_arg, parse_stats_arg
from tradingbot.notify.formatters import (
    format_daily_report,
    format_signal,
    format_system,
    format_test_message,
    format_trade,
)
from tradingbot.notify.queue import NotificationDispatcher, seconds_until_daily
from tradingbot.notify.telegram import RateLimiter, TelegramNotifier, build_notifier
from tradingbot.storage.repositories import bind_repos

EXAMPLE = make_signal(
    reason="пробой Donchian-20 вверх",
    indicators={"adx": 27.4, "atr": 1520.0, "ema_fast": 64_000.0, "ema_slow": 61_000.0},
    take_profits=[TakeProfit(price=67_550.0, r_multiple=2.0, fraction=0.5)],
)


class RecordingNotifier(NullNotifier):
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.signals: list[Signal] = []
        self.trades: list[Trade] = []
        self.photos: list[tuple[bytes, str]] = []

    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        self.signals.append(signal)
        await super().send_signal(signal, rejected=rejected, reason=reason)
        self.texts.append(format_signal(signal, rejected=rejected, reason=reason))

    async def send_trade_closed(
        self,
        trade: Trade,
        *,
        equity: float | None = None,
        initial_capital: float | None = None,
        timeframe: str = "4h",
        signal_uid: str = "",
    ) -> None:
        self.trades.append(trade)
        self.texts.append(
            format_trade(
                trade,
                equity=equity,
                initial_capital=initial_capital,
                timeframe=timeframe,
                signal_uid=signal_uid,
            )
        )

    async def send_text(self, text: str) -> None:
        self.texts.append(text)

    async def send_photo(self, image: bytes, caption: str) -> None:
        self.photos.append((image, caption))


class BoomNotifier(NullNotifier):
    async def send_text(self, text: str) -> None:
        raise RuntimeError("telegram down")

    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        raise RuntimeError("telegram down")

    async def send_trade_closed(self, trade: Trade, **kwargs: Any) -> None:
        raise RuntimeError("telegram down")

    async def send_photo(self, image: bytes, caption: str) -> None:
        raise RuntimeError("telegram down")


class TestFormatters:
    def test_entry_matches_the_spec_shape(self) -> None:
        text = format_signal(EXAMPLE, equity=10_000.0)
        assert "🟢 <b>LONG</b> — BTC/USDT (4h)" in text
        assert "63 250.00" in text
        assert "61 100.00" in text
        assert "−3.40%" in text
        assert "67 550.00" in text
        assert "+6.80%" in text
        assert "0.1450 BTC" in text
        assert "9 171" in text
        assert "1.00% капитала" in text
        assert "100.00 USDT" in text
        assert "пробой Donchian-20 вверх" in text
        assert "UP (EMA50 > EMA200)" in text
        assert "ADX(14): 27.4" in text
        assert "1 520.0" in text
        assert "2.40%" in text
        assert "2026-09-08 12:00 UTC" in text
        assert "SIG-20260908-1200-BTCUSDT-L" in text

    def test_exit_trade_matches_the_spec_shape(self) -> None:
        trade = make_trade(
            exit_price=67_550.0,
            pnl_net=623.50,
            pnl_r=2.0,
            pnl_pct=0.068,
            bars_held=9,
            mae=-0.35,
            mfe=2.41,
            exit_reason=ExitReason.TP1,
        )
        text = format_trade(
            trade,
            equity=10_623.50,
            initial_capital=10_000.0,
            signal_uid="SIG-20260908-1200-BTCUSDT-L",
        )
        assert "🔵 <b>ВЫХОД LONG</b>" in text
        assert "67 550.00" in text
        assert "+2.00R" in text
        assert "+6.80%" in text
        assert "+623.50 USDT" in text
        assert "−0.35R / +2.41R" in text
        assert "9 баров (36 ч)" in text
        assert "Take Profit 1" in text
        assert "10 623.50 USDT" in text
        assert "всего" in text

    def test_daily_and_system_templates(self) -> None:
        daily = format_daily_report(
            when=TS,
            day_pnl=-12.5,
            week_pnl=40.0,
            month_pnl=100.0,
            trades_today=2,
            open_positions=[],
            equity=10_000.0,
            drawdown_pct=0.05,
        )
        assert "Ежедневный отчёт" in daily
        assert "−12.50 USDT" in daily
        started = format_system("started", mode="paper", symbols="BTC/USDT", timeframe="4h")
        assert "Бот запущен" in started
        assert "paper" in started
        assert format_test_message().startswith("✅")

    def test_rejected_signal_mentions_the_reason(self) -> None:
        text = format_signal(EXAMPLE, rejected=True, reason="portfolio heat")
        assert "ОТКЛОНЁН" in text
        assert "portfolio heat" in text


class TestSafeNotifier:
    @pytest.mark.asyncio
    async def test_exceptions_are_swallowed(self) -> None:
        wrapped = SafeNotifier(BoomNotifier())
        await wrapped.send_text("hello")
        await wrapped.send_signal(EXAMPLE)
        assert wrapped.last_error == "telegram down"


class TestQueue:
    @pytest.mark.asyncio
    async def test_drain_marks_sent_and_is_idempotent(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        store = LiveStore.from_config(config)
        recorder = RecordingNotifier()
        dispatcher = NotificationDispatcher(store, recorder, config)
        dispatcher.enqueue(
            {"kind": "signal", "signal": EXAMPLE.model_dump(mode="json"), "signal_uid": EXAMPLE.uid}
        )
        assert await dispatcher.drain() == 1
        assert await dispatcher.drain() == 0
        assert len(recorder.signals) == 1
        with store.db.session_scope() as session:
            pending = bind_repos(session).notifications.pending()
            assert pending == []
            row = bind_repos(session).signals.get_by_uid(EXAMPLE.uid)
            # signal row is only created by persist_bar, not by enqueue alone
            assert row is None

    @pytest.mark.asyncio
    async def test_failed_delivery_stays_queued(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        store = LiveStore.from_config(config)
        dispatcher = NotificationDispatcher(store, BoomNotifier(), config)
        dispatcher.enqueue({"kind": "text", "text": "hello"})
        assert await dispatcher.drain() == 0
        with store.db.session_scope() as session:
            pending = bind_repos(session).notifications.pending()
            assert len(pending) == 1
            assert pending[0].attempts == 1

    @pytest.mark.asyncio
    async def test_rejected_signals_are_skipped_by_default(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        store = LiveStore.from_config(config)
        recorder = RecordingNotifier()
        dispatcher = NotificationDispatcher(store, recorder, config)
        dispatcher.enqueue(
            {
                "kind": "rejected_signal",
                "signal": EXAMPLE.model_dump(mode="json"),
                "reason": "heat",
            }
        )
        assert await dispatcher.drain() == 1
        assert recorder.signals == []

    def test_seconds_until_daily_waits_for_tomorrow_after_the_clock(self) -> None:
        now = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)
        delay = seconds_until_daily(now, time(9, 0))
        assert delay == pytest.approx(23 * 3600)


class TestTelegramTransport:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        secrets = TelegramSecrets(bot_token="1:abc", chat_ids=[42])  # type: ignore[arg-type]
        calls = {"n": 0}

        class FakeBot:
            async def send_message(self, chat_id: int, text: str) -> None:
                calls["n"] += 1
                if calls["n"] < 3:
                    raise RuntimeError("timeout")

            @property
            def session(self) -> Any:
                return self

            async def close(self) -> None:
                return None

        notifier = TelegramNotifier(secrets, bot=FakeBot())
        await notifier.send_text("hi")
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_exhausted_retries_raise(self) -> None:
        secrets = TelegramSecrets(bot_token="1:abc", chat_ids=[42])  # type: ignore[arg-type]

        class FakeBot:
            async def send_message(self, chat_id: int, text: str) -> None:
                raise RuntimeError("down")

        notifier = TelegramNotifier(secrets, bot=FakeBot())
        with pytest.raises(NotificationError):
            await notifier.send_text("hi")

    def test_build_notifier_is_null_when_unconfigured(self) -> None:
        assert isinstance(build_notifier(TelegramSecrets()), NullNotifier)

    @pytest.mark.asyncio
    async def test_rate_limiter_releases_a_slot(self) -> None:
        limiter = RateLimiter(max_per_minute=2)
        await limiter.acquire(1)
        await limiter.acquire(1)


class TestCommands:
    def test_pause_and_resume(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        store = LiveStore.from_config(config)
        service = CommandService(config, store)
        assert "приостановлена" in service.pause()
        with store.db.session_scope() as session:
            assert bind_repos(session).state.is_paused() is True
        assert "возобновлена" in service.resume()
        with store.db.session_scope() as session:
            assert bind_repos(session).state.is_paused() is False

    def test_help_lists_every_command(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        text = CommandService(config, LiveStore.from_config(config)).help()
        for name in (
            "/status",
            "/positions",
            "/equity",
            "/stats",
            "/last",
            "/chart",
            "/backtest",
            "/pause",
            "/resume",
            "/config",
            "/help",
        ):
            assert name in text

    def test_unknown_and_parsers(self) -> None:
        assert parse_stats_arg("30d") == "30d"
        assert parse_stats_arg("all") == "all"
        assert parse_last_arg("3") == 3
        assert parse_last_arg("nope") == 5

    def test_whitelist(self) -> None:
        secrets = TelegramSecrets(bot_token="1:a", chat_ids=[111])  # type: ignore[arg-type]
        assert allowed_chat(111, secrets) is True
        assert allowed_chat(222, secrets) is False


class TestLiveDoesNotCrash:
    @pytest.mark.asyncio
    async def test_tick_survives_a_dead_telegram(self, tmp_path: Path) -> None:
        from tests.unit.test_backtest_runner import SYMBOL, trending_frame
        from tradingbot.live.runner import LiveRunner
        from tradingbot.live.scheduler import bar_close

        frame = trending_frame()
        config = make_config(tmp_path, live=LiveConfig(state_db=tmp_path / "state.db"))
        store = LiveStore.from_config(config)
        runner = LiveRunner(
            config,
            store=store,
            notifier=BoomNotifier(),
            history={SYMBOL: frame},
            clock=lambda: bar_close(frame.index[-1].to_pydatetime(), "4h") + timedelta(seconds=10),
        )
        with store.db.session_scope() as session:
            bind_repos(session).state.set_last_processed_bar(
                SYMBOL, frame.index[runner.strategy.warmup_period() - 1].to_pydatetime()
            )
        handled = runner.tick()
        assert handled >= 0
        dispatcher = NotificationDispatcher(store, BoomNotifier(), config)
        sent = await dispatcher.drain()
        assert sent == 0
