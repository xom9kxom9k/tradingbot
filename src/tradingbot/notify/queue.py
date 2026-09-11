"""Persistent outbound queue: drain SQLite, retry, never crash the live loop."""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from typing import Any

from loguru import logger

from tradingbot.config.models import AppConfig
from tradingbot.core.models import Signal, Trade
from tradingbot.live.state import LiveStore
from tradingbot.notify.base import Notifier
from tradingbot.notify.formatters import format_system
from tradingbot.storage.repositories import bind_repos

KEY_LAST_DAILY_REPORT = "last_daily_report_date"


class NotificationDispatcher:
    """Turn queued payloads into Telegram messages and mark them sent."""

    def __init__(self, store: LiveStore, notifier: Notifier, config: AppConfig) -> None:
        self.store = store
        self.notifier = notifier
        self.config = config

    def enqueue(self, payload: dict[str, Any]) -> None:
        """Append a payload. Safe to call from the trading thread."""
        with self.store.db.session_scope() as session:
            bind_repos(session).notifications.enqueue(dict(payload))

    def enqueue_system(self, event: str, **fields: Any) -> None:
        """Queue a start/stop/halt/feed notice."""
        payload: dict[str, Any] = {"kind": "system", "event": event, **fields}
        self.enqueue(payload)

    def enqueue_daily_report(self) -> None:
        """Queue today's digest if it has not been sent yet."""
        today = datetime.now(tz=UTC).date().isoformat()
        with self.store.db.session_scope() as session:
            state = bind_repos(session).state
            if state.get(KEY_LAST_DAILY_REPORT) == today:
                return
            bind_repos(session).notifications.enqueue({"kind": "daily_report"})
            state.set(KEY_LAST_DAILY_REPORT, today)

    async def drain(self, *, limit: int = 50) -> int:
        """Deliver pending rows. Failed rows stay queued with an incremented attempt count.

        Returns:
            Number of rows marked sent.
        """
        with self.store.db.session_scope() as session:
            pending = bind_repos(session).notifications.pending(limit=limit)
            rows = [(row.id, dict(row.payload or {})) for row in pending]
        sent = 0
        for row_id, payload in rows:
            try:
                await self._deliver(payload)
            except Exception as exc:
                logger.warning("notification {id} delivery failed: {exc}", id=row_id, exc=exc)
                with self.store.db.session_scope() as session:
                    bind_repos(session).notifications.record_attempt(row_id, str(exc))
                continue
            with self.store.db.session_scope() as session:
                repos = bind_repos(session)
                repos.notifications.mark_sent(row_id)
                uid = payload.get("signal_uid")
                if isinstance(uid, str) and uid and payload.get("kind") == "signal":
                    repos.signals.mark_notified(uid)
            sent += 1
        return sent

    async def _deliver(self, payload: dict[str, Any]) -> None:
        kind = str(payload.get("kind") or "text")
        if kind == "signal":
            signal = Signal.model_validate(payload.get("signal") or payload)
            await self.notifier.send_signal(signal)
            return
        if kind == "rejected_signal":
            if not self.config.notify.notify_rejected_signals:
                return
            signal = Signal.model_validate(payload.get("signal") or payload)
            await self.notifier.send_signal(
                signal, rejected=True, reason=str(payload.get("reason") or "")
            )
            return
        if kind == "trade":
            trade = Trade.model_validate(payload["trade"])
            await self.notifier.send_trade_closed(
                trade,
                equity=_optional_float(payload.get("equity")),
                initial_capital=_optional_float(payload.get("initial_capital")),
                timeframe=str(payload.get("timeframe") or self.config.exchange.timeframe),
                signal_uid=str(payload.get("signal_uid") or ""),
            )
            return
        if kind == "text":
            await self.notifier.send_text(str(payload.get("text") or ""))
            return
        if kind == "photo":
            image = payload.get("image")
            if not isinstance(image, (bytes, bytearray)):
                return
            await self.notifier.send_photo(bytes(image), str(payload.get("caption") or ""))
            return
        if kind == "system":
            await self.notifier.send_text(format_system(str(payload.get("event") or ""), **payload))
            return
        if kind == "daily_report":
            text, png = self._build_daily()
            if png:
                try:
                    await self.notifier.send_photo(png, text)
                    return
                except Exception as exc:
                    logger.warning("daily report PNG failed, sending text: {exc}", exc=exc)
            await self.notifier.send_text(text)
            return
        await self.notifier.send_text(str(payload))

    def _build_daily(self) -> tuple[str, bytes | None]:
        from tradingbot.notify.commands import CommandService

        service = CommandService(self.config, self.store)
        text = service.daily_report_text()
        png: bytes | None = None
        try:
            png = service.equity_png()
        except Exception as exc:
            logger.warning("daily equity PNG skipped: {exc}", exc=exc)
        return text, png


def seconds_until_daily(now: datetime, report: time) -> float:
    """Seconds until the next ``report`` clock time in UTC."""
    utc = now.astimezone(UTC) if now.tzinfo else now.replace(tzinfo=UTC)
    target = datetime.combine(utc.date(), report, tzinfo=UTC)
    if utc >= target:
        target += timedelta(days=1)
    return (target - utc).total_seconds()


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
