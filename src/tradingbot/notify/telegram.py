"""aiogram transport: retries, per-chat rate limit, HTML + PNG (tech.md 11.4)."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from tradingbot.config.models import TelegramSecrets
from tradingbot.core.exceptions import NotificationError
from tradingbot.core.models import Signal, Trade
from tradingbot.notify.base import Notifier
from tradingbot.notify.formatters import format_signal, format_test_message, format_trade

MAX_RETRIES = 5

SendOp = Callable[[], Awaitable[Any]]


class RateLimiter:
    """At most ``max_per_minute`` sends per chat, buffering by sleeping."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max(1, max_per_minute)
        self._hits: dict[int, deque[float]] = defaultdict(deque)

    async def acquire(self, chat_id: int) -> None:
        """Block until a slot is free for ``chat_id``."""
        while True:
            now = time.monotonic()
            window = self._hits[chat_id]
            while window and now - window[0] >= 60.0:
                window.popleft()
            if len(window) < self.max_per_minute:
                window.append(now)
                return
            wait = 60.0 - (now - window[0]) + 0.01
            await asyncio.sleep(max(wait, 0.01))


class TelegramNotifier(Notifier):
    """Sends HTML messages to every whitelisted chat_id."""

    def __init__(
        self,
        secrets: TelegramSecrets,
        *,
        max_per_minute: int = 20,
        bot: Any | None = None,
    ) -> None:
        if not secrets.configured:
            raise NotificationError("Telegram is not configured")
        self.secrets = secrets
        self.limiter = RateLimiter(max_per_minute)
        self._bot = bot
        self._owns_bot = bot is None

    def _ensure_bot(self) -> Any:
        if self._bot is not None:
            return self._bot
        from aiogram import Bot
        from aiogram.client.default import DefaultBotProperties
        from aiogram.enums import ParseMode

        token = self.secrets.bot_token
        assert token is not None
        self._bot = Bot(
            token=token.get_secret_value(),
            default=DefaultBotProperties(parse_mode=ParseMode.HTML),
        )
        return self._bot

    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        await self.send_text(format_signal(signal, rejected=rejected, reason=reason))

    async def send_trade_closed(
        self,
        trade: Trade,
        *,
        equity: float | None = None,
        initial_capital: float | None = None,
        timeframe: str = "4h",
        signal_uid: str = "",
    ) -> None:
        await self.send_text(
            format_trade(
                trade,
                equity=equity,
                initial_capital=initial_capital,
                timeframe=timeframe,
                signal_uid=signal_uid,
            )
        )

    async def send_text(self, text: str) -> None:
        bot = self._ensure_bot()
        for chat_id in self.secrets.chat_ids:
            await self.limiter.acquire(chat_id)

            async def _send(cid: int = chat_id) -> Any:
                return await bot.send_message(cid, text)

            await self._retry(_send)

    async def send_photo(self, image: bytes, caption: str) -> None:
        from aiogram.types import BufferedInputFile

        bot = self._ensure_bot()
        photo = BufferedInputFile(image, filename="chart.png")
        for chat_id in self.secrets.chat_ids:
            await self.limiter.acquire(chat_id)

            async def _send(cid: int = chat_id) -> Any:
                return await bot.send_photo(cid, photo, caption=caption)

            await self._retry(_send)

    async def send_test(self) -> None:
        """CLI ``telegram test`` helper."""
        await self.send_text(format_test_message())

    async def close(self) -> None:
        if self._owns_bot and self._bot is not None:
            await self._bot.session.close()
            self._bot = None

    async def _retry(self, operation: SendOp) -> None:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                await operation()
                return
            except Exception as exc:
                last_exc = exc
                delay = _retry_delay(exc, attempt)
                logger.warning(
                    "telegram send failed (attempt {n}/{max}): {exc}",
                    n=attempt + 1,
                    max=MAX_RETRIES,
                    exc=exc,
                )
                await asyncio.sleep(delay)
        raise NotificationError(str(last_exc) if last_exc else "telegram send failed") from last_exc


def _retry_delay(exc: Exception, attempt: int) -> float:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        try:
            return max(float(retry_after), 0.1)
        except (TypeError, ValueError):
            pass
    return min(2.0**attempt, 30.0)


def build_notifier(
    secrets: TelegramSecrets, *, max_per_minute: int = 20, enabled: bool = True
) -> Notifier:
    """Telegram when configured and enabled, otherwise :class:`NullNotifier`."""
    from tradingbot.notify.base import NullNotifier

    if not enabled or not secrets.enabled or not secrets.configured:
        return NullNotifier()
    return TelegramNotifier(secrets, max_per_minute=max_per_minute)
