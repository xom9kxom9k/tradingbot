"""Notifier ABC and the safety wrapper that protects the trading loop (FR-6.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any

from loguru import logger

from tradingbot.core.models import Signal, Trade


class Notifier(ABC):
    """Outbound channel for signals, closed trades and free-form text."""

    @abstractmethod
    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        """Announce an entry, exit or scale-out intent."""

    @abstractmethod
    async def send_trade_closed(
        self,
        trade: Trade,
        *,
        equity: float | None = None,
        initial_capital: float | None = None,
        timeframe: str = "4h",
        signal_uid: str = "",
    ) -> None:
        """Announce a filled (partial or full) exit."""

    @abstractmethod
    async def send_text(self, text: str) -> None:
        """Send a pre-formatted HTML message."""

    @abstractmethod
    async def send_photo(self, image: bytes, caption: str) -> None:
        """Send a PNG with an HTML caption."""

    async def close(self) -> None:
        """Release transport resources. Default is a no-op."""
        return None


class NullNotifier(Notifier):
    """Drops every message. Used when Telegram is disabled or unconfigured."""

    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        return None

    async def send_trade_closed(
        self,
        trade: Trade,
        *,
        equity: float | None = None,
        initial_capital: float | None = None,
        timeframe: str = "4h",
        signal_uid: str = "",
    ) -> None:
        return None

    async def send_text(self, text: str) -> None:
        return None

    async def send_photo(self, image: bytes, caption: str) -> None:
        return None


class SafeNotifier(Notifier):
    """Forwards to an inner notifier and swallows every exception.

    The live loop must never die because Telegram is down (tech.md 11.4).
    """

    def __init__(self, inner: Notifier) -> None:
        self.inner = inner
        self.last_error: str | None = None

    async def send_signal(
        self, signal: Signal, *, rejected: bool = False, reason: str = ""
    ) -> None:
        await self._guard(self.inner.send_signal(signal, rejected=rejected, reason=reason))

    async def send_trade_closed(
        self,
        trade: Trade,
        *,
        equity: float | None = None,
        initial_capital: float | None = None,
        timeframe: str = "4h",
        signal_uid: str = "",
    ) -> None:
        await self._guard(
            self.inner.send_trade_closed(
                trade,
                equity=equity,
                initial_capital=initial_capital,
                timeframe=timeframe,
                signal_uid=signal_uid,
            )
        )

    async def send_text(self, text: str) -> None:
        await self._guard(self.inner.send_text(text))

    async def send_photo(self, image: bytes, caption: str) -> None:
        await self._guard(self.inner.send_photo(image, caption))

    async def close(self) -> None:
        await self._guard(self.inner.close())

    async def _guard(self, awaitable: Awaitable[Any]) -> None:
        try:
            await awaitable
        except Exception as exc:
            self.last_error = str(exc)
            logger.exception("notifier failed; trading loop continues")
