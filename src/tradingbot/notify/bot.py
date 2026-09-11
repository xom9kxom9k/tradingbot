"""aiogram command router with a chat_id whitelist (tech.md 11.3)."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from loguru import logger

from tradingbot.config.models import AppConfig, TelegramSecrets
from tradingbot.live.state import LiveStore
from tradingbot.notify.commands import (
    CommandService,
    parse_last_arg,
    parse_stats_arg,
)


def allowed_chat(chat_id: int, secrets: TelegramSecrets) -> bool:
    """True when ``chat_id`` is on the whitelist."""
    return chat_id in set(secrets.chat_ids)


def build_dispatcher(config: AppConfig, store: LiveStore, secrets: TelegramSecrets) -> Any:
    """aiogram Dispatcher wired to :class:`CommandService`."""
    from aiogram import Dispatcher, F, Router
    from aiogram.filters import Command, CommandObject
    from aiogram.types import BufferedInputFile, Message

    service = CommandService(config, store)
    router = Router()

    def _authorized(message: Message) -> bool:
        chat = message.chat
        return chat is not None and allowed_chat(chat.id, secrets)

    @router.message(Command("start"))
    async def cmd_start(message: Message) -> None:
        if not _authorized(message):
            await message.answer("Нет доступа. Ваш chat_id не в TELEGRAM_CHAT_IDS.")
            return
        await message.answer(service.start())

    @router.message(Command("help"))
    async def cmd_help(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(service.help())

    @router.message(Command("status"))
    async def cmd_status(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(f"<pre>{service.status()}</pre>")

    @router.message(Command("positions"))
    async def cmd_positions(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(service.positions())

    @router.message(Command("equity"))
    async def cmd_equity(message: Message) -> None:
        if not _authorized(message):
            return
        caption = service.equity_caption()
        png = service.equity_png()
        if png:
            await message.answer_photo(
                BufferedInputFile(png, filename="equity.png"), caption=caption
            )
            return
        await message.answer(caption)

    @router.message(Command("stats"))
    async def cmd_stats(message: Message, command: CommandObject) -> None:
        if not _authorized(message):
            return
        await message.answer(service.stats(parse_stats_arg(command.args)))

    @router.message(Command("last"))
    async def cmd_last(message: Message, command: CommandObject) -> None:
        if not _authorized(message):
            return
        await message.answer(service.last_signals(parse_last_arg(command.args)))

    @router.message(Command("chart"))
    async def cmd_chart(message: Message, command: CommandObject) -> None:
        if not _authorized(message):
            return
        symbol = service.resolve_symbol(command.args)
        if symbol is None:
            await message.answer("Укажите символ, например /chart BTC/USDT")
            return
        png = service.chart_png(symbol)
        if png is None:
            await message.answer(f"Нет данных для {symbol}. Сначала `tradingbot data download`.")
            return
        await message.answer_photo(
            BufferedInputFile(png, filename=f"{symbol.replace('/', '')}.png"),
            caption=f"{symbol} {config.exchange.timeframe}",
        )

    @router.message(Command("backtest"))
    async def cmd_backtest(message: Message) -> None:
        if not _authorized(message):
            return
        text, png = service.backtest_summary()
        if png:
            await message.answer_photo(
                BufferedInputFile(png, filename="backtest.png"), caption=text
            )
            return
        await message.answer(text)

    @router.message(Command("pause"))
    async def cmd_pause(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(service.pause())

    @router.message(Command("resume"))
    async def cmd_resume(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(service.resume())

    @router.message(Command("config"))
    async def cmd_config(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(f"<pre>{service.config_dump()}</pre>")

    @router.message(F.text.startswith("/"))
    async def cmd_unknown(message: Message) -> None:
        if not _authorized(message):
            return
        await message.answer(service.unknown())

    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    return dispatcher


async def run_bot(
    config: AppConfig,
    store: LiveStore,
    secrets: TelegramSecrets,
    stop: Any,
) -> None:
    """Poll Telegram until ``stop`` is set. Never raises into the caller."""
    if not secrets.configured or not secrets.enabled or not config.notify.telegram_enabled:
        return
    if stop.is_set():
        return
    from aiogram import Bot
    from aiogram.client.default import DefaultBotProperties
    from aiogram.enums import ParseMode

    token = secrets.bot_token
    assert token is not None
    bot = Bot(
        token=token.get_secret_value(),
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dispatcher = build_dispatcher(config, store, secrets)
    task: asyncio.Task[Any] | None = None
    try:
        task = asyncio.create_task(dispatcher.start_polling(bot, handle_signals=False))
        await stop.wait()
        await dispatcher.stop_polling()
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
    except Exception:
        logger.exception("telegram polling failed")
    finally:
        await bot.session.close()
