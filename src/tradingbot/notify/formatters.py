"""HTML message templates for Telegram (tech.md section 11.2)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from tradingbot.config.models import TIMEFRAME_MINUTES
from tradingbot.core.enums import ExitReason, Side, SignalType
from tradingbot.core.models import Position, Signal, Trade

MINUS = "−"

EXIT_LABELS: dict[ExitReason, str] = {
    ExitReason.STOP: "Стоп",
    ExitReason.TP1: "Take Profit 1",
    ExitReason.TRAIL: "Trailing stop",
    ExitReason.CHANNEL_EXIT: "Выход из канала",
    ExitReason.REGIME_FLIP: "Смена режима",
    ExitReason.EOD: "Конец данных",
    ExitReason.RISK_STOP: "Риск-стоп",
}

SYSTEM_TEMPLATES: dict[str, str] = {
    "started": "🟢 <b>Бот запущен</b>\nРежим: {mode}\nСимволы: {symbols}\nТаймфрейм: {timeframe}",
    "stopped": "⏹ <b>Бот остановлен</b>",
    "feed_lost": "🔴 <b>Нет связи с биржей</b>\n{detail}",
    "feed_restored": "🟢 <b>Связь с биржей восстановлена</b>",
    "risk_halt": "🛑 <b>Торговля остановлена риск-менеджером</b>\n{detail}",
    "exception": "⚠️ <b>Необработанное исключение</b>\n<code>{detail}</code>",
}


def format_signal(
    signal: Signal,
    *,
    equity: float | None = None,
    adx_min: float = 20.0,
    rejected: bool = False,
    reason: str = "",
) -> str:
    """Entry / exit / scale-out intent in the section 11.2 layout."""
    if rejected:
        header = f"⚠️ <b>ОТКЛОНЁН {signal.side.value}</b> — {signal.symbol} ({signal.timeframe})"
        body = format_signal(signal, equity=equity, adx_min=adx_min)
        extra = f"\n🚫 Причина: {reason}" if reason else ""
        return f"{header}\n\n{body.split(chr(10), 1)[-1]}{extra}"
    if signal.signal_type is SignalType.ENTRY:
        return _format_entry(signal, equity=equity, adx_min=adx_min)
    return _format_exit_intent(signal)


def format_trade(
    trade: Trade,
    *,
    equity: float | None = None,
    initial_capital: float | None = None,
    timeframe: str = "4h",
    signal_uid: str = "",
) -> str:
    """Filled exit matching the «Сигнал на выход» template."""
    emoji = "🔵"
    header = f"{emoji} <b>ВЫХОД {trade.side.value}</b> — {trade.symbol} ({timeframe})"
    hours = _hours(trade.bars_held, timeframe)
    ident = signal_uid or _fallback_uid(trade)
    lines = [
        header,
        "",
        f"💵 Выход:       {_num(trade.exit_price)}",
        (
            f"📈 Результат:   {_signed(trade.pnl_r, 2)}R  |  "
            f"{_signed(trade.pnl_pct * 100, 2)}%  |  {_signed(trade.pnl_net, 2)} USDT"
        ),
        f"📉 MAE / MFE:   {_signed(trade.mae, 2)}R / {_signed(trade.mfe, 2)}R",
        f"⏱ В позиции:   {trade.bars_held} баров ({hours})",
        f"🚪 Причина:     {EXIT_LABELS.get(trade.exit_reason, trade.exit_reason.value)}",
    ]
    if equity is not None:
        total = ""
        if initial_capital and initial_capital > 0:
            total = f"  ({_signed((equity / initial_capital - 1) * 100, 2)}% всего)"
        lines.append("")
        lines.append(f"💼 Капитал: {_num(equity)} USDT{total}")
    lines.append(f"🆔 {ident}")
    return "\n".join(lines)


def format_daily_report(
    *,
    when: datetime,
    day_pnl: float,
    week_pnl: float,
    month_pnl: float,
    trades_today: int,
    open_positions: Sequence[Position],
    equity: float | None,
    drawdown_pct: float | None,
) -> str:
    """Daily digest sent at ``notify.daily_report_utc``."""
    stamp = _bar_clock(when)
    lines = [
        f"📅 <b>Ежедневный отчёт</b> — {stamp}",
        "",
        f"Сделок за сутки: {trades_today}",
        f"PnL дня:    {_signed(day_pnl, 2)} USDT",
        f"PnL недели: {_signed(week_pnl, 2)} USDT",
        f"PnL месяца: {_signed(month_pnl, 2)} USDT",
    ]
    if equity is not None:
        dd = "" if drawdown_pct is None else f"  (просадка {_num(drawdown_pct * 100, 1)}%)"
        lines.append(f"Капитал:    {_num(equity)} USDT{dd}")
    lines.append("")
    if open_positions:
        lines.append("Открытые позиции:")
        for position in open_positions:
            lines.append(
                f"  • {position.symbol} {position.side.value} "
                f"{_num(position.size, 4)} @ {_num(position.entry_price)} "
                f"стоп {_num(position.current_stop)}"
            )
    else:
        lines.append("Открытых позиций нет.")
    return "\n".join(lines)


def format_system(event: str, **fields: Any) -> str:
    """Start/stop, feed, risk-halt and crash notices."""
    template = SYSTEM_TEMPLATES.get(event)
    if template is None:
        detail = str(fields.get("detail") or event)
        return f"ℹ️ {detail}"
    payload = {
        "detail": _html(str(fields.get("detail") or "")),
        "mode": fields.get("mode") or "",
        "symbols": fields.get("symbols") or "",
        "timeframe": fields.get("timeframe") or "",
    }
    return template.format(**payload)


def format_test_message() -> str:
    """Payload of ``tradingbot telegram test``."""
    return "✅ <b>tradingbot</b>: тестовое сообщение. Связь с Telegram работает."


def _format_entry(signal: Signal, *, equity: float | None, adx_min: float) -> str:
    emoji = "🟢" if signal.side is Side.LONG else "🔴"
    stop_pct = (signal.stop_loss - signal.price) / signal.price * 100
    base, quote = _pair(signal.symbol)
    notional = signal.size * signal.price
    capital = equity if equity and equity > 0 else _implied_equity(signal)
    risk_usdt = capital * signal.risk_pct if capital is not None else signal.size * signal.r_value
    lines = [
        f"{emoji} <b>{signal.side.value}</b> — {signal.symbol} ({signal.timeframe})",
        "",
        f"💵 Вход:        {_num(signal.price)}",
        f"🛑 Стоп:        {_num(signal.stop_loss)}  ({_signed(stop_pct, 2)}% / 1R)",
    ]
    for index, level in enumerate(signal.take_profits, start=1):
        tp_pct = (level.price - signal.price) / signal.price * 100
        lines.append(
            f"🎯 TP{index}:         {_num(level.price)}  "
            f"({_signed(tp_pct, 2)}% / {level.r_multiple:.0f}R, {level.fraction:.0%})"
        )
    lines.extend(
        [
            f"📦 Объём:       {_num(signal.size, 4)} {base}  (≈ {_num(notional, 0)} {quote})",
            f"⚠️ Риск:        {signal.risk_pct:.2%} капитала  ({_num(risk_usdt)} {quote})",
            "",
            f"📊 Причина: {signal.reason}",
            *_indicator_lines(signal, adx_min=adx_min),
            "",
            f"🕐 Бар: {_bar_clock(signal.bar_ts)} (закрыт)",
            f"🆔 {signal.display_id}",
        ]
    )
    return "\n".join(lines)


def _format_exit_intent(signal: Signal) -> str:
    emoji = "🔵"
    verb = "ВЫХОД" if signal.signal_type is SignalType.EXIT else "ЧАСТИЧНЫЙ ВЫХОД"
    reason = (
        EXIT_LABELS.get(signal.exit_reason, signal.reason)
        if signal.exit_reason is not None
        else signal.reason
    )
    return "\n".join(
        [
            f"{emoji} <b>{verb} {signal.side.value}</b> — {signal.symbol} ({signal.timeframe})",
            "",
            f"💵 Цена:        {_num(signal.price)}",
            f"📦 Объём:       {_num(signal.size, 4)}",
            f"🚪 Причина:     {reason}",
            "",
            f"🕐 Бар: {_bar_clock(signal.bar_ts)} (закрыт)",
            f"🆔 {signal.display_id}",
        ]
    )


def _indicator_lines(signal: Signal, *, adx_min: float) -> list[str]:
    values = signal.indicators
    lines: list[str] = []
    regime = _regime_label(values)
    if regime:
        lines.append(f"   • Режим: {regime}")
    adx = values.get("adx")
    if adx is not None:
        lines.append(f"   • ADX(14): {_num(adx, 1)}  ≥ {adx_min:.0f}")
    atr = values.get("atr")
    if atr is not None:
        atr_pct = values.get("atr_pct")
        if atr_pct is None and signal.price:
            atr_pct = atr / signal.price
        extra = f"  ({atr_pct:.2%} от цены)" if atr_pct is not None else ""
        lines.append(f"   • ATR(14): {_num(atr, 1)}{extra}")
    return lines


def _regime_label(values: dict[str, float]) -> str | None:
    raw = values.get("regime")
    if raw is not None:
        return str(raw)
    fast, slow = values.get("ema_fast"), values.get("ema_slow")
    if fast is None or slow is None:
        return None
    if fast > slow:
        return "UP (EMA50 > EMA200)"
    if fast < slow:
        return "DOWN (EMA50 < EMA200)"
    return "FLAT"


def _implied_equity(signal: Signal) -> float | None:
    if signal.risk_pct <= 0:
        return None
    risked = signal.size * signal.r_value
    if risked <= 0:
        return None
    return risked / signal.risk_pct


def _pair(symbol: str) -> tuple[str, str]:
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return base, quote.split(":")[0]
    return symbol, "USDT"


def _hours(bars: int, timeframe: str) -> str:
    minutes = TIMEFRAME_MINUTES.get(timeframe, 0)
    hours = bars * minutes / 60
    if hours >= 1 and abs(hours - round(hours)) < 1e-9:
        return f"{round(hours)} ч"
    if hours >= 1:
        return f"{hours:.1f} ч"
    return f"{int(bars * minutes)} мин"


def _bar_clock(moment: datetime) -> str:
    utc = moment.astimezone(UTC) if moment.tzinfo else moment.replace(tzinfo=UTC)
    return utc.strftime("%Y-%m-%d %H:%M UTC")


def _fallback_uid(trade: Trade) -> str:
    tag = trade.symbol.replace("/", "").replace(":", "")
    return f"SIG-{trade.entry_ts:%Y%m%d-%H%M}-{tag}-{trade.side.letter}"


def _num(value: float, decimals: int = 2) -> str:
    formatted = f"{value:,.{decimals}f}"
    return formatted.replace(",", " ")


def _signed(value: float, decimals: int) -> str:
    if value > 0:
        return f"+{_num(value, decimals)}"
    if value < 0:
        return f"{MINUS}{_num(abs(value), decimals)}"
    return _num(value, decimals)


def _html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
