"""Pure command handlers used by the Telegram bot (tech.md 11.3)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from loguru import logger

from tradingbot.config.models import AppConfig
from tradingbot.core.models import Position
from tradingbot.live.state import LiveStore, format_status
from tradingbot.notify.formatters import format_daily_report
from tradingbot.storage.models import SignalRow, TradeRow
from tradingbot.storage.repositories import bind_repos

HELP_TEXT = """\
<b>Команды tradingbot</b>
/status — uptime, режим, последний бар
/positions — открытые позиции
/equity — капитал + график
/stats [7d|30d|all] — сводка сделок
/last [n] — последние сигналы
/chart SYMBOL — свечной график
/backtest — метрики последнего прогона
/pause — не открывать новые сделки
/resume — возобновить
/config — параметры стратегии и риска
/help — эта справка
"""

START_TEXT = "Привет. Это tradingbot — paper/signal бот по Donchian Trend.\n\n" + HELP_TEXT


class CommandService:
    """Read-only (plus pause/resume) façade over the live database and config."""

    def __init__(self, config: AppConfig, store: LiveStore) -> None:
        self.config = config
        self.store = store

    def start(self) -> str:
        return START_TEXT

    def help(self) -> str:
        return HELP_TEXT

    def status(self) -> str:
        status = self.store.read_status()
        exchange = "ok"
        lines = [
            "<b>Статус</b>",
            format_status(status),
            f"symbols          {', '.join(self.config.exchange.symbols)}",
            f"timeframe        {self.config.exchange.timeframe}",
            f"exchange         {exchange}",
        ]
        return "\n".join(lines)

    def positions(self) -> str:
        status = self.store.read_status()
        if not status.open_positions:
            return "Открытых позиций нет."
        lines = ["<b>Открытые позиции</b>"]
        for position in status.open_positions:
            lines.append(_position_line(position))
        return "\n".join(lines)

    def stats(self, period: str = "7d") -> str:
        period = (period or "7d").strip().lower()
        trades = self._trades()
        if period != "all":
            days = 7 if period.startswith("7") else 30
            cutoff = datetime.now(tz=UTC) - timedelta(days=days)
            trades = [row for row in trades if row.exit_ts >= cutoff]
            label = f"{days}д"
        else:
            label = "всё время"
        if not trades:
            return f"Сделок за {label} нет."
        wins = sum(1 for row in trades if row.pnl_net > 0)
        pnl = sum(row.pnl_net for row in trades)
        avg_r = sum(row.pnl_r for row in trades) / len(trades)
        return "\n".join(
            [
                f"<b>Сводка ({label})</b>",
                f"сделок     {len(trades)}",
                f"win rate   {wins / len(trades):.0%}",
                f"PnL        {pnl:+,.2f} USDT",
                f"средний R  {avg_r:+.2f}",
            ]
        )

    def last_signals(self, count: int = 5) -> str:
        n = max(1, min(count, 20))
        with self.store.db.session_scope() as session:
            rows = bind_repos(session).signals.recent(limit=n)
        if not rows:
            return "Сигналов пока нет."
        lines = [f"<b>Последние {len(rows)} сигналов</b>"]
        for row in rows:
            lines.append(_signal_line(row))
        return "\n".join(lines)

    def config_dump(self) -> str:
        strategy = self.config.strategy.model_dump(mode="json")
        risk = self.config.risk.model_dump(mode="json")
        live = self.config.live.model_dump(mode="json")
        return "\n".join(
            [
                "<b>Конфигурация</b> (без секретов)",
                f"strategy: {strategy.get('name')}",
                *(f"  {key}: {value}" for key, value in dict(strategy.get("params") or {}).items()),
                "risk:",
                *(f"  {key}: {value}" for key, value in risk.items()),
                "live:",
                *(f"  {key}: {value}" for key, value in live.items()),
            ]
        )

    def pause(self) -> str:
        with self.store.db.session_scope() as session:
            bind_repos(session).state.set_paused(True)
        return (
            "⏸ Генерация новых сигналов приостановлена. Открытые позиции продолжают сопровождаться."
        )

    def resume(self) -> str:
        with self.store.db.session_scope() as session:
            bind_repos(session).state.set_paused(False)
        return "▶️ Генерация сигналов возобновлена."

    def unknown(self) -> str:
        return "Неизвестная команда. /help — список доступных."

    def daily_report_text(self, *, now: datetime | None = None) -> str:
        moment = now or datetime.now(tz=UTC)
        trades = self._trades()
        day = [row for row in trades if row.exit_ts.date() == moment.date()]
        week_cut = moment - timedelta(days=7)
        month_cut = moment - timedelta(days=30)
        status = self.store.read_status()
        equity = status.latest_equity
        return format_daily_report(
            when=moment,
            day_pnl=sum(row.pnl_net for row in day),
            week_pnl=sum(row.pnl_net for row in trades if row.exit_ts >= week_cut),
            month_pnl=sum(row.pnl_net for row in trades if row.exit_ts >= month_cut),
            trades_today=len(day),
            open_positions=status.open_positions,
            equity=equity.equity if equity is not None else None,
            drawdown_pct=equity.drawdown_pct if equity is not None else None,
        )

    def equity_png(self) -> bytes | None:
        from tradingbot.reporting.charts import build_charts
        from tradingbot.reporting.png_export import PngExportError, png_bytes

        frame = self._equity_frame()
        if frame.empty:
            return None
        bundle = build_charts(
            equity=frame,
            trades=pd.DataFrame(),
            initial_capital=self.config.backtest.initial_capital,
        )
        try:
            return png_bytes(bundle.figures["equity"])
        except PngExportError as exc:
            logger.warning("equity PNG: {exc}", exc=exc)
            return None

    def chart_png(self, symbol: str) -> bytes | None:
        from tradingbot.reporting.charts import build_charts
        from tradingbot.reporting.png_export import PngExportError, png_bytes

        prices = self._prices(symbol)
        if prices.empty:
            return None
        trades = self._trades_frame(symbol)
        bundle = build_charts(
            equity=self._equity_frame(),
            trades=trades,
            prices={symbol: prices},
            initial_capital=self.config.backtest.initial_capital,
            symbol=symbol,
            strategy_params=self.config.strategy.params,
        )
        try:
            return png_bytes(bundle.figures["price"])
        except PngExportError as rec:
            logger.warning("chart PNG: {exc}", exc=rec)
            return None

    def backtest_summary(self) -> tuple[str, bytes | None]:
        from tradingbot.backtest.runner import latest_run_id, load_run
        from tradingbot.core.exceptions import TradingBotError
        from tradingbot.reporting.html_report import charts_for_run
        from tradingbot.reporting.png_export import PngExportError, png_bytes

        run_id = latest_run_id(self.config.backtest.runs_dir)
        if run_id is None:
            return "Нет сохранённых бэктестов. Сначала `tradingbot backtest run`.", None
        try:
            stored = load_run(run_id, self.config.backtest.runs_dir)
        except TradingBotError as exc:
            return str(exc), None
        headline = stored.metrics or {}
        ratios = headline.get("ratios") or {}
        trading = headline.get("trading") or {}
        returns = headline.get("returns") or {}
        text = "\n".join(
            [
                f"<b>Бэктест {stored.run_id}</b>",
                f"Sharpe     {ratios.get('sharpe', '—')}",
                f"CAGR       {returns.get('cagr_pct', '—')}",
                f"Max DD     {headline.get('risk', {}).get('max_drawdown_pct', '—')}",
                f"Trades     {trading.get('trades', '—')}",
                f"Win rate   {trading.get('win_rate_pct', '—')}",
            ]
        )
        png: bytes | None = None
        try:
            png = png_bytes(charts_for_run(stored).figures["equity"])
        except (PngExportError, KeyError, TradingBotError) as exc:
            logger.warning("backtest PNG: {exc}", exc=exc)
        return text, png

    def equity_caption(self) -> str:
        status = self.store.read_status()
        row = status.latest_equity
        if row is None:
            return "Снимков капитала пока нет."
        return (
            f"<b>Капитал</b>: {row.equity:,.2f} USDT\n"
            f"баланс {row.balance:,.2f}  просадка {row.drawdown_pct:.1%}"
        )

    def resolve_symbol(self, raw: str | None) -> str | None:
        if not raw:
            return self.config.exchange.symbols[0]
        needle = raw.strip().upper().replace("-", "/")
        for symbol in self.config.exchange.symbols:
            if symbol.upper() == needle or symbol.upper().replace("/", "") == needle.replace(
                "/", ""
            ):
                return symbol
        return None

    def _trades(self) -> list[TradeRow]:
        with self.store.db.session_scope() as session:
            return bind_repos(session).trades.recent(limit=5_000)

    def _trades_frame(self, symbol: str | None = None) -> pd.DataFrame:
        rows = self._trades()
        if symbol:
            rows = [row for row in rows if row.symbol == symbol]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([row.to_trade().to_row() for row in rows])

    def _equity_frame(self) -> pd.DataFrame:
        with self.store.db.session_scope() as session:
            history = bind_repos(session).state.equity_history()
        if not history:
            return pd.DataFrame()
        frame = pd.DataFrame(
            {
                "ts": [row.ts for row in history],
                "balance": [row.balance for row in history],
                "equity": [row.equity for row in history],
                "open_positions": [row.open_positions_count for row in history],
                "drawdown_pct": [row.drawdown_pct for row in history],
            }
        )
        return frame.set_index("ts")

    def _prices(self, symbol: str) -> pd.DataFrame:
        from tradingbot.data.cache import ParquetCache

        cache = ParquetCache(self.config.data.cache_dir, self.config.exchange.name)
        frame = cache.read(symbol, self.config.exchange.timeframe)
        if frame.empty:
            return frame
        return frame.tail(180)


def parse_stats_arg(arg: str | None) -> str:
    raw = (arg or "7d").strip().lower()
    if raw in {"7", "7d", "week"}:
        return "7d"
    if raw in {"30", "30d", "month"}:
        return "30d"
    if raw in {"all", "*"}:
        return "all"
    return "7d"


def parse_last_arg(arg: str | None) -> int:
    try:
        return max(1, min(int(arg or "5"), 20))
    except ValueError:
        return 5


def _position_line(position: Position) -> str:
    return (
        f"{position.symbol} {position.side.value} "
        f"вход {_fmt(position.entry_price)}  размер {position.size:.4f}  "
        f"стоп {_fmt(position.current_stop)}"
    )


def _signal_line(row: SignalRow) -> str:
    try:
        signal = row.to_signal()
        ident = signal.display_id
    except Exception:
        ident = row.signal_uid
    return f"{row.bar_ts:%Y-%m-%d %H:%M}  {row.side} {row.signal_type} {row.symbol}  {ident}"


def _fmt(value: float) -> str:
    return f"{value:,.2f}".replace(",", " ")
