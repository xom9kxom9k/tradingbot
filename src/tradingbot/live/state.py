"""Restore and snapshot the live loop's working memory (FR-6.3).

SQLite holds the durable facts: signals, positions, trades, equity, the
key/value bag. This module serialises the extra runtime the paper engine needs
on top of that — pending orders, cash, risk halt flags — so a process that dies
between two bars comes back with the same book.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingbot.backtest.engine import PendingOrder
from tradingbot.backtest.portfolio import ExitChunk, OpenTrade, Portfolio
from tradingbot.config.models import AppConfig
from tradingbot.core.enums import ExitReason, SignalStatus
from tradingbot.core.models import Position, Signal, Trade, utcnow
from tradingbot.risk.manager import RiskManager, RiskState
from tradingbot.storage.db import Database
from tradingbot.storage.models import EquitySnapshotRow, SignalRow, TradeRow
from tradingbot.storage.repositories import Repositories, bind_repos

KEY_HEALTH = "live_health"
KEY_RUNTIME = "live_runtime"

MappingLike = dict[str, Any] | None


@dataclass(slots=True)
class LiveHealth:
    """What ``live status`` and the dashboard Live page show."""

    running: bool = False
    mode: str = "paper"
    paused: bool = False
    started_at: datetime | None = None
    last_cycle_at: datetime | None = None
    last_error: str | None = None
    bars_processed: int = 0
    pid: int | None = None

    def to_json(self) -> dict[str, Any]:
        """JSON-ready snapshot."""
        return {
            "running": self.running,
            "mode": self.mode,
            "paused": self.paused,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_cycle_at": self.last_cycle_at.isoformat() if self.last_cycle_at else None,
            "last_error": self.last_error,
            "bars_processed": self.bars_processed,
            "pid": self.pid,
        }

    @classmethod
    def from_json(cls, payload: MappingLike) -> LiveHealth:
        """Rebuild from the key/value bag."""
        data = dict(payload or {})
        raw_pid = data.get("pid")
        return cls(
            running=bool(data.get("running", False)),
            mode=str(data.get("mode", "paper")),
            paused=bool(data.get("paused", False)),
            started_at=_parse_ts(data.get("started_at")),
            last_cycle_at=_parse_ts(data.get("last_cycle_at")),
            last_error=str(data["last_error"]) if data.get("last_error") else None,
            bars_processed=int(data.get("bars_processed") or 0),
            pid=int(raw_pid) if raw_pid is not None else None,
        )


@dataclass(slots=True)
class LiveStatus:
    """Aggregated view of the live database for CLI and the dashboard."""

    health: LiveHealth
    last_processed: dict[str, datetime] = field(default_factory=dict)
    open_positions: list[Position] = field(default_factory=list)
    recent_signals: list[SignalRow] = field(default_factory=list)
    recent_trades: list[TradeRow] = field(default_factory=list)
    latest_equity: EquitySnapshotRow | None = None
    pending_notifications: int = 0
    uptime_sec: float | None = None


class LiveStore:
    """Thin façade over :class:`Database` for the live loop."""

    def __init__(self, database: Database) -> None:
        self.db = database

    @classmethod
    def from_config(cls, config: AppConfig) -> LiveStore:
        """Open ``live.state_db``, creating the file if needed."""
        return cls(Database.from_config(config))

    @classmethod
    def from_path(cls, path: Path | str) -> LiveStore:
        """Open an explicit SQLite path."""
        return cls(Database(path))

    def read_status(self, *, now: datetime | None = None) -> LiveStatus:
        """Assemble the dashboard/CLI snapshot from the current database."""
        with self.db.session_scope() as session:
            repos = bind_repos(session)
            health = LiveHealth.from_json(repos.state.get(KEY_HEALTH, {}))
            health.paused = repos.state.is_paused()
            health.started_at = health.started_at or repos.state.started_at()
            mapping = repos.state.get("last_processed_bar") or {}
            last_processed = {
                str(symbol): parsed
                for symbol, raw in dict(mapping).items()
                if (parsed := _parse_ts(raw)) is not None
            }
            moment = now or utcnow()
            uptime = None
            if health.started_at is not None and health.running:
                uptime = (moment - health.started_at).total_seconds()
            return LiveStatus(
                health=health,
                last_processed=last_processed,
                open_positions=repos.positions.open_as_domain(),
                recent_signals=repos.signals.recent(limit=20),
                recent_trades=repos.trades.recent(limit=20),
                latest_equity=repos.state.latest_equity(),
                pending_notifications=len(repos.notifications.pending(limit=100)),
                uptime_sec=uptime,
            )

    def save_health(self, health: LiveHealth) -> None:
        """Persist the health block."""
        with self.db.session_scope() as session:
            bind_repos(session).state.set(KEY_HEALTH, health.to_json())

    def save_runtime(
        self,
        *,
        portfolio: Portfolio,
        pending: Mapping[str, PendingOrder | None],
        last_exit_ts: dict[str, datetime],
        risk: RiskManager,
    ) -> None:
        """Snapshot cash, pending orders and risk flags."""
        payload = {
            "balance": portfolio.balance,
            "peak_equity": portfolio.peak_equity,
            "slippage_paid": portfolio.slippage_paid,
            "pending": {
                symbol: _dump_pending(order)
                for symbol, order in pending.items()
                if order is not None
            },
            "last_exit_ts": {symbol: ts.isoformat() for symbol, ts in last_exit_ts.items()},
            "open_books": _dump_books(portfolio),
            "risk": {
                "peak_equity": risk.state.peak_equity,
                "current_equity": risk.state.current_equity,
                "day": risk.state.day.isoformat() if risk.state.day else None,
                "day_start_equity": risk.state.day_start_equity,
                "day_realized_pnl": risk.state.day_realized_pnl,
                "halted": risk.state.halted,
                "halt_reason": risk.state.halt_reason,
            },
        }
        with self.db.session_scope() as session:
            bind_repos(session).state.set(KEY_RUNTIME, payload)

    def load_runtime(self) -> dict[str, Any]:
        """Raw runtime payload, or an empty mapping on a fresh database."""
        with self.db.session_scope() as session:
            payload = bind_repos(session).state.get(KEY_RUNTIME) or {}
            return dict(payload) if isinstance(payload, dict) else {}

    def persist_bar(
        self,
        *,
        symbol: str,
        bar_ts: datetime,
        portfolio: Portfolio,
        marks: dict[str, float],
        new_signals: list[tuple[Signal, bool, str]],
        new_trades: list[Trade],
    ) -> None:
        """Write everything that changed on one processed bar."""
        with self.db.session_scope() as session:
            repos = bind_repos(session)
            repos.state.set_last_processed_bar(symbol, bar_ts)
            for signal, approved, reason in new_signals:
                status = SignalStatus.NEW if approved else SignalStatus.REJECTED
                _, created = repos.signals.save(signal, status=status)
                if not created:
                    continue
                payload: dict[str, Any] = {
                    "kind": "signal" if approved else "rejected_signal",
                    "signal_uid": signal.uid,
                    "symbol": signal.symbol,
                    "side": signal.side.value,
                    "signal_type": signal.signal_type.value,
                    "status": status.value,
                }
                if not approved:
                    payload["reason"] = reason
                repos.notifications.enqueue(payload)
            for trade in new_trades:
                repos.trades.add(trade)
            _sync_positions(repos, portfolio)
            equity = portfolio.equity(marks)
            drawdown = (
                (portfolio.peak_equity - equity) / portfolio.peak_equity
                if portfolio.peak_equity > 0
                else 0.0
            )
            repos.state.record_equity(
                bar_ts,
                balance=portfolio.balance,
                equity=equity,
                open_positions=len(portfolio.open_trades),
                drawdown_pct=drawdown,
            )


def restore_portfolio(config: AppConfig, store: LiveStore, engine_portfolio: Portfolio) -> None:
    """Load cash and open positions into ``engine_portfolio``."""
    runtime = store.load_runtime()
    engine_portfolio.balance = float(runtime.get("balance", config.backtest.initial_capital))
    engine_portfolio.peak_equity = float(runtime.get("peak_equity", engine_portfolio.balance))
    engine_portfolio.slippage_paid = float(runtime.get("slippage_paid", 0.0))
    books = runtime.get("open_books") or {}
    status = store.read_status()
    for position in status.open_positions:
        book = books.get(position.symbol) or {}
        record = OpenTrade(
            position=position,
            entry_fee=float(book.get("entry_fee", 0.0)),
            entry_bar=int(book.get("entry_bar", 0)),
            mae_r=float(book.get("mae_r", 0.0)),
            mfe_r=float(book.get("mfe_r", 0.0)),
            funding=float(book.get("funding", 0.0)),
            exits=[_load_chunk(item) for item in book.get("exits") or []],
        )
        engine_portfolio.open_trades[position.symbol] = record


def restore_pending(store: LiveStore) -> dict[str, PendingOrder]:
    """Queued next-bar orders, keyed by symbol."""
    runtime = store.load_runtime()
    pending: dict[str, PendingOrder] = {}
    for symbol, blob in dict(runtime.get("pending") or {}).items():
        order = _load_pending(blob)
        if order is not None:
            pending[str(symbol)] = order
    return pending


def restore_last_exits(store: LiveStore) -> dict[str, datetime]:
    """Last exit timestamps used to rebuild the cooldown."""
    runtime = store.load_runtime()
    result: dict[str, datetime] = {}
    for symbol, raw in dict(runtime.get("last_exit_ts") or {}).items():
        parsed = _parse_ts(raw)
        if parsed is not None:
            result[str(symbol)] = parsed
    return result


def restore_risk(store: LiveStore, risk: RiskManager) -> None:
    """Copy halt / day-loss bookkeeping back onto the risk manager."""
    runtime = store.load_runtime()
    blob = runtime.get("risk") or {}
    if not blob:
        return
    day_raw = blob.get("day")
    day = None
    if day_raw:
        day = datetime.fromisoformat(str(day_raw)).date()
    risk.state = RiskState(
        peak_equity=float(blob.get("peak_equity") or 0.0),
        current_equity=float(blob.get("current_equity") or 0.0),
        day=day,
        day_start_equity=float(blob.get("day_start_equity") or 0.0),
        day_realized_pnl=float(blob.get("day_realized_pnl") or 0.0),
        halted=bool(blob.get("halted", False)),
        halt_reason=str(blob.get("halt_reason") or ""),
    )


def _sync_positions(repos: Repositories, portfolio: Portfolio) -> None:
    open_symbols = set(portfolio.open_trades)
    for row in repos.positions.list_open():
        if row.symbol not in open_symbols:
            repos.positions.close(row.symbol)
    for record in portfolio.open_trades.values():
        repos.positions.upsert_open(record.position)


def _dump_pending(order: PendingOrder) -> dict[str, Any]:
    return {
        "size": order.size,
        "signal": order.signal.model_dump(mode="json"),
    }


def _load_pending(blob: Any) -> PendingOrder | None:
    if not isinstance(blob, dict) or "signal" not in blob:
        return None
    signal = Signal.model_validate(blob["signal"])
    return PendingOrder(signal=signal, size=float(blob.get("size") or signal.size))


def _dump_books(portfolio: Portfolio) -> dict[str, Any]:
    books: dict[str, Any] = {}
    for symbol, record in portfolio.open_trades.items():
        books[symbol] = {
            "entry_fee": record.entry_fee,
            "entry_bar": record.entry_bar,
            "mae_r": record.mae_r,
            "mfe_r": record.mfe_r,
            "funding": record.funding,
            "exits": [
                {
                    "price": chunk.price,
                    "size": chunk.size,
                    "fee": chunk.fee,
                    "reason": chunk.reason.value,
                    "moment": chunk.moment.isoformat(),
                    "slippage": chunk.slippage,
                }
                for chunk in record.exits
            ],
        }
    return books


def _load_chunk(item: dict[str, Any]) -> ExitChunk:
    moment = _parse_ts(item.get("moment")) or utcnow()
    return ExitChunk(
        price=float(item["price"]),
        size=float(item["size"]),
        fee=float(item["fee"]),
        reason=ExitReason(str(item["reason"])),
        moment=moment,
        slippage=float(item.get("slippage") or 0.0),
    )


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def format_status(status: LiveStatus) -> str:
    """Human-readable ``live status`` dump."""
    health = status.health
    equity = status.latest_equity
    lines = [
        f"mode              {health.mode}",
        f"running           {'yes' if health.running else 'no'}",
        f"paused            {'yes' if health.paused else 'no'}",
        f"pid               {health.pid if health.pid is not None else '—'}",
        f"started           {_fmt_ts(health.started_at)}",
        f"uptime            {_fmt_uptime(status.uptime_sec)}",
        f"last cycle        {_fmt_ts(health.last_cycle_at)}",
        f"bars processed    {health.bars_processed}",
        f"last error        {health.last_error or '—'}",
        f"equity            {_fmt_equity(equity)}",
        f"open positions    {len(status.open_positions)}",
        f"pending notify    {status.pending_notifications}",
        "",
        "last processed",
    ]
    if status.last_processed:
        for symbol, ts in sorted(status.last_processed.items()):
            lines.append(f"  {symbol:12} {_fmt_ts(ts)}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("open positions")
    if status.open_positions:
        for position in status.open_positions:
            lines.append(
                f"  {position.symbol:12} {position.side.value:5} "
                f"size={position.size:.6g} entry={position.entry_price:.6g} "
                f"stop={position.current_stop:.6g}"
            )
    else:
        lines.append("  (none)")
    return "\n".join(lines)


def _fmt_ts(value: datetime | None) -> str:
    if value is None:
        return "—"
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")


def _fmt_uptime(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _fmt_equity(row: EquitySnapshotRow | None) -> str:
    if row is None:
        return "—"
    return f"{row.equity:,.2f} (dd {row.drawdown_pct:.1%})"
