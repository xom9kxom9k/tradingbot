"""Repositories over the live-state tables (tech.md section 13.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from tradingbot.core.enums import PositionStatus, SignalStatus
from tradingbot.core.exceptions import StorageError
from tradingbot.core.models import Position, Signal, Trade, utcnow
from tradingbot.storage.models import (
    BotStateRow,
    EquitySnapshotRow,
    NotificationRow,
    PositionRow,
    SignalRow,
    TradeRow,
)

KEY_LAST_PROCESSED_BAR = "last_processed_bar"
KEY_PAUSED = "paused"
KEY_STARTED_AT = "started_at"
KEY_SCHEMA_VERSION = "schema_version"


class SignalRepo:
    """Insert-or-get signals. The unique ``signal_uid`` index is the idempotency gate."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_by_uid(self, signal_uid: str) -> SignalRow | None:
        """Return the stored row for ``signal_uid``, if any."""
        return self.session.scalar(select(SignalRow).where(SignalRow.signal_uid == signal_uid))

    def save(
        self,
        signal: Signal,
        *,
        status: SignalStatus = SignalStatus.NEW,
    ) -> tuple[SignalRow, bool]:
        """Persist ``signal`` once.

        Returns:
            ``(row, created)``. A second call with the same ``signal.uid`` returns
            the existing row and ``created=False`` instead of inserting a duplicate.
        """
        existing = self.get_by_uid(signal.uid)
        if existing is not None:
            return existing, False
        row = SignalRow.from_signal(signal, status=status)
        try:
            with self.session.begin_nested():
                self.session.add(row)
                self.session.flush()
        except IntegrityError as exc:
            duplicate = self.get_by_uid(signal.uid)
            if duplicate is None:
                raise StorageError(f"failed to persist signal {signal.uid}") from exc
            return duplicate, False
        return row, True

    def mark_notified(self, signal_uid: str, when: datetime | None = None) -> SignalRow | None:
        """Stamp ``notified_at`` and move the row to NOTIFIED."""
        row = self.get_by_uid(signal_uid)
        if row is None:
            return None
        row.notified_at = when or utcnow()
        row.status = SignalStatus.NOTIFIED.value
        self.session.flush()
        return row

    def mark_status(self, signal_uid: str, status: SignalStatus) -> SignalRow | None:
        """Update the lifecycle flag without touching ``notified_at``."""
        row = self.get_by_uid(signal_uid)
        if row is None:
            return None
        row.status = status.value
        self.session.flush()
        return row

    def recent(self, *, limit: int = 50) -> list[SignalRow]:
        """Newest signals first."""
        stmt = select(SignalRow).order_by(SignalRow.bar_ts.desc(), SignalRow.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        """Number of stored signals."""
        value = self.session.scalar(select(func.count()).select_from(SignalRow))
        return int(value or 0)


class PositionRepo:
    """Open and closed live positions, at most one OPEN row per symbol."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get_open(self, symbol: str) -> PositionRow | None:
        """The open position on ``symbol``, if there is one."""
        stmt = select(PositionRow).where(
            PositionRow.symbol == symbol,
            PositionRow.status == PositionStatus.OPEN.value,
        )
        return self.session.scalar(stmt)

    def list_open(self) -> list[PositionRow]:
        """Every currently open position."""
        stmt = select(PositionRow).where(PositionRow.status == PositionStatus.OPEN.value)
        return list(self.session.scalars(stmt))

    def upsert_open(self, position: Position) -> PositionRow:
        """Insert or update the OPEN row for ``position.symbol``."""
        row = self.get_open(position.symbol)
        if row is None:
            row = PositionRow.from_position(position, status=PositionStatus.OPEN)
            self.session.add(row)
        else:
            row.apply_position(position, status=PositionStatus.OPEN)
        self.session.flush()
        return row

    def close(self, symbol: str, position: Position | None = None) -> PositionRow | None:
        """Mark the open position on ``symbol`` as CLOSED."""
        row = self.get_open(symbol)
        if row is None:
            return None
        if position is not None:
            row.apply_position(position, status=PositionStatus.CLOSED)
        else:
            row.status = PositionStatus.CLOSED.value
        self.session.flush()
        return row

    def open_as_domain(self) -> list[Position]:
        """Open positions as domain objects, for live-state recovery."""
        return [row.to_position() for row in self.list_open()]


class TradeRepo:
    """Closed-trade journal (section 8.4 schema)."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, trade_id: str) -> TradeRow | None:
        """Lookup by the stable trade identifier."""
        return self.session.scalar(select(TradeRow).where(TradeRow.trade_id == trade_id))

    def add(self, trade: Trade) -> tuple[TradeRow, bool]:
        """Insert ``trade``; a repeated ``trade_id`` is a no-op."""
        existing = self.get(trade.trade_id)
        if existing is not None:
            return existing, False
        row = TradeRow.from_trade(trade)
        try:
            with self.session.begin_nested():
                self.session.add(row)
                self.session.flush()
        except IntegrityError as exc:
            duplicate = self.get(trade.trade_id)
            if duplicate is None:
                raise StorageError(f"failed to persist trade {trade.trade_id}") from exc
            return duplicate, False
        return row, True

    def recent(self, *, limit: int = 100) -> list[TradeRow]:
        """Newest exits first."""
        stmt = select(TradeRow).order_by(TradeRow.exit_ts.desc(), TradeRow.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def count(self) -> int:
        """Number of closed trades."""
        value = self.session.scalar(select(func.count()).select_from(TradeRow))
        return int(value or 0)


class StateRepo:
    """JSON key/value bag plus equity snapshots for the running bot."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, key: str, default: Any = None) -> Any:
        """Return the JSON value for ``key``, or ``default``."""
        row = self.session.get(BotStateRow, key)
        return default if row is None else row.value

    def set(self, key: str, value: Any) -> None:
        """Create or replace ``key``."""
        row = self.session.get(BotStateRow, key)
        if row is None:
            self.session.add(BotStateRow(key=key, value=value))
        else:
            row.value = value
        self.session.flush()

    def ensure_schema_version(self, version: int) -> None:
        """Write the schema version the first time the database is created."""
        if self.get(KEY_SCHEMA_VERSION) is None:
            self.set(KEY_SCHEMA_VERSION, version)

    def schema_version(self) -> int:
        """Current schema version stored in the bag, or 0 if missing."""
        value = self.get(KEY_SCHEMA_VERSION, 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def is_paused(self) -> bool:
        """True when the live loop should not open new positions."""
        return bool(self.get(KEY_PAUSED, False))

    def set_paused(self, paused: bool) -> None:
        """Toggle the pause flag used by ``/pause`` and ``/resume``."""
        self.set(KEY_PAUSED, paused)

    def started_at(self) -> datetime | None:
        """Process start time, if the runner has recorded one."""
        value = self.get(KEY_STARTED_AT)
        if not value:
            return None
        return datetime.fromisoformat(str(value))

    def set_started_at(self, when: datetime) -> None:
        """Record when this process began, for the uptime display."""
        self.set(KEY_STARTED_AT, when.isoformat())

    def last_processed_bar(self, symbol: str) -> datetime | None:
        """Last closed bar the live loop accepted for ``symbol``."""
        mapping = self.get(KEY_LAST_PROCESSED_BAR) or {}
        if not isinstance(mapping, dict):
            return None
        raw = mapping.get(symbol)
        if not raw:
            return None
        return datetime.fromisoformat(str(raw))

    def set_last_processed_bar(self, symbol: str, ts: datetime) -> None:
        """Remember that ``symbol`` has been processed through ``ts``."""
        mapping = self.get(KEY_LAST_PROCESSED_BAR) or {}
        if not isinstance(mapping, dict):
            mapping = {}
        mapping[symbol] = ts.isoformat()
        self.set(KEY_LAST_PROCESSED_BAR, mapping)

    def record_equity(
        self,
        ts: datetime,
        *,
        balance: float,
        equity: float,
        open_positions: int,
        drawdown_pct: float,
    ) -> EquitySnapshotRow:
        """Append one marked-to-market sample."""
        row = EquitySnapshotRow(
            ts=ts,
            balance=balance,
            equity=equity,
            open_positions_count=open_positions,
            drawdown_pct=drawdown_pct,
        )
        self.session.add(row)
        self.session.flush()
        return row

    def latest_equity(self) -> EquitySnapshotRow | None:
        """Most recent equity snapshot, if any."""
        stmt = select(EquitySnapshotRow).order_by(EquitySnapshotRow.ts.desc()).limit(1)
        return self.session.scalar(stmt)

    def equity_history(self) -> list[EquitySnapshotRow]:
        """Every marked-to-market sample, oldest first."""
        stmt = select(EquitySnapshotRow).order_by(EquitySnapshotRow.ts.asc())
        return list(self.session.scalars(stmt))


class NotificationQueueRepo:
    """Persistent outbound queue so a Telegram outage cannot drop a message."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(self, payload: dict[str, Any]) -> NotificationRow:
        """Append a message. Delivery is the notifier's job (stage 14)."""
        row = NotificationRow(payload=dict(payload))
        self.session.add(row)
        self.session.flush()
        return row

    def pending(self, *, limit: int = 50) -> list[NotificationRow]:
        """Unsent rows, oldest first."""
        stmt = (
            select(NotificationRow)
            .where(NotificationRow.sent_at.is_(None))
            .order_by(NotificationRow.id.asc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def mark_sent(
        self, notification_id: int, when: datetime | None = None
    ) -> NotificationRow | None:
        """Record a successful delivery."""
        row = self.session.get(NotificationRow, notification_id)
        if row is None:
            return None
        row.sent_at = when or utcnow()
        self.session.flush()
        return row

    def record_attempt(self, notification_id: int, error: str) -> NotificationRow | None:
        """Increment the retry counter and keep the last error string."""
        row = self.session.get(NotificationRow, notification_id)
        if row is None:
            return None
        row.attempts += 1
        row.last_error = error
        self.session.flush()
        return row


@dataclass(slots=True)
class Repositories:
    """The five repositories of stage 12, bound to one session."""

    signals: SignalRepo
    positions: PositionRepo
    trades: TradeRepo
    state: StateRepo
    notifications: NotificationQueueRepo


def bind_repos(session: Session) -> Repositories:
    """Construct every repository against ``session``."""
    return Repositories(
        signals=SignalRepo(session),
        positions=PositionRepo(session),
        trades=TradeRepo(session),
        state=StateRepo(session),
        notifications=NotificationQueueRepo(session),
    )
