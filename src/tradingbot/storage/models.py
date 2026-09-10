"""SQLAlchemy ORM models for the live-mode SQLite state (tech.md 13.2)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

from tradingbot.core.enums import ExitReason, PositionStatus, Side, SignalStatus, SignalType
from tradingbot.core.models import Position, Signal, TakeProfit, Trade, utcnow


class UTCDateTime(TypeDecorator[datetime]):
    """Store datetimes in UTC and always hand them back timezone-aware."""

    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Declarative base for every live-state table."""


class SignalRow(Base):
    """Intent produced on a closed bar, keyed by ``signal_uid`` (FR-6.3)."""

    __tablename__ = "signals"
    __table_args__ = (UniqueConstraint("signal_uid", name="uq_signals_signal_uid"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signal_uid: Mapped[str] = mapped_column(String(160), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(8), nullable=False)
    bar_ts: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    signal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    stop_loss: Mapped[float] = mapped_column(Float, nullable=False)
    take_profits: Mapped[list[Any]] = mapped_column(JSON, default=list)
    size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    risk_pct: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    reason: Mapped[str] = mapped_column(Text, default="")
    indicators: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    notified_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default=SignalStatus.NEW.value)

    @classmethod
    def from_signal(
        cls,
        signal: Signal,
        *,
        status: SignalStatus = SignalStatus.NEW,
        notified_at: datetime | None = None,
    ) -> SignalRow:
        """Build a row from the domain signal; ``signal_uid`` is ``Signal.uid``."""
        return cls(
            signal_uid=signal.uid,
            symbol=signal.symbol,
            timeframe=signal.timeframe,
            bar_ts=signal.bar_ts,
            side=signal.side.value,
            signal_type=signal.signal_type.value,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profits=[tp.model_dump(mode="json") for tp in signal.take_profits],
            size=signal.size,
            risk_pct=signal.risk_pct,
            reason=signal.reason,
            indicators=dict(signal.indicators),
            status=status.value,
            notified_at=notified_at,
        )

    def to_signal(self) -> Signal:
        """Rebuild the domain signal from this row."""
        targets = [TakeProfit.model_validate(item) for item in self.take_profits or []]
        return Signal(
            signal_type=SignalType(self.signal_type),
            side=Side(self.side),
            symbol=self.symbol,
            timeframe=self.timeframe,
            bar_ts=self.bar_ts,
            price=self.price,
            stop_loss=self.stop_loss,
            take_profits=targets,
            size=self.size,
            risk_pct=self.risk_pct,
            reason=self.reason,
            indicators={key: float(value) for key, value in (self.indicators or {}).items()},
        )


class PositionRow(Base):
    """Open or closed live position. Extra fields round-trip the domain Position."""

    __tablename__ = "positions"
    __table_args__ = (
        Index("ix_positions_symbol_status", "symbol", "status"),
        Index(
            "uq_positions_open_symbol",
            "symbol",
            unique=True,
            sqlite_where=text("status = 'OPEN'"),
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_ts: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    current_stop: Mapped[float] = mapped_column(Float, nullable=False)
    initial_stop: Mapped[float] = mapped_column(Float, nullable=False)
    tp_levels: Mapped[list[Any]] = mapped_column(JSON, default=list)
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(8), default=PositionStatus.OPEN.value)
    signal_uid: Mapped[str] = mapped_column(String(160), default="")
    initial_size: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    highest_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lowest_price: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    breakeven_moved: Mapped[bool] = mapped_column(Boolean, default=False)
    filled_take_profits: Mapped[int] = mapped_column(Integer, default=0)
    fees_paid: Mapped[float] = mapped_column(Float, default=0.0)
    funding_paid: Mapped[float] = mapped_column(Float, default=0.0)
    bars_held: Mapped[int] = mapped_column(Integer, default=0)
    entry_reason: Mapped[str] = mapped_column(Text, default="")
    equity_at_entry: Mapped[float] = mapped_column(Float, default=0.0)

    @classmethod
    def from_position(
        cls,
        position: Position,
        *,
        status: PositionStatus = PositionStatus.OPEN,
    ) -> PositionRow:
        """Persist a domain position, including trailing-stop bookkeeping."""
        return cls(
            symbol=position.symbol,
            side=position.side.value,
            entry_ts=position.entry_ts,
            entry_price=position.entry_price,
            size=position.size,
            current_stop=position.current_stop,
            initial_stop=position.initial_stop,
            tp_levels=[tp.model_dump(mode="json") for tp in position.take_profits],
            realized_pnl=position.realized_pnl,
            status=status.value,
            signal_uid=position.signal_uid,
            initial_size=position.initial_size,
            highest_price=position.highest_price,
            lowest_price=position.lowest_price,
            breakeven_moved=position.breakeven_moved,
            filled_take_profits=position.filled_take_profits,
            fees_paid=position.fees_paid,
            funding_paid=position.funding_paid,
            bars_held=position.bars_held,
            entry_reason=position.entry_reason,
            equity_at_entry=position.equity_at_entry,
        )

    def to_position(self) -> Position:
        """Rebuild the mutable domain position."""
        targets = [TakeProfit.model_validate(item) for item in self.tp_levels or []]
        initial_size = self.initial_size if self.initial_size > 0 else max(self.size, 1e-12)
        high = self.highest_price if self.highest_price > 0 else self.entry_price
        low = self.lowest_price if self.lowest_price > 0 else self.entry_price
        return Position(
            symbol=self.symbol,
            side=Side(self.side),
            entry_ts=self.entry_ts,
            entry_price=self.entry_price,
            size=self.size,
            initial_size=initial_size,
            initial_stop=self.initial_stop,
            current_stop=self.current_stop,
            take_profits=targets,
            realized_pnl=self.realized_pnl,
            fees_paid=self.fees_paid,
            funding_paid=self.funding_paid,
            bars_held=self.bars_held,
            highest_price=high,
            lowest_price=low,
            breakeven_moved=self.breakeven_moved,
            filled_take_profits=self.filled_take_profits,
            signal_uid=self.signal_uid,
            entry_reason=self.entry_reason,
            equity_at_entry=self.equity_at_entry,
        )

    def apply_position(self, position: Position, *, status: PositionStatus | None = None) -> None:
        """Copy live fields from ``position`` onto this row."""
        self.size = position.size
        self.current_stop = position.current_stop
        self.initial_stop = position.initial_stop
        self.tp_levels = [tp.model_dump(mode="json") for tp in position.take_profits]
        self.realized_pnl = position.realized_pnl
        self.signal_uid = position.signal_uid
        self.initial_size = position.initial_size
        self.highest_price = position.highest_price
        self.lowest_price = position.lowest_price
        self.breakeven_moved = position.breakeven_moved
        self.filled_take_profits = position.filled_take_profits
        self.fees_paid = position.fees_paid
        self.funding_paid = position.funding_paid
        self.bars_held = position.bars_held
        self.entry_reason = position.entry_reason
        self.equity_at_entry = position.equity_at_entry
        if status is not None:
            self.status = status.value


class TradeRow(Base):
    """Closed trade, matching the section 8.4 journal schema."""

    __tablename__ = "trades"
    __table_args__ = (UniqueConstraint("trade_id", name="uq_trades_trade_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    entry_ts: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    entry_price: Mapped[float] = mapped_column(Float, nullable=False)
    exit_ts: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    exit_price: Mapped[float] = mapped_column(Float, nullable=False)
    size: Mapped[float] = mapped_column(Float, nullable=False)
    initial_stop: Mapped[float] = mapped_column(Float, nullable=False)
    r_value: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_gross: Mapped[float] = mapped_column(Float, nullable=False)
    fees: Mapped[float] = mapped_column(Float, default=0.0)
    funding: Mapped[float] = mapped_column(Float, default=0.0)
    pnl_net: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_r: Mapped[float] = mapped_column(Float, nullable=False)
    pnl_pct: Mapped[float] = mapped_column(Float, nullable=False)
    bars_held: Mapped[int] = mapped_column(Integer, default=0)
    exit_reason: Mapped[str] = mapped_column(String(32), nullable=False)
    mae: Mapped[float] = mapped_column(Float, default=0.0)
    mfe: Mapped[float] = mapped_column(Float, default=0.0)
    entry_reason: Mapped[str] = mapped_column(Text, default="")

    @classmethod
    def from_trade(cls, trade: Trade) -> TradeRow:
        """Persist a closed trade."""
        row = trade.to_row()
        return cls(
            trade_id=str(row["trade_id"]),
            symbol=str(row["symbol"]),
            side=str(row["side"]),
            entry_ts=trade.entry_ts,
            entry_price=float(row["entry_price"]),
            exit_ts=trade.exit_ts,
            exit_price=float(row["exit_price"]),
            size=float(row["size"]),
            initial_stop=float(row["initial_stop"]),
            r_value=float(row["r_value"]),
            pnl_gross=float(row["pnl_gross"]),
            fees=float(row["fees"]),
            funding=float(row["funding"]),
            pnl_net=float(row["pnl_net"]),
            pnl_r=float(row["pnl_r"]),
            pnl_pct=float(row["pnl_pct"]),
            bars_held=int(row["bars_held"]),
            exit_reason=str(row["exit_reason"]),
            mae=float(row["mae"]),
            mfe=float(row["mfe"]),
            entry_reason=str(row["entry_reason"]),
        )

    def to_trade(self) -> Trade:
        """Rebuild the domain trade."""
        return Trade(
            trade_id=self.trade_id,
            symbol=self.symbol,
            side=Side(self.side),
            entry_ts=self.entry_ts,
            entry_price=self.entry_price,
            exit_ts=self.exit_ts,
            exit_price=self.exit_price,
            size=self.size,
            initial_stop=self.initial_stop,
            r_value=self.r_value,
            pnl_gross=self.pnl_gross,
            fees=self.fees,
            funding=self.funding,
            pnl_net=self.pnl_net,
            pnl_r=self.pnl_r,
            pnl_pct=self.pnl_pct,
            bars_held=self.bars_held,
            exit_reason=ExitReason(self.exit_reason),
            mae=self.mae,
            mfe=self.mfe,
            entry_reason=self.entry_reason,
        )


class EquitySnapshotRow(Base):
    """Marked-to-market account sample, one row per bar close."""

    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, index=True)
    balance: Mapped[float] = mapped_column(Float, nullable=False)
    equity: Mapped[float] = mapped_column(Float, nullable=False)
    open_positions_count: Mapped[int] = mapped_column(Integer, default=0)
    drawdown_pct: Mapped[float] = mapped_column(Float, default=0.0)


class BotStateRow(Base):
    """Key/value JSON bag: last processed bar, paused flag, started_at."""

    __tablename__ = "bot_state"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)


class NotificationRow(Base):
    """Outbound message waiting to be delivered or already sent."""

    __tablename__ = "notification_queue"
    __table_args__ = (Index("ix_notification_queue_sent_at", "sent_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
