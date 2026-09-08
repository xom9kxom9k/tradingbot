"""Capital accounting: balance, equity and the trade journal.

Two numbers are tracked separately and both matter. ``balance`` is realised cash
after fees and funding; ``equity`` adds the mark-to-market value of whatever is
still open. Drawdown is measured on equity, because that is what an account
holder actually watches.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime

import pandas as pd

from tradingbot.core.enums import ExitReason, Side
from tradingbot.core.models import Position, Trade, deterministic_trade_id


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One sample of the equity curve."""

    ts: datetime
    balance: float
    equity: float
    open_positions: int
    drawdown_pct: float


@dataclass(slots=True)
class ExitChunk:
    """One partial or full reduction of a position."""

    price: float
    size: float
    fee: float
    reason: ExitReason
    moment: datetime


@dataclass(slots=True)
class OpenTrade:
    """A position plus the bookkeeping needed to close it into a :class:`Trade`."""

    position: Position
    entry_fee: float
    entry_bar: int
    mae_r: float = 0.0
    mfe_r: float = 0.0
    funding: float = 0.0
    exits: list[ExitChunk] = field(default_factory=list)

    def observe(self, high: float, low: float) -> None:
        """Update the running MAE/MFE from a bar's extremes."""
        position = self.position
        position.register_extremes(high=high, low=low)
        favourable = position.excursion_r(high if position.side is Side.LONG else low)
        adverse = position.excursion_r(low if position.side is Side.LONG else high)
        self.mfe_r = max(self.mfe_r, favourable)
        self.mae_r = min(self.mae_r, adverse)

    @property
    def closed_size(self) -> float:
        """Total size already exited."""
        return sum(chunk.size for chunk in self.exits)

    @property
    def is_closed(self) -> bool:
        """True once nothing remains open."""
        return self.position.size <= 1e-12


class Portfolio:
    """Holds cash, open positions and the journal of closed trades."""

    def __init__(self, initial_capital: float) -> None:
        self.initial_capital = initial_capital
        self.balance = initial_capital
        self.open_trades: dict[str, OpenTrade] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[EquityPoint] = []
        self.peak_equity = initial_capital

    @property
    def positions(self) -> list[Position]:
        """Currently open positions, in deterministic symbol order."""
        return [self.open_trades[symbol].position for symbol in sorted(self.open_trades)]

    def position_for(self, symbol: str) -> Position | None:
        """Open position for ``symbol``, if any."""
        record = self.open_trades.get(symbol)
        return record.position if record else None

    def unrealized(self, marks: Mapping[str, float]) -> float:
        """Mark-to-market value of open positions at ``marks``."""
        total = 0.0
        for symbol, record in self.open_trades.items():
            price = marks.get(symbol)
            if price is not None:
                total += record.position.unrealized_pnl(price)
        return total

    def equity(self, marks: Mapping[str, float]) -> float:
        """Balance plus unrealised PnL."""
        return self.balance + self.unrealized(marks)

    def open_position(self, record: OpenTrade) -> None:
        """Register a freshly filled entry and charge its fee."""
        self.open_trades[record.position.symbol] = record
        self.balance -= record.entry_fee

    def apply_funding(self, symbol: str, amount: float) -> None:
        """Charge (or credit) funding on an open position."""
        record = self.open_trades.get(symbol)
        if record is None:
            return
        record.funding += amount
        record.position.funding_paid += amount
        self.balance -= amount

    def reduce(self, symbol: str, chunk: ExitChunk, bar_index: int) -> Trade | None:
        """Close part or all of a position.

        Returns:
            The finished :class:`Trade` when the position is fully closed,
            otherwise ``None``.
        """
        record = self.open_trades[symbol]
        position = record.position
        size = min(chunk.size, position.size)
        if size <= 0:
            return None

        gross = (chunk.price - position.entry_price) * size * position.side.sign
        self.balance += gross - chunk.fee
        position.realized_pnl += gross
        position.fees_paid += chunk.fee
        position.size -= size
        record.exits.append(ExitChunk(chunk.price, size, chunk.fee, chunk.reason, chunk.moment))

        if not record.is_closed:
            return None

        trade = self._finalise(record, bar_index)
        self.trades.append(trade)
        del self.open_trades[symbol]
        return trade

    def _finalise(self, record: OpenTrade, bar_index: int) -> Trade:
        """Fold a fully exited position into a single journal row."""
        position = record.position
        total_size = sum(chunk.size for chunk in record.exits)
        weighted_exit = (
            sum(chunk.price * chunk.size for chunk in record.exits) / total_size
            if total_size
            else position.entry_price
        )
        fees = record.entry_fee + sum(chunk.fee for chunk in record.exits)
        gross = position.realized_pnl
        net = gross - fees - record.funding
        risk_total = position.r_value * position.initial_size

        exit_ts = record.exits[-1].moment
        return Trade(
            trade_id=deterministic_trade_id(
                position.symbol, position.side, position.entry_ts, exit_ts
            ),
            symbol=position.symbol,
            side=position.side,
            entry_ts=position.entry_ts,
            entry_price=position.entry_price,
            exit_ts=exit_ts,
            exit_price=weighted_exit,
            size=position.initial_size,
            initial_stop=position.initial_stop,
            r_value=position.r_value,
            pnl_gross=gross,
            fees=fees,
            funding=record.funding,
            pnl_net=net,
            pnl_r=net / risk_total if risk_total else 0.0,
            pnl_pct=net / position.equity_at_entry if position.equity_at_entry else 0.0,
            bars_held=max(0, bar_index - record.entry_bar),
            exit_reason=record.exits[-1].reason,
            mae=record.mae_r,
            mfe=record.mfe_r,
            entry_reason=position.entry_reason,
        )

    def remark(self, moment: datetime, marks: Mapping[str, float]) -> float:
        """Recompute the newest equity point in place instead of adding one."""
        if self.equity_curve and self.equity_curve[-1].ts == moment:
            self.equity_curve.pop()
        return self.mark(moment, marks)

    def mark(self, moment: datetime, marks: Mapping[str, float]) -> float:
        """Append an equity-curve point and return the current equity."""
        equity = self.equity(marks)
        self.peak_equity = max(self.peak_equity, equity)
        drawdown = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        self.equity_curve.append(
            EquityPoint(
                ts=moment,
                balance=self.balance,
                equity=equity,
                open_positions=len(self.open_trades),
                drawdown_pct=drawdown,
            )
        )
        return equity

    def equity_frame(self) -> pd.DataFrame:
        """Equity curve as a frame indexed by timestamp."""
        if not self.equity_curve:
            return pd.DataFrame(
                columns=["balance", "equity", "open_positions", "drawdown_pct"],
                index=pd.DatetimeIndex([], tz="UTC", name="ts"),
            )
        frame = pd.DataFrame([asdict(point) for point in self.equity_curve]).set_index("ts")
        frame.index = pd.DatetimeIndex(frame.index, name="ts")
        return frame

    def trades_frame(self) -> pd.DataFrame:
        """Trade journal as a frame matching the tech.md section 8.4 schema."""
        columns = [
            "trade_id",
            "symbol",
            "side",
            "entry_ts",
            "entry_price",
            "exit_ts",
            "exit_price",
            "size",
            "initial_stop",
            "r_value",
            "pnl_gross",
            "fees",
            "funding",
            "pnl_net",
            "pnl_r",
            "pnl_pct",
            "bars_held",
            "exit_reason",
            "mae",
            "mfe",
            "entry_reason",
        ]
        if not self.trades:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame([trade.to_row() for trade in self.trades])[columns]
