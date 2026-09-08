"""Event-driven backtest engine.

The bar processing order of tech.md section 8.2 is the whole point of this
module, and it is deliberately rigid:

1. at the **open** of bar ``t`` the order queued on bar ``t-1`` is filled;
2. **inside** bar ``t`` stops and targets are checked against high/low, with the
   stop always assumed to trigger first;
3. funding accrues for every 8-hour boundary crossed;
4. at the **close** of bar ``t`` the strategy sees the bar for the first time and
   may queue an order for ``t+1``;
5. equity is marked to market and recorded.

Because step 4 comes last, a signal produced on bar ``t`` physically cannot be
filled at bar ``t`` prices. That property is asserted by a dedicated test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger

from tradingbot.backtest.broker import PaperBroker
from tradingbot.backtest.portfolio import ExitChunk, OpenTrade, Portfolio
from tradingbot.config.models import AppConfig
from tradingbot.core.enums import ExitReason, Side, SignalType
from tradingbot.core.exceptions import BacktestError
from tradingbot.core.models import MarketMeta, Position, Signal, TakeProfit
from tradingbot.risk.manager import RiskManager
from tradingbot.strategy.base import Strategy

SIGNAL_COLUMNS = [
    "bar_ts",
    "symbol",
    "timeframe",
    "signal_type",
    "side",
    "price",
    "stop_loss",
    "size",
    "risk_pct",
    "status",
    "reason",
    "reject_reason",
]


@dataclass(frozen=True, slots=True)
class PendingOrder:
    """An intent waiting for the next bar's open."""

    signal: Signal
    size: float


@dataclass(slots=True)
class SymbolState:
    """Per-instrument bookkeeping carried across bars."""

    pending: PendingOrder | None = None
    last_exit_bar: int | None = None


@dataclass(slots=True)
class BacktestResult:
    """Everything a run produces, before it is written to disk."""

    trades: pd.DataFrame
    equity: pd.DataFrame
    signals: pd.DataFrame
    initial_capital: float
    final_equity: float
    symbols: list[str]
    timeframe: str
    bars_processed: int = 0
    slippage_cost: float = 0.0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def total_return_pct(self) -> float:
        """Overall return as a fraction of starting capital."""
        if self.initial_capital <= 0:
            return 0.0
        return self.final_equity / self.initial_capital - 1.0


class BacktestEngine:
    """Runs one strategy over one or more instruments with shared capital."""

    def __init__(
        self,
        config: AppConfig,
        strategy: Strategy,
        *,
        broker: PaperBroker | None = None,
        risk: RiskManager | None = None,
        markets: Mapping[str, MarketMeta] | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.markets = dict(markets or {})
        self.broker = broker or PaperBroker(config.costs, self.markets)
        self.risk = risk or RiskManager(config.risk, self.markets)
        self.portfolio = Portfolio(config.backtest.initial_capital)
        self._signals: list[dict[str, Any]] = []
        self._states: dict[str, SymbolState] = {}

    def run(self, data: Mapping[str, pd.DataFrame]) -> BacktestResult:
        """Execute the backtest.

        Args:
            data: Raw OHLCV frames keyed by symbol, indexed by candle open time.

        Returns:
            The trade journal, equity curve and signal log of the run.

        Raises:
            BacktestError: No usable data was supplied.
        """
        prepared = self._prepare(data)
        if not prepared:
            raise BacktestError("no data supplied to the backtest engine")

        symbols = sorted(prepared)
        positions = {symbol: _position_lookup(prepared[symbol]) for symbol in symbols}
        self._states = {symbol: SymbolState() for symbol in symbols}
        warmup = self.strategy.warmup_period()
        marks: dict[str, float] = {}

        timeline = _timeline(prepared)
        self.risk.update_equity(timeline[0], self.portfolio.balance)

        for moment in timeline:
            for symbol in symbols:
                index = positions[symbol].get(moment)
                if index is None:
                    continue
                self._process_bar(
                    symbol=symbol,
                    frame=prepared[symbol],
                    index=index,
                    moment=moment,
                    state=self._states[symbol],
                    warmup=warmup,
                    marks=marks,
                )
            self.portfolio.mark(moment, marks)

        self._close_remaining(prepared, timeline[-1], marks)
        final_equity = self.portfolio.remark(timeline[-1], marks)

        return BacktestResult(
            trades=self.portfolio.trades_frame(),
            equity=self.portfolio.equity_frame(),
            signals=self._signals_frame(),
            initial_capital=self.portfolio.initial_capital,
            final_equity=final_equity,
            symbols=symbols,
            timeframe=self.config.exchange.timeframe,
            bars_processed=sum(len(frame) for frame in prepared.values()),
            slippage_cost=self.portfolio.slippage_paid,
            rejections=self._rejection_counts(),
        )

    def _prepare(self, data: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Add indicator columns to every non-empty frame."""
        prepared: dict[str, pd.DataFrame] = {}
        for symbol, frame in data.items():
            if frame.empty:
                logger.warning("{symbol}: no candles, skipping", symbol=symbol)
                continue
            prepared[symbol] = self.strategy.prepare(frame)
        return prepared

    def _process_bar(
        self,
        *,
        symbol: str,
        frame: pd.DataFrame,
        index: int,
        moment: datetime,
        state: SymbolState,
        warmup: int,
        marks: dict[str, float],
    ) -> None:
        """Run the five steps of section 8.2 for one instrument on one bar."""
        row = frame.iloc[index]
        atr_value = _optional_float(row.get("atr"))

        self._fill_pending(symbol, state, row, moment, index, atr_value)
        self._check_intrabar_exits(symbol, row, moment, index, atr_value)
        self._accrue_funding(symbol, row, moment, frame, index)

        marks[symbol] = float(row["close"])
        equity = self.portfolio.equity(marks)
        self.risk.update_equity(moment, equity)

        if index >= warmup:
            self._consult_strategy(symbol, frame, index, moment, state, equity)

    def _fill_pending(
        self,
        symbol: str,
        state: SymbolState,
        row: pd.Series,
        moment: datetime,
        index: int,
        atr_value: float | None,
    ) -> None:
        """Step 1: execute the order queued on the previous bar, at this open."""
        order = state.pending
        state.pending = None
        if order is None:
            return

        bar_open = float(row["open"])
        signal = order.signal
        if signal.signal_type is SignalType.ENTRY:
            self._open_position(symbol, order, bar_open, moment, index, atr_value)
            return

        record = self.portfolio.open_trades.get(symbol)
        if record is None:
            return
        reason = signal.exit_reason or ExitReason.CHANNEL_EXIT
        self._close(symbol, record.position.size, bar_open, reason, moment, index, atr_value)

    def _open_position(
        self,
        symbol: str,
        order: PendingOrder,
        bar_open: float,
        moment: datetime,
        index: int,
        atr_value: float | None,
    ) -> None:
        """Turn an approved entry order into an open position at the bar open."""
        if symbol in self.portfolio.open_trades:
            return
        signal = order.signal
        fill = self.broker.open_position_fill(signal.side, bar_open, order.size, atr_value)

        # The stop distance travels with the signal; anchoring it to the actual
        # fill keeps the risked amount equal to what the risk manager approved.
        distance = signal.r_value
        stop = fill.price - distance if signal.side is Side.LONG else fill.price + distance
        targets = [
            TakeProfit(
                price=fill.price + level.r_multiple * distance * signal.side.sign,
                r_multiple=level.r_multiple,
                fraction=level.fraction,
            )
            for level in signal.take_profits
        ]

        position = Position(
            symbol=symbol,
            side=signal.side,
            entry_ts=moment,
            entry_price=fill.price,
            size=fill.size,
            initial_size=fill.size,
            initial_stop=stop,
            current_stop=stop,
            take_profits=targets,
            highest_price=fill.price,
            lowest_price=fill.price,
            signal_uid=signal.uid,
            entry_reason=signal.reason,
            equity_at_entry=self.portfolio.balance,
        )
        self.portfolio.open_position(
            OpenTrade(position=position, entry_fee=fill.fee, entry_bar=index),
            slippage=fill.slippage,
        )
        logger.debug(
            "{symbol} {side} entry at {price:.6g}, size {size:.6g}",
            symbol=symbol,
            side=signal.side.value,
            price=fill.price,
            size=fill.size,
        )

    def _check_intrabar_exits(
        self,
        symbol: str,
        row: pd.Series,
        moment: datetime,
        index: int,
        atr_value: float | None,
    ) -> None:
        """Step 2: stops and targets, with the stop winning any tie."""
        record = self.portfolio.open_trades.get(symbol)
        if record is None:
            return

        high, low = float(row["high"]), float(row["low"])
        bar_open = float(row["open"])
        record.observe(high=high, low=low)
        position = record.position

        stop_touched = (
            low <= position.current_stop
            if position.side is Side.LONG
            else high >= position.current_stop
        )
        if stop_touched:
            reference = self.broker.stop_fill_reference(
                position.side, position.current_stop, bar_open
            )
            reason = ExitReason.TRAIL if position.filled_take_profits else ExitReason.STOP
            self._close(symbol, position.size, reference, reason, moment, index, atr_value)
            return

        for level in list(position.take_profits):
            touched = high >= level.price if position.side is Side.LONG else low <= level.price
            if not touched:
                continue
            reference = self.broker.target_fill_reference(position.side, level.price, bar_open)
            size = min(position.initial_size * level.fraction, position.size)
            position.take_profits = [
                item for item in position.take_profits if item.price != level.price
            ]
            position.filled_take_profits += 1
            self._close(symbol, size, reference, ExitReason.TP1, moment, index, atr_value)
            if symbol not in self.portfolio.open_trades:
                return

    def _accrue_funding(
        self,
        symbol: str,
        row: pd.Series,
        moment: datetime,
        frame: pd.DataFrame,
        index: int,
    ) -> None:
        """Step 3: charge funding for the 8-hour boundaries inside this bar."""
        record = self.portfolio.open_trades.get(symbol)
        if record is None:
            return
        bar_end = _bar_end(frame, index, moment)
        events = self.broker.funding_events(record.position, moment, bar_end)
        if events:
            cost = self.broker.funding_cost(record.position, float(row["close"]), events)
            self.portfolio.apply_funding(symbol, cost)

    def _consult_strategy(
        self,
        symbol: str,
        frame: pd.DataFrame,
        index: int,
        moment: datetime,
        state: SymbolState,
        equity: float,
    ) -> None:
        """Step 4: let the strategy see the closed bar and queue an order."""
        from tradingbot.core.models import BarContext  # local import keeps the module graph flat

        position = self.portfolio.position_for(symbol)
        bars_since_exit = index - state.last_exit_bar if state.last_exit_bar is not None else None
        ctx = BarContext(
            symbol=symbol,
            timeframe=self.config.exchange.timeframe,
            index=index,
            history=frame.iloc[: index + 1],
            equity=equity,
            position=position,
            bars_since_exit=bars_since_exit,
        )

        if position is not None:
            new_stop = self.strategy.update_stop(ctx)
            if new_stop is not None:
                _tighten(position, new_stop)

        signal = self.strategy.on_bar(ctx)
        if signal is None:
            return

        if signal.signal_type is SignalType.ENTRY:
            decision = self.risk.evaluate(
                signal, equity=equity, open_positions=self._committed_positions()
            )
            self._log_signal(signal, decision.approved, decision.reason)
            if decision.approved:
                state.pending = PendingOrder(signal=signal, size=decision.size)
            return

        self._log_signal(signal, True, "")
        state.pending = PendingOrder(signal=signal, size=signal.size)

    def _committed_positions(self) -> list[Position]:
        """Open positions plus entries already queued for the next open.

        Without the queued ones, several symbols signalling on the same bar would
        each see an empty book and blow straight through the portfolio limits.
        """
        committed = list(self.portfolio.positions)
        for symbol in sorted(self._states):
            order = self._states[symbol].pending
            if order is not None and order.signal.signal_type is SignalType.ENTRY:
                committed.append(_provisional_position(order))
        return committed

    def _close(
        self,
        symbol: str,
        size: float,
        reference: float,
        reason: ExitReason,
        moment: datetime,
        index: int,
        atr_value: float | None,
    ) -> None:
        """Reduce a position and record the trade when it becomes flat."""
        record = self.portfolio.open_trades.get(symbol)
        if record is None or size <= 0:
            return
        fill = self.broker.close_position_fill(record.position.side, reference, size, atr_value)
        trade = self.portfolio.reduce(
            symbol,
            ExitChunk(
                price=fill.price,
                size=fill.size,
                fee=fill.fee,
                reason=reason,
                moment=moment,
                slippage=fill.slippage,
            ),
            bar_index=index,
        )
        if trade is not None:
            self.risk.register_realized_pnl(trade.pnl_net)
            logger.debug(
                "{symbol} closed: {pnl:.2f} ({r:.2f}R) via {reason}",
                symbol=symbol,
                pnl=trade.pnl_net,
                r=trade.pnl_r,
                reason=reason.value,
            )

    def _close_remaining(
        self,
        prepared: Mapping[str, pd.DataFrame],
        moment: datetime,
        marks: Mapping[str, float],
    ) -> None:
        """Flatten anything still open at the end of the data."""
        for symbol in sorted(self.portfolio.open_trades):
            frame = prepared[symbol]
            price = marks.get(symbol, float(frame["close"].iloc[-1]))
            atr_value = _optional_float(frame["atr"].iloc[-1]) if "atr" in frame else None
            record = self.portfolio.open_trades[symbol]
            self._close(
                symbol,
                record.position.size,
                price,
                ExitReason.EOD,
                moment,
                len(frame) - 1,
                atr_value,
            )

    def _log_signal(self, signal: Signal, approved: bool, reason: str) -> None:
        """Append a row to the signal journal, approved or not."""
        self._signals.append(
            {
                "bar_ts": signal.bar_ts,
                "symbol": signal.symbol,
                "timeframe": signal.timeframe,
                "signal_type": signal.signal_type.value,
                "side": signal.side.value,
                "price": signal.price,
                "stop_loss": signal.stop_loss,
                "size": signal.size,
                "risk_pct": signal.risk_pct,
                "status": "APPROVED" if approved else "REJECTED",
                "reason": signal.reason,
                "reject_reason": "" if approved else reason,
            }
        )

    def _signals_frame(self) -> pd.DataFrame:
        """Signal journal as a frame, including rejections and their reasons."""
        if not self._signals:
            return pd.DataFrame(columns=SIGNAL_COLUMNS)
        return pd.DataFrame(self._signals)[SIGNAL_COLUMNS]

    def _rejection_counts(self) -> dict[str, int]:
        """Rejection tallies keyed by reason."""
        counts: dict[str, int] = {}
        for entry in self._signals:
            if entry["status"] == "REJECTED":
                reason = str(entry["reject_reason"])
                counts[reason] = counts.get(reason, 0) + 1
        return counts


def _provisional_position(order: PendingOrder) -> Position:
    """A queued entry seen as if it were already filled, for limit checks."""
    signal = order.signal
    return Position(
        symbol=signal.symbol,
        side=signal.side,
        entry_ts=signal.bar_ts,
        entry_price=signal.price,
        size=order.size,
        initial_size=order.size,
        initial_stop=signal.stop_loss,
        current_stop=signal.stop_loss,
        highest_price=signal.price,
        lowest_price=signal.price,
    )


def _tighten(position: Position, new_stop: float) -> None:
    """Move a stop only in the direction that reduces risk."""
    improved = (
        new_stop > position.current_stop
        if position.side is Side.LONG
        else new_stop < position.current_stop
    )
    if not improved:
        return
    position.current_stop = new_stop
    if not position.breakeven_moved:
        reached_entry = (
            new_stop >= position.entry_price
            if position.side is Side.LONG
            else new_stop <= position.entry_price
        )
        position.breakeven_moved = reached_entry


def _timeline(prepared: Mapping[str, pd.DataFrame]) -> list[datetime]:
    """Sorted union of every symbol's candle timestamps."""
    stamps: set[pd.Timestamp] = set()
    for frame in prepared.values():
        stamps.update(frame.index)
    return [moment.to_pydatetime() for moment in sorted(stamps)]


def _position_lookup(frame: pd.DataFrame) -> dict[datetime, int]:
    """Map each timestamp to its positional index inside ``frame``."""
    return {moment.to_pydatetime(): i for i, moment in enumerate(frame.index)}


def _bar_end(frame: pd.DataFrame, index: int, moment: datetime) -> datetime:
    """Closing time of a bar, inferred from the next candle's open."""
    if index + 1 < len(frame):
        following: pd.Timestamp = frame.index[index + 1]
        return following.to_pydatetime()
    if index > 0:
        previous: pd.Timestamp = frame.index[index - 1]
        return moment + (moment - previous.to_pydatetime())
    return moment


def _optional_float(value: Any) -> float | None:
    """Convert a possibly missing indicator value to ``float`` or ``None``."""
    if value is None or pd.isna(value):
        return None
    return float(value)


def symbols_of(data: Mapping[str, pd.DataFrame]) -> Sequence[str]:
    """Deterministic symbol ordering used throughout the engine."""
    return sorted(data)
