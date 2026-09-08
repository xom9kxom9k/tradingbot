"""Domain models: the vocabulary shared by strategy, backtest and live code.

Everything here is pure data. Models never touch the network, the database or
the clock, which is what lets the very same objects flow through a backtest and
through the live runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tradingbot.core.enums import ExitReason, Regime, RunStatus, Side, SignalType

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance for type checkers only
    import pandas as pd


def utcnow() -> datetime:
    """Current time as a timezone-aware UTC datetime."""
    return datetime.now(tz=UTC)


def _require_utc(value: datetime) -> datetime:
    """Normalise a datetime to UTC, rejecting naive values."""
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware (UTC)")
    return value.astimezone(UTC)


class DomainModel(BaseModel):
    """Base for immutable domain records."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class MutableDomainModel(BaseModel):
    """Base for records the engine updates in place, such as open positions."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Bar(DomainModel):
    """A single OHLCV candle.

    ``ts`` is the *opening* time of the candle, always UTC (FR-1.7).
    """

    ts: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    is_gap: bool = False

    @field_validator("ts")
    @classmethod
    def _utc_ts(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def _consistent_ohlc(self) -> Bar:
        if self.high < self.low:
            raise ValueError("high must be greater than or equal to low")
        if self.high < max(self.open, self.close):
            raise ValueError("high must be the maximum of the bar")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be the minimum of the bar")
        return self

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low


class MarketMeta(DomainModel):
    """Instrument trading rules that constrain order prices and sizes."""

    symbol: str
    price_tick: float = Field(default=0.01, gt=0)
    lot_step: float = Field(default=1e-8, gt=0)
    min_size: float = Field(default=0.0, ge=0)
    min_notional: float = Field(default=0.0, ge=0)

    @classmethod
    def permissive(cls, symbol: str) -> MarketMeta:
        """Fallback metadata used when the exchange does not expose limits."""
        return cls(symbol=symbol)

    def round_price(self, price: float) -> float:
        """Snap ``price`` to the instrument tick size."""
        return round(round(price / self.price_tick) * self.price_tick, 10)

    def round_size(self, size: float) -> float:
        """Floor ``size`` to the lot step; never rounds up into extra risk."""
        steps = math.floor(size / self.lot_step + 1e-9)
        return round(max(steps, 0) * self.lot_step, 10)

    def is_tradable(self, size: float, price: float) -> bool:
        """True when ``size`` clears both the minimum size and notional filters."""
        return size >= self.min_size and size > 0 and size * price >= self.min_notional


class TakeProfit(DomainModel):
    """One partial profit target."""

    price: float = Field(gt=0)
    r_multiple: float = Field(gt=0)
    fraction: float = Field(gt=0, le=1)


class Signal(DomainModel):
    """An intent produced by a strategy on a closed bar.

    A signal is a *request*, not a fill: the backtest and the live runner both
    execute it at the open of the following bar (tech.md section 8.2).
    """

    signal_type: SignalType
    side: Side
    symbol: str
    timeframe: str
    bar_ts: datetime
    price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    take_profits: list[TakeProfit] = Field(default_factory=list)
    size: float = Field(ge=0)
    risk_pct: float = Field(ge=0, le=1)
    reason: str = ""
    indicators: dict[str, float] = Field(default_factory=dict)
    exit_reason: ExitReason | None = None

    @field_validator("bar_ts")
    @classmethod
    def _utc_bar_ts(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def _stop_on_the_losing_side(self) -> Signal:
        if self.signal_type is not SignalType.ENTRY:
            return self
        if self.size <= 0:
            raise ValueError("entry signals must have a positive size")
        if self.side is Side.LONG and self.stop_loss >= self.price:
            raise ValueError("long stop_loss must be below the entry price")
        if self.side is Side.SHORT and self.stop_loss <= self.price:
            raise ValueError("short stop_loss must be above the entry price")
        return self

    @property
    def r_value(self) -> float:
        """Risk per unit: the distance between entry and initial stop."""
        return abs(self.price - self.stop_loss)

    @property
    def symbol_tag(self) -> str:
        """Symbol without the separator, e.g. ``BTCUSDT``."""
        return self.symbol.replace("/", "").replace(":", "")

    @property
    def display_id(self) -> str:
        """Short identifier shown to humans, e.g. ``SIG-20260908-1200-BTCUSDT-L``."""
        return f"SIG-{self.bar_ts:%Y%m%d-%H%M}-{self.symbol_tag}-{self.side.letter}"

    @property
    def uid(self) -> str:
        """Idempotency key: symbol, timeframe, bar and signal type (FR-6.3).

        Stored as ``signals.signal_uid``, whose unique index is what technically
        prevents duplicate notifications after a restart.
        """
        return f"{self.display_id}-{self.timeframe}-{self.signal_type.value}"


@dataclass(frozen=True, slots=True)
class BarContext:
    """Everything a strategy may look at when deciding on a closed bar.

    Attributes:
        symbol: Instrument the decision is about.
        timeframe: Candle size, e.g. ``4h``.
        index: Positional index of the closed bar inside ``history``.
        history: Prepared frame with indicator columns; rows after ``index`` are
            never present, which is what structurally prevents look-ahead.
        position: Currently open position for this symbol, if any.
        equity: Account equity marked to market at the close of the bar.
        bars_since_exit: Bars elapsed since the last exit, used for the cooldown.
    """

    symbol: str
    timeframe: str
    index: int
    history: pd.DataFrame
    equity: float
    position: Position | None = None
    bars_since_exit: int | None = None

    @property
    def bar(self) -> pd.Series:
        """The closed bar being evaluated."""
        return self.history.iloc[self.index]

    @property
    def ts(self) -> datetime:
        """Opening timestamp of the closed bar."""
        moment = self.history.index[self.index]
        return cast(datetime, moment.to_pydatetime())


class Position(MutableDomainModel):
    """An open position, mutated by the engine as bars go by."""

    symbol: str
    side: Side
    entry_ts: datetime
    entry_price: float = Field(gt=0)
    size: float = Field(gt=0)
    initial_size: float = Field(gt=0)
    initial_stop: float = Field(gt=0)
    current_stop: float = Field(gt=0)
    take_profits: list[TakeProfit] = Field(default_factory=list)
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0
    bars_held: int = 0
    highest_price: float = Field(gt=0)
    lowest_price: float = Field(gt=0)
    breakeven_moved: bool = False
    filled_take_profits: int = 0
    signal_uid: str = ""
    entry_reason: str = ""
    equity_at_entry: float = 0.0

    @field_validator("entry_ts")
    @classmethod
    def _utc_entry_ts(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @property
    def r_value(self) -> float:
        """Risk per unit at entry, in quote currency."""
        return abs(self.entry_price - self.initial_stop)

    @property
    def notional(self) -> float:
        """Current exposure at entry price."""
        return self.size * self.entry_price

    def unrealized_pnl(self, price: float) -> float:
        """Mark-to-market PnL of the remaining size at ``price``."""
        return (price - self.entry_price) * self.size * self.side.sign

    def excursion_r(self, price: float) -> float:
        """Signed excursion of ``price`` from entry, expressed in R."""
        if self.r_value == 0:
            return 0.0
        return (price - self.entry_price) * self.side.sign / self.r_value

    def register_extremes(self, high: float, low: float) -> None:
        """Track the running high/low used by the trailing stop and MAE/MFE."""
        self.highest_price = max(self.highest_price, high)
        self.lowest_price = min(self.lowest_price, low)


class Trade(DomainModel):
    """A closed trade, matching the ``trades.parquet`` schema (tech.md 8.4)."""

    trade_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    side: Side
    entry_ts: datetime
    entry_price: float = Field(gt=0)
    exit_ts: datetime
    exit_price: float = Field(gt=0)
    size: float = Field(gt=0)
    initial_stop: float = Field(gt=0)
    r_value: float = Field(ge=0)
    pnl_gross: float
    fees: float
    funding: float
    pnl_net: float
    pnl_r: float
    pnl_pct: float
    bars_held: int = Field(ge=0)
    exit_reason: ExitReason
    mae: float
    mfe: float
    entry_reason: str = ""

    @field_validator("entry_ts", "exit_ts")
    @classmethod
    def _utc_timestamps(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @model_validator(mode="after")
    def _exit_after_entry(self) -> Trade:
        if self.exit_ts < self.entry_ts:
            raise ValueError("exit_ts must not precede entry_ts")
        return self

    @property
    def is_win(self) -> bool:
        """True when the trade closed with a positive net PnL."""
        return self.pnl_net > 0

    def to_row(self) -> dict[str, Any]:
        """Flat, Parquet-friendly representation."""
        row = self.model_dump()
        row["side"] = self.side.value
        row["exit_reason"] = self.exit_reason.value
        return row


class RunMeta(DomainModel):
    """Provenance of a stored run (``runs/<run_id>/meta.json``)."""

    run_id: str
    created_at: datetime = Field(default_factory=utcnow)
    code_version: str
    git_sha: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    data_hash: str | None = None
    symbols: list[str] = Field(default_factory=list)
    timeframe: str = ""
    status: RunStatus = RunStatus.RUNNING
    duration_sec: float | None = None
    note: str = ""

    @field_validator("created_at")
    @classmethod
    def _utc_created(cls, value: datetime) -> datetime:
        return _require_utc(value)


@dataclass(slots=True)
class RegimeSnapshot:
    """Regime plus the indicator values that produced it, kept for audit trails."""

    regime: Regime
    values: dict[str, float] = field(default_factory=dict)


def config_hash(config: dict[str, Any], length: int = 8) -> str:
    """Short, stable hash of a configuration mapping.

    Args:
        config: Any JSON-serialisable mapping.
        length: Number of hex characters to keep.

    Returns:
        The first ``length`` characters of the SHA-256 digest.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def make_run_id(config: dict[str, Any], moment: datetime | None = None) -> str:
    """Build a run identifier of the form ``YYYYMMDD-HHMMSS-<confighash>`` (FR-6.4)."""
    stamp = (moment or utcnow()).astimezone(UTC)
    return f"{stamp:%Y%m%d-%H%M%S}-{config_hash(config)}"
