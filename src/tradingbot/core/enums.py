"""Enumerations shared across the domain.

All of them derive from :class:`enum.StrEnum` so they serialise to plain strings
in Parquet, JSON and SQLite without extra converters.
"""

from __future__ import annotations

from enum import StrEnum


class Side(StrEnum):
    """Direction of a position or signal."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        """``+1`` for long, ``-1`` for short; multiplies raw price deltas into PnL."""
        return 1 if self is Side.LONG else -1

    @property
    def opposite(self) -> Side:
        """The other side."""
        return Side.SHORT if self is Side.LONG else Side.LONG

    @property
    def letter(self) -> str:
        """Single-character tag used in human-readable identifiers."""
        return "L" if self is Side.LONG else "S"


class SignalType(StrEnum):
    """What a signal asks the execution layer to do."""

    ENTRY = "ENTRY"
    EXIT = "EXIT"
    SCALE_OUT = "SCALE_OUT"


class ExitReason(StrEnum):
    """Why a position was closed, fully or partially."""

    STOP = "STOP"
    TP1 = "TP1"
    TRAIL = "TRAIL"
    CHANNEL_EXIT = "CHANNEL_EXIT"
    REGIME_FLIP = "REGIME_FLIP"
    EOD = "EOD"
    RISK_STOP = "RISK_STOP"


class Regime(StrEnum):
    """Market regime derived from the EMA filter (tech.md section 7.4)."""

    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class OrderType(StrEnum):
    """Order flavours understood by the broker adapters."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class RunStatus(StrEnum):
    """Lifecycle of a stored run."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class PositionStatus(StrEnum):
    """Lifecycle of a position record."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class SignalStatus(StrEnum):
    """Lifecycle of a persisted live signal (section 13.2)."""

    NEW = "NEW"
    NOTIFIED = "NOTIFIED"
    REJECTED = "REJECTED"
