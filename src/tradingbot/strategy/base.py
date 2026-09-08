"""The strategy contract.

A strategy is a pure function of market history plus current position state. It
receives a :class:`~tradingbot.core.models.BarContext` and returns an intent; it
never fetches data, writes to a database or sends a message. That restriction is
what allows the backtest and the live runner to share one implementation, and it
makes every rule testable with a handful of synthetic candles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from tradingbot.core.models import BarContext, Signal


class Strategy(ABC):
    """Decision logic evaluated once per closed bar."""

    name: str = "abstract"

    @abstractmethod
    def warmup_period(self) -> int:
        """Number of leading bars needed before indicators are meaningful."""

    @abstractmethod
    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of ``frame`` with indicator columns added.

        Implementations must be vectorised and free of look-ahead: the value in
        row ``t`` may only depend on rows ``<= t``.
        """

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> Signal | None:
        """Decide what to do at the close of ``ctx.bar``.

        Returns:
            An entry or exit intent to be executed at the next bar's open, or
            ``None`` to stay put.
        """

    def update_stop(self, ctx: BarContext) -> float | None:
        """Propose a new protective stop for the open position.

        The default is to leave the stop alone. Implementations must never move
        a stop against the position; the engine enforces that too, but returning
        a worse level is a bug worth catching in the strategy itself.

        Returns:
            The new stop price, or ``None`` when nothing changes.
        """
        return None

    def describe(self) -> dict[str, Any]:
        """Parameters of this instance, for run metadata and the ``/config`` command."""
        return {"name": self.name}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
