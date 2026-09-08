"""Paper broker: fills, fees, slippage and funding (tech.md section 8.1).

Every assumption here is deliberately pessimistic. Slippage always works against
us, a bar that gaps through a stop fills at the gap rather than at the stop, and
when a single bar touches both the stop and the target the stop is assumed to
have come first. Optimistic fills are the easiest way to produce a backtest that
looks great and loses money.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from tradingbot.config.models import CostsConfig
from tradingbot.core.enums import Side
from tradingbot.core.models import MarketMeta, Position

FUNDING_INTERVAL = timedelta(hours=8)
BPS = 10_000.0


@dataclass(frozen=True, slots=True)
class Fill:
    """A single execution."""

    price: float
    size: float
    fee: float
    slippage: float = 0.0

    @property
    def notional(self) -> float:
        """Traded value before fees."""
        return self.price * self.size


class PaperBroker:
    """Simulated execution venue used by the backtest and by paper trading."""

    def __init__(
        self,
        costs: CostsConfig,
        markets: Mapping[str, MarketMeta] | None = None,
    ) -> None:
        self.costs = costs
        self.markets = dict(markets or {})

    def market(self, symbol: str) -> MarketMeta:
        """Instrument limits for ``symbol``, permissive when unknown."""
        return self.markets.get(symbol) or MarketMeta.permissive(symbol)

    def slippage_amount(self, price: float, atr: float | None = None) -> float:
        """Absolute price concession for a market order.

        Args:
            price: Reference price of the fill.
            atr: Current ATR, required by the ``atr`` model.
        """
        model = self.costs.slippage_model
        if model == "fixed_bps":
            return price * self.costs.slippage_bps / BPS
        if model == "percent":
            return price * self.costs.slippage_percent
        if atr is None or atr != atr:  # NaN check without importing math
            return 0.0
        return self.costs.slippage_atr_mult * atr

    def fill_price(
        self,
        reference: float,
        *,
        buying: bool,
        atr: float | None = None,
        apply_slippage: bool = True,
    ) -> float:
        """Price actually paid, with slippage pushing against the trade."""
        if not apply_slippage:
            return reference
        concession = self.slippage_amount(reference, atr)
        return reference + concession if buying else reference - concession

    def fee_for(self, notional: float, *, taker: bool = True) -> float:
        """Commission charged on ``notional``."""
        rate = self.costs.taker_fee if taker else self.costs.maker_fee
        return abs(notional) * rate

    def open_position_fill(
        self,
        side: Side,
        reference: float,
        size: float,
        atr: float | None = None,
    ) -> Fill:
        """Fill that establishes a position: buying a long, selling a short."""
        return self._fill(reference, size, buying=side is Side.LONG, atr=atr)

    def close_position_fill(
        self,
        side: Side,
        reference: float,
        size: float,
        atr: float | None = None,
    ) -> Fill:
        """Fill that reduces a position: selling a long, buying back a short."""
        return self._fill(reference, size, buying=side is Side.SHORT, atr=atr)

    def _fill(self, reference: float, size: float, *, buying: bool, atr: float | None) -> Fill:
        """Execute against ``reference``, recording what slippage actually cost."""
        price = self.fill_price(reference, buying=buying, atr=atr)
        return Fill(
            price=price,
            size=size,
            fee=self.fee_for(price * size),
            slippage=abs(price - reference) * size,
        )

    def stop_fill_reference(self, side: Side, stop: float, bar_open: float) -> float:
        """Reference price for a stop that has been touched.

        A bar that opens beyond the stop has gapped through it, and the fill
        happens at the open rather than at the stop level.
        """
        if side is Side.LONG:
            return min(stop, bar_open)
        return max(stop, bar_open)

    def target_fill_reference(self, side: Side, target: float, bar_open: float) -> float:
        """Reference price for a profit target that has been touched."""
        if side is Side.LONG:
            return max(target, bar_open)
        return min(target, bar_open)

    def funding_events(self, position: Position, bar_start: datetime, bar_end: datetime) -> int:
        """Number of 8-hour funding boundaries crossed inside a bar."""
        if not self.costs.apply_funding:
            return 0
        start = max(position.entry_ts, bar_start)
        return len(funding_boundaries(start, bar_end))

    def funding_cost(self, position: Position, price: float, events: int) -> float:
        """Funding paid over ``events`` boundaries; longs pay, shorts receive.

        A positive result is money leaving the account.
        """
        if events <= 0:
            return 0.0
        notional = abs(position.size * price)
        return self.costs.funding_rate_8h * notional * events * position.side.sign


def funding_boundaries(start: datetime, end: datetime) -> list[datetime]:
    """UTC 00:00 / 08:00 / 16:00 marks strictly after ``start`` and up to ``end``."""
    if end <= start:
        return []
    seconds = int(FUNDING_INTERVAL.total_seconds())
    epoch = int(start.timestamp())
    first = datetime.fromtimestamp((epoch // seconds + 1) * seconds, tz=UTC)
    marks: list[datetime] = []
    moment = first
    while moment <= end:
        marks.append(moment)
        moment += FUNDING_INTERVAL
    return marks
