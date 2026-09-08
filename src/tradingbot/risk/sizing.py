"""Position sizing from risk, not from capital.

The size of a trade follows from one question: how much of the account are we
willing to lose if the stop is hit? A wider ATR stop therefore buys fewer units,
so the loss on a bad trade is the same fraction of equity regardless of which
instrument it happened on (tech.md section 7.6).
"""

from __future__ import annotations

from dataclasses import dataclass

from tradingbot.core.models import MarketMeta


@dataclass(frozen=True, slots=True)
class SizingResult:
    """Outcome of a sizing calculation."""

    size: float
    risk_amount: float
    r_value: float
    capped_by_notional: bool = False
    rounded_away: bool = False
    reason: str = ""

    @property
    def is_tradable(self) -> bool:
        """True when the computed size can actually be sent to a broker."""
        return self.size > 0

    def notional(self, price: float) -> float:
        """Exposure of the sized position at ``price``."""
        return self.size * price


def position_size(
    *,
    equity: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float,
    max_notional_pct: float = 1.0,
    market: MarketMeta | None = None,
) -> SizingResult:
    """Compute the position size for one trade.

    Args:
        equity: Account equity the risk fraction applies to.
        entry_price: Expected fill price.
        stop_price: Initial protective stop.
        risk_pct: Fraction of equity to put at risk, e.g. ``0.01``.
        max_notional_pct: Hard cap on exposure as a fraction of equity.
        market: Instrument limits; permissive defaults are used when omitted.

    Returns:
        A :class:`SizingResult`; a zero size always carries a ``reason``.
    """
    meta = market or MarketMeta.permissive("UNKNOWN")
    r_value = abs(entry_price - stop_price)

    if equity <= 0:
        return SizingResult(0.0, 0.0, r_value, reason="equity is not positive")
    if entry_price <= 0:
        return SizingResult(0.0, 0.0, r_value, reason="entry price is not positive")
    if r_value <= 0:
        return SizingResult(0.0, 0.0, 0.0, reason="stop coincides with entry, risk is undefined")
    if risk_pct <= 0:
        return SizingResult(0.0, 0.0, r_value, reason="risk_per_trade_pct is not positive")

    risk_amount = equity * risk_pct
    raw_size = risk_amount / r_value

    capped = False
    notional_cap = max_notional_pct * equity
    if raw_size * entry_price > notional_cap:
        raw_size = notional_cap / entry_price
        capped = True

    size = meta.round_size(raw_size)
    if size <= 0:
        return SizingResult(
            0.0,
            risk_amount,
            r_value,
            capped_by_notional=capped,
            rounded_away=True,
            reason=f"size {raw_size:.10g} is below the lot step {meta.lot_step:g}",
        )
    if not meta.is_tradable(size, entry_price):
        return SizingResult(
            0.0,
            risk_amount,
            r_value,
            capped_by_notional=capped,
            reason=(
                f"size {size:g} does not clear the exchange minimums "
                f"(min size {meta.min_size:g}, min notional {meta.min_notional:g})"
            ),
        )

    return SizingResult(
        size=size,
        risk_amount=size * r_value,
        r_value=r_value,
        capped_by_notional=capped,
    )
