"""Portfolio-level risk limits (tech.md section 7.9).

The strategy decides *whether* the market looks right; the risk manager decides
whether the account can afford the trade. Every rejection carries a reason so it
can be logged, counted in the run report and optionally forwarded to Telegram.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime

from loguru import logger

from tradingbot.config.models import RiskConfig
from tradingbot.core.models import MarketMeta, Position, Signal
from tradingbot.risk.sizing import SizingResult, position_size


@dataclass(frozen=True, slots=True)
class RiskDecision:
    """Verdict on a proposed entry."""

    approved: bool
    reason: str
    size: float = 0.0
    sizing: SizingResult | None = None

    @classmethod
    def reject(cls, reason: str, sizing: SizingResult | None = None) -> RiskDecision:
        """Build a rejection carrying a human-readable explanation."""
        return cls(approved=False, reason=reason, size=0.0, sizing=sizing)

    @classmethod
    def accept(cls, sizing: SizingResult, reason: str = "") -> RiskDecision:
        """Build an approval with the final position size."""
        return cls(approved=True, reason=reason, size=sizing.size, sizing=sizing)


@dataclass(slots=True)
class RiskState:
    """Mutable bookkeeping the limits are evaluated against."""

    peak_equity: float = 0.0
    current_equity: float = 0.0
    day: date | None = None
    day_start_equity: float = 0.0
    day_realized_pnl: float = 0.0
    halted: bool = False
    halt_reason: str = ""

    @property
    def drawdown_pct(self) -> float:
        """Current distance below the equity peak, as a fraction."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, (self.peak_equity - self.current_equity) / self.peak_equity)

    @property
    def day_loss_pct(self) -> float:
        """Today's realized loss as a positive fraction of the day's opening equity."""
        if self.day_start_equity <= 0:
            return 0.0
        return max(0.0, -self.day_realized_pnl / self.day_start_equity)


class RiskManager:
    """Applies per-trade sizing and portfolio limits to strategy signals."""

    def __init__(
        self,
        config: RiskConfig,
        markets: Mapping[str, MarketMeta] | None = None,
    ) -> None:
        self.config = config
        self.markets = dict(markets or {})
        self.state = RiskState()

    def market(self, symbol: str) -> MarketMeta:
        """Instrument limits for ``symbol``, permissive when unknown."""
        return self.markets.get(symbol) or MarketMeta.permissive(symbol)

    def update_equity(self, moment: datetime, equity: float) -> None:
        """Record a new equity reading, rolling the day over when needed.

        This drives both the daily loss limit and the maximum drawdown stop, so
        it must be called on every bar, not only when a trade closes.
        """
        state = self.state
        today = moment.date()
        if state.day != today:
            state.day = today
            state.day_start_equity = equity
            state.day_realized_pnl = 0.0

        state.current_equity = equity
        state.peak_equity = max(state.peak_equity, equity)

        if not state.halted and state.drawdown_pct >= self.config.max_drawdown_stop_pct:
            state.halted = True
            state.halt_reason = (
                f"max drawdown stop hit: {state.drawdown_pct:.1%} below the peak "
                f"of {state.peak_equity:,.2f} (limit {self.config.max_drawdown_stop_pct:.1%})"
            )
            logger.critical(state.halt_reason)

    def register_realized_pnl(self, amount: float) -> None:
        """Add a closed trade's net PnL to today's running total."""
        self.state.day_realized_pnl += amount

    def resume(self) -> None:
        """Clear the drawdown halt, for use after a manual review."""
        self.state.halted = False
        self.state.halt_reason = ""

    def evaluate(
        self,
        signal: Signal,
        *,
        equity: float,
        open_positions: Sequence[Position] = (),
    ) -> RiskDecision:
        """Approve or reject an entry signal and compute its final size.

        Args:
            signal: The entry intent produced by a strategy.
            equity: Current account equity.
            open_positions: Positions already held across the portfolio.

        Returns:
            A decision whose ``size`` is authoritative when ``approved`` is true.
        """
        blocked = self._blocking_limit(signal, equity, open_positions)
        if blocked is not None:
            return RiskDecision.reject(blocked)

        sizing = position_size(
            equity=equity,
            entry_price=signal.price,
            stop_price=signal.stop_loss,
            risk_pct=self.config.risk_per_trade_pct,
            max_notional_pct=self.config.max_notional_pct,
            market=self.market(signal.symbol),
        )
        if not sizing.is_tradable:
            return RiskDecision.reject(sizing.reason, sizing)

        portfolio_reason = self._portfolio_risk_exceeded(sizing, equity, open_positions)
        if portfolio_reason is not None:
            return RiskDecision.reject(portfolio_reason, sizing)

        note = "notional cap applied" if sizing.capped_by_notional else ""
        return RiskDecision.accept(sizing, note)

    def _blocking_limit(
        self,
        signal: Signal,
        equity: float,
        open_positions: Sequence[Position],
    ) -> str | None:
        """Return the first hard limit that forbids this entry, if any."""
        if self.state.halted:
            return self.state.halt_reason
        if equity <= 0:
            return "equity is exhausted"
        if self.state.day_loss_pct >= self.config.daily_loss_limit_pct:
            return (
                f"daily loss limit reached: {self.state.day_loss_pct:.1%} "
                f"(limit {self.config.daily_loss_limit_pct:.1%})"
            )
        if any(position.symbol == signal.symbol for position in open_positions):
            return f"a position in {signal.symbol} is already open"
        if len(open_positions) >= self.config.max_concurrent_positions:
            return (
                f"already holding {len(open_positions)} positions "
                f"(limit {self.config.max_concurrent_positions})"
            )
        same_side = sum(1 for position in open_positions if position.side is signal.side)
        if same_side >= self.config.max_positions_per_side:
            return (
                f"already holding {same_side} {signal.side.value} positions "
                f"(limit {self.config.max_positions_per_side})"
            )
        return None

    def _portfolio_risk_exceeded(
        self,
        sizing: SizingResult,
        equity: float,
        open_positions: Sequence[Position],
    ) -> str | None:
        """Check the combined open risk against ``max_portfolio_risk_pct``."""
        open_risk = sum(open_risk_amount(position) for position in open_positions)
        total = (open_risk + sizing.risk_amount) / equity
        if total > self.config.max_portfolio_risk_pct + 1e-12:
            return (
                f"portfolio risk would reach {total:.2%} "
                f"(limit {self.config.max_portfolio_risk_pct:.2%})"
            )
        return None


def open_risk_amount(position: Position) -> float:
    """Money still at risk in an open position, given where its stop sits now.

    Once the stop is above entry on a long, the position can no longer lose, so
    its contribution to portfolio risk is zero rather than negative.
    """
    per_unit = (position.entry_price - position.current_stop) * position.side.sign
    return max(0.0, per_unit) * position.size


@dataclass(slots=True)
class RejectionLog:
    """Counts of rejected signals by reason, surfaced in the run report."""

    entries: list[tuple[datetime, str, str]] = field(default_factory=list)

    def add(self, moment: datetime, symbol: str, reason: str) -> None:
        """Record one rejection."""
        self.entries.append((moment, symbol, reason))

    def summary(self) -> dict[str, int]:
        """Rejection counts keyed by reason."""
        counts: dict[str, int] = {}
        for _, _, reason in self.entries:
            counts[reason] = counts.get(reason, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.entries)


__all__ = [
    "RejectionLog",
    "RiskDecision",
    "RiskManager",
    "RiskState",
    "open_risk_amount",
]
