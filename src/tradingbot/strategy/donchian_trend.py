"""Donchian trend breakout with ATR risk management (tech.md section 7).

The rules, in one paragraph: trade only in the direction of the EMA regime, enter
when price closes beyond the Donchian channel of the previous ``entry_channel``
bars, require a minimum ADX and a sane ATR-to-price ratio, risk a fixed fraction
of equity with the stop placed ``atr_stop_mult`` ATR away, take partial profit at
+2R, then trail the remainder with a chandelier stop until either the opposite
channel or a regime flip closes the trade.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tradingbot.core.enums import ExitReason, Regime, Side, SignalType
from tradingbot.core.exceptions import ConfigError
from tradingbot.core.models import BarContext, Position, Signal, TakeProfit
from tradingbot.indicators import adx, atr, donchian, ema
from tradingbot.strategy.base import Strategy
from tradingbot.strategy.registry import register


class DonchianTrendParams(BaseModel):
    """Validated parameter block for :class:`DonchianTrendStrategy`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entry_channel: int = Field(default=20, ge=2)
    exit_channel: int = Field(default=10, ge=2)
    trend_fast: int = Field(default=50, ge=2)
    trend_slow: int = Field(default=200, ge=3)
    atr_period: int = Field(default=14, ge=2)
    adx_period: int = Field(default=14, ge=2)
    adx_min: float = Field(default=20.0, ge=0.0, le=100.0)
    atr_stop_mult: float = Field(default=2.5, gt=0.0)
    atr_trail_mult: float = Field(default=3.0, gt=0.0)
    breakeven_at_r: float = Field(default=1.0, ge=0.0)
    take_profit_r: list[float] = Field(default_factory=lambda: [2.0])
    take_profit_fraction: float = Field(default=0.5, gt=0.0, le=1.0)
    min_atr_pct: float = Field(default=0.005, ge=0.0)
    max_atr_pct: float = Field(default=0.12, gt=0.0)
    cooldown_bars: int = Field(default=3, ge=0)

    @model_validator(mode="after")
    def _coherent(self) -> DonchianTrendParams:
        if self.trend_fast >= self.trend_slow:
            raise ValueError("trend_fast must be shorter than trend_slow")
        if self.min_atr_pct >= self.max_atr_pct:
            raise ValueError("min_atr_pct must be below max_atr_pct")
        if any(level <= 0 for level in self.take_profit_r):
            raise ValueError("take_profit_r levels must be positive")
        if self.take_profit_r != sorted(self.take_profit_r):
            raise ValueError("take_profit_r levels must be listed in ascending order")
        return self


@register
class DonchianTrendStrategy(Strategy):
    """Symmetric trend-following breakout strategy.

    Args:
        params: Rule parameters; defaults match ``configs/strategy_donchian_trend.yaml``.
        risk_per_trade_pct: Fraction of equity risked per trade, used to turn the
            ATR stop distance into a position size. The portfolio-level limits of
            section 7.9 are applied later by the risk manager.
    """

    name = "donchian_trend"

    def __init__(
        self,
        params: DonchianTrendParams | None = None,
        risk_per_trade_pct: float = 0.01,
        fee_buffer_pct: float = 0.0,
    ) -> None:
        self.params = params or DonchianTrendParams()
        self.risk_per_trade_pct = risk_per_trade_pct
        self.fee_buffer_pct = fee_buffer_pct

    @classmethod
    def from_params(
        cls,
        params: dict[str, Any] | None = None,
        risk_per_trade_pct: float = 0.01,
        fee_buffer_pct: float = 0.0,
    ) -> DonchianTrendStrategy:
        """Build the strategy from a raw configuration mapping."""
        try:
            validated = DonchianTrendParams.model_validate(params or {})
        except ValueError as exc:
            raise ConfigError(f"invalid parameters for {cls.name}: {exc}") from exc
        return cls(
            validated,
            risk_per_trade_pct=risk_per_trade_pct,
            fee_buffer_pct=fee_buffer_pct,
        )

    def describe(self) -> dict[str, Any]:
        """Full parameter snapshot, stored in run metadata."""
        return {
            "name": self.name,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "fee_buffer_pct": self.fee_buffer_pct,
            **self.params.model_dump(),
        }

    def warmup_period(self) -> int:
        """Longest indicator lookback, with room for the ADX double smoothing."""
        params = self.params
        return max(
            params.trend_slow,
            params.entry_channel + 1,
            params.exit_channel + 1,
            params.atr_period * 2,
            params.adx_period * 3,
        )

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add every indicator column the rules depend on."""
        params = self.params
        high, low, close = frame["high"], frame["low"], frame["close"]
        prepared = frame.copy()

        prepared["ema_fast"] = ema(close, params.trend_fast)
        prepared["ema_slow"] = ema(close, params.trend_slow)
        prepared["atr"] = atr(high, low, close, params.atr_period)
        prepared["atr_pct"] = prepared["atr"] / close
        prepared["adx"] = adx(high, low, close, params.adx_period)

        entry_channel = donchian(high, low, params.entry_channel)
        prepared["dc_upper"] = entry_channel["dc_upper"]
        prepared["dc_lower"] = entry_channel["dc_lower"]
        prepared["dc_mid"] = entry_channel["dc_mid"]

        exit_channel = donchian(high, low, params.exit_channel)
        prepared["dc_exit_upper"] = exit_channel["dc_upper"]
        prepared["dc_exit_lower"] = exit_channel["dc_lower"]

        prepared["regime"] = _regime(prepared)
        return prepared

    def on_bar(self, ctx: BarContext) -> Signal | None:
        """Evaluate exit rules for an open position, otherwise look for an entry."""
        row = ctx.bar
        if _has_nan(row):
            return None
        if ctx.position is not None:
            return self._exit_signal(ctx, row)
        return self._entry_signal(ctx, row)

    def update_stop(self, ctx: BarContext) -> float | None:
        """Move the stop to breakeven at +1R, then trail it after the first target.

        Returns:
            A strictly better stop level, or ``None`` when the current one stands.
        """
        position = ctx.position
        if position is None:
            return None
        row = ctx.bar
        if _has_nan(row):
            return None

        candidate: float | None = None
        if position.filled_take_profits > 0:
            candidate = self._chandelier(position, float(row["atr"]))
        elif (
            not position.breakeven_moved
            and position.excursion_r(float(row["close"])) >= self.params.breakeven_at_r
        ):
            # Breakeven means "no loss after costs", so the level sits a round
            # trip of fees beyond the entry price.
            candidate = position.entry_price * (1.0 + self.fee_buffer_pct * position.side.sign)

        if candidate is None:
            return None
        improved = (
            candidate > position.current_stop
            if position.side is Side.LONG
            else candidate < position.current_stop
        )
        return candidate if improved else None

    def _chandelier(self, position: Position, atr_value: float) -> float:
        """Trailing stop anchored to the extreme reached while in the position."""
        distance = self.params.atr_trail_mult * atr_value
        if position.side is Side.LONG:
            return position.highest_price - distance
        return position.lowest_price + distance

    def _entry_signal(self, ctx: BarContext, row: pd.Series) -> Signal | None:
        """Apply the section 7.5 entry checklist."""
        params = self.params
        if ctx.bars_since_exit is not None and ctx.bars_since_exit < params.cooldown_bars:
            return None

        atr_value = float(row["atr"])
        close = float(row["close"])
        atr_ratio = float(row["atr_pct"])
        adx_value = float(row["adx"])
        regime = Regime(row["regime"])

        if not params.min_atr_pct <= atr_ratio <= params.max_atr_pct:
            return None
        if adx_value < params.adx_min:
            return None

        if regime is Regime.UP and close > float(row["dc_upper"]):
            side = Side.LONG
        elif regime is Regime.DOWN and close < float(row["dc_lower"]):
            side = Side.SHORT
        else:
            return None

        stop_distance = params.atr_stop_mult * atr_value
        if stop_distance <= 0:
            return None
        stop_loss = close - stop_distance if side is Side.LONG else close + stop_distance
        size = ctx.equity * self.risk_per_trade_pct / stop_distance
        if size <= 0:
            return None

        channel = float(row["dc_upper"] if side is Side.LONG else row["dc_lower"])
        reason = (
            f"Donchian-{params.entry_channel} breakout "
            f"{'up' if side is Side.LONG else 'down'} through {channel:.6g}"
        )
        return Signal(
            signal_type=SignalType.ENTRY,
            side=side,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            bar_ts=ctx.ts,
            price=close,
            stop_loss=stop_loss,
            take_profits=self._take_profits(close, stop_distance, side),
            size=size,
            risk_pct=self.risk_per_trade_pct,
            reason=reason,
            indicators=_snapshot(row),
        )

    def _take_profits(self, entry: float, r_value: float, side: Side) -> list[TakeProfit]:
        """Build the partial profit ladder described in section 7.6."""
        fraction = self.params.take_profit_fraction
        levels: list[TakeProfit] = []
        for multiple in self.params.take_profit_r:
            offset = multiple * r_value * side.sign
            levels.append(TakeProfit(price=entry + offset, r_multiple=multiple, fraction=fraction))
        return levels

    def _exit_signal(self, ctx: BarContext, row: pd.Series) -> Signal | None:
        """Close-based exits: the opposite channel, then a regime flip."""
        position = ctx.position
        assert position is not None
        close = float(row["close"])
        regime = Regime(row["regime"])

        reason: str | None = None
        exit_reason: ExitReason | None = None

        if position.side is Side.LONG:
            channel = float(row["dc_exit_lower"])
            if close < channel:
                exit_reason = ExitReason.CHANNEL_EXIT
                reason = f"close below Donchian-{self.params.exit_channel} low {channel:.6g}"
            elif regime is Regime.DOWN:
                exit_reason = ExitReason.REGIME_FLIP
                reason = "regime flipped to DOWN"
        else:
            channel = float(row["dc_exit_upper"])
            if close > channel:
                exit_reason = ExitReason.CHANNEL_EXIT
                reason = f"close above Donchian-{self.params.exit_channel} high {channel:.6g}"
            elif regime is Regime.UP:
                exit_reason = ExitReason.REGIME_FLIP
                reason = "regime flipped to UP"

        if exit_reason is None or reason is None:
            return None

        return Signal(
            signal_type=SignalType.EXIT,
            side=position.side,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            bar_ts=ctx.ts,
            price=close,
            stop_loss=position.current_stop,
            size=position.size,
            risk_pct=0.0,
            reason=reason,
            indicators=_snapshot(row),
            exit_reason=exit_reason,
        )


def _regime(frame: pd.DataFrame) -> pd.Series:
    """Classify each bar as UP, DOWN or FLAT (tech.md section 7.4)."""
    fast, slow, close = frame["ema_fast"], frame["ema_slow"], frame["close"]
    up = (fast > slow) & (close > slow)
    down = (fast < slow) & (close < slow)
    regime = pd.Series(Regime.FLAT.value, index=frame.index, dtype="object")
    regime[up] = Regime.UP.value
    regime[down] = Regime.DOWN.value
    regime[slow.isna() | fast.isna()] = Regime.FLAT.value
    return regime


REQUIRED_COLUMNS = ("atr", "atr_pct", "adx", "dc_upper", "dc_lower", "close")


def _has_nan(row: pd.Series) -> bool:
    """True while any indicator the rules need is still warming up."""
    return bool(pd.isna([row.get(column) for column in REQUIRED_COLUMNS]).any())


def _snapshot(row: pd.Series) -> dict[str, float]:
    """Indicator values recorded on the signal for later audit."""
    keys = (
        "close",
        "ema_fast",
        "ema_slow",
        "atr",
        "atr_pct",
        "adx",
        "dc_upper",
        "dc_lower",
        "dc_exit_upper",
        "dc_exit_lower",
    )
    return {key: float(row[key]) for key in keys if key in row.index and not pd.isna(row[key])}
