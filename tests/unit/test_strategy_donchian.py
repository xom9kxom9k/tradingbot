"""Scenario tests for the Donchian trend strategy (tech.md sections 7 and 15.1)."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingbot.core.enums import ExitReason, Regime, Side, SignalType
from tradingbot.core.exceptions import ConfigError
from tradingbot.core.models import BarContext, Position, Signal
from tradingbot.strategy import (
    DonchianTrendParams,
    DonchianTrendStrategy,
    Strategy,
    available,
    create_strategy,
    get_strategy_class,
)
from tradingbot.strategy.registry import register

EQUITY = 10_000.0

# Short lookbacks keep the synthetic fixtures small while exercising the same code.
FAST_PARAMS = DonchianTrendParams(
    entry_channel=5,
    exit_channel=3,
    trend_fast=5,
    trend_slow=10,
    atr_period=5,
    adx_period=5,
    adx_min=15.0,
    cooldown_bars=3,
    min_atr_pct=0.0001,
    max_atr_pct=0.5,
)


def candles(closes: list[float], spread: float = 0.5) -> pd.DataFrame:
    """Build a candle frame from a close path, with a constant intrabar range."""
    index = pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC", name="ts")
    close = np.array(closes, dtype=float)
    opens = np.empty_like(close)
    opens[0] = close[0]
    opens[1:] = close[:-1]
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, close) + spread,
            "low": np.minimum(opens, close) - spread,
            "close": close,
            "volume": np.full_like(close, 1_000.0),
        },
        index=index,
    )


def rising(bars: int = 60, step: float = 2.0, start: float = 100.0) -> pd.DataFrame:
    return candles([start + step * i for i in range(bars)])


def falling(bars: int = 60, step: float = 2.0, start: float = 220.0) -> pd.DataFrame:
    return candles([start - step * i for i in range(bars)])


def choppy_uptrend(bars: int = 60) -> pd.DataFrame:
    """An uptrend with regular pullbacks, so ADX stays well below its ceiling."""
    pattern = (2.5, 2.5, -3.0)
    closes: list[float] = []
    price = 100.0
    for i in range(bars):
        price += pattern[i % len(pattern)]
        closes.append(price)
    return candles(closes)


def flat(bars: int = 60, level: float = 100.0) -> pd.DataFrame:
    wobble = [level + (1.0 if i % 2 else -1.0) * 0.4 for i in range(bars)]
    return candles(wobble, spread=0.2)


def make_strategy(**overrides: object) -> DonchianTrendStrategy:
    params = FAST_PARAMS.model_copy(update=overrides) if overrides else FAST_PARAMS
    return DonchianTrendStrategy(params)


def evaluate(
    strategy: DonchianTrendStrategy,
    frame: pd.DataFrame,
    *,
    position: Position | None = None,
    bars_since_exit: int | None = None,
    index: int | None = None,
) -> Signal | None:
    """Run ``on_bar`` for a single bar, defaulting to the last one."""
    prepared = strategy.prepare(frame)
    position_index = len(prepared) - 1 if index is None else index
    ctx = BarContext(
        symbol="BTC/USDT",
        timeframe="4h",
        index=position_index,
        history=prepared.iloc[: position_index + 1],
        equity=EQUITY,
        position=position,
        bars_since_exit=bars_since_exit,
    )
    return strategy.on_bar(ctx)


def all_signals(strategy: DonchianTrendStrategy, frame: pd.DataFrame) -> list[Signal]:
    """Collect entry signals across the whole frame, ignoring position state."""
    prepared = strategy.prepare(frame)
    signals = []
    for i in range(len(prepared)):
        ctx = BarContext(
            symbol="BTC/USDT",
            timeframe="4h",
            index=i,
            history=prepared.iloc[: i + 1],
            equity=EQUITY,
        )
        signal = strategy.on_bar(ctx)
        if signal is not None:
            signals.append(signal)
    return signals


def open_position(
    side: Side = Side.LONG,
    entry_price: float = 100.0,
    stop: float = 90.0,
    **overrides: object,
) -> Position:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "side": side,
        "entry_ts": datetime(2024, 1, 1, tzinfo=UTC),
        "entry_price": entry_price,
        "size": 1.0,
        "initial_size": 1.0,
        "initial_stop": stop,
        "current_stop": stop,
        "highest_price": entry_price,
        "lowest_price": entry_price,
    }
    values.update(overrides)
    return Position(**values)  # type: ignore[arg-type]


class TestParams:
    def test_defaults_match_the_specification(self) -> None:
        params = DonchianTrendParams()
        assert params.entry_channel == 20
        assert params.exit_channel == 10
        assert params.atr_stop_mult == 2.5
        assert params.take_profit_r == [2.0]

    def test_fast_ema_must_be_shorter_than_slow(self) -> None:
        with pytest.raises(ValueError, match="trend_fast"):
            DonchianTrendParams(trend_fast=200, trend_slow=50)

    def test_atr_band_must_be_ordered(self) -> None:
        with pytest.raises(ValueError, match="min_atr_pct"):
            DonchianTrendParams(min_atr_pct=0.2, max_atr_pct=0.1)

    def test_take_profit_levels_must_ascend(self) -> None:
        with pytest.raises(ValueError, match="ascending"):
            DonchianTrendParams(take_profit_r=[3.0, 2.0])

    def test_take_profit_levels_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            DonchianTrendParams(take_profit_r=[0.0])

    def test_unknown_parameter_is_rejected(self) -> None:
        with pytest.raises(ConfigError, match="invalid parameters"):
            DonchianTrendStrategy.from_params({"nonsense": 1})

    def test_params_are_frozen(self) -> None:
        with pytest.raises(ValueError, match="frozen"):
            DonchianTrendParams().entry_channel = 5  # type: ignore[misc]


class TestPrepare:
    def test_adds_every_required_column(self) -> None:
        prepared = make_strategy().prepare(rising())
        for column in (
            "ema_fast",
            "ema_slow",
            "atr",
            "atr_pct",
            "adx",
            "dc_upper",
            "dc_lower",
            "dc_mid",
            "dc_exit_upper",
            "dc_exit_lower",
            "regime",
        ):
            assert column in prepared.columns

    def test_input_frame_is_not_mutated(self) -> None:
        frame = rising()
        make_strategy().prepare(frame)
        assert list(frame.columns) == ["open", "high", "low", "close", "volume"]

    def test_regime_is_up_in_an_uptrend(self) -> None:
        prepared = make_strategy().prepare(rising())
        assert prepared["regime"].iloc[-1] == Regime.UP.value

    def test_regime_is_down_in_a_downtrend(self) -> None:
        prepared = make_strategy().prepare(falling())
        assert prepared["regime"].iloc[-1] == Regime.DOWN.value

    def test_regime_is_flat_during_warmup(self) -> None:
        prepared = make_strategy().prepare(rising())
        assert prepared["regime"].iloc[0] == Regime.FLAT.value

    def test_warmup_period_covers_the_slowest_indicator(self) -> None:
        assert DonchianTrendStrategy().warmup_period() >= 200


class TestEntryScenarios:
    def test_uptrend_breakout_produces_a_long(self) -> None:
        signals = all_signals(make_strategy(), rising())
        assert signals
        assert all(signal.side is Side.LONG for signal in signals)
        assert all(signal.signal_type is SignalType.ENTRY for signal in signals)

    def test_downtrend_breakout_produces_a_short(self) -> None:
        signals = all_signals(make_strategy(), falling())
        assert signals
        assert all(signal.side is Side.SHORT for signal in signals)

    def test_sideways_market_produces_nothing(self) -> None:
        assert all_signals(make_strategy(), flat()) == []

    def test_adx_below_threshold_blocks_entry(self) -> None:
        frame = choppy_uptrend()
        strongest = float(make_strategy().prepare(frame)["adx"].max())
        assert strongest < 100.0, "fixture must not saturate the indicator"

        assert all_signals(make_strategy(adx_min=strongest - 1.0), frame)
        assert all_signals(make_strategy(adx_min=strongest + 1.0), frame) == []

    def test_atr_below_the_floor_blocks_entry(self) -> None:
        assert all_signals(make_strategy(min_atr_pct=0.9), rising()) == []

    def test_atr_above_the_ceiling_blocks_entry(self) -> None:
        assert all_signals(make_strategy(max_atr_pct=1e-6), rising()) == []

    def test_cooldown_blocks_a_quick_re_entry(self) -> None:
        strategy = make_strategy()
        frame = rising()
        assert evaluate(strategy, frame, bars_since_exit=0) is None
        assert evaluate(strategy, frame, bars_since_exit=2) is None
        assert evaluate(strategy, frame, bars_since_exit=3) is not None

    def test_no_entry_while_a_position_is_open(self) -> None:
        signal = evaluate(make_strategy(), rising(), position=open_position())
        assert signal is None or signal.signal_type is SignalType.EXIT

    def test_no_signal_during_warmup(self) -> None:
        assert evaluate(make_strategy(), rising(bars=8)) is None


class TestEntrySignalContents:
    @pytest.fixture
    def long_signal(self) -> Signal:
        signal = evaluate(make_strategy(), rising())
        assert signal is not None
        return signal

    def test_stop_is_atr_multiples_below_entry(self, long_signal: Signal) -> None:
        prepared = make_strategy().prepare(rising())
        atr_value = float(prepared["atr"].iloc[-1])
        assert long_signal.price - long_signal.stop_loss == pytest.approx(2.5 * atr_value)

    def test_take_profit_sits_two_r_away(self, long_signal: Signal) -> None:
        target = long_signal.take_profits[0]
        assert target.r_multiple == 2.0
        assert target.fraction == 0.5
        assert target.price == pytest.approx(long_signal.price + 2.0 * long_signal.r_value)

    def test_size_risks_exactly_one_percent(self, long_signal: Signal) -> None:
        risked = long_signal.size * long_signal.r_value
        assert risked == pytest.approx(EQUITY * 0.01)

    def test_reason_is_human_readable(self, long_signal: Signal) -> None:
        assert "Donchian-5 breakout up" in long_signal.reason

    def test_indicator_snapshot_is_attached(self, long_signal: Signal) -> None:
        assert {"adx", "atr", "close", "ema_fast", "ema_slow"} <= set(long_signal.indicators)

    def test_short_mirrors_the_long_geometry(self) -> None:
        signal = evaluate(make_strategy(), falling())
        assert signal is not None
        assert signal.stop_loss > signal.price
        assert signal.take_profits[0].price < signal.price


class TestExitScenarios:
    def test_long_exits_when_price_breaks_the_exit_channel(self) -> None:
        path = [100.0 + 2.0 * i for i in range(40)] + [178.0, 168.0, 158.0]
        signal = evaluate(make_strategy(), candles(path), position=open_position())
        assert signal is not None
        assert signal.signal_type is SignalType.EXIT
        assert signal.exit_reason is ExitReason.CHANNEL_EXIT

    def test_short_exits_when_price_breaks_the_exit_channel(self) -> None:
        path = [220.0 - 2.0 * i for i in range(40)] + [142.0, 152.0, 162.0]
        position = open_position(side=Side.SHORT, entry_price=150.0, stop=160.0)
        signal = evaluate(make_strategy(), candles(path), position=position)
        assert signal is not None
        assert signal.exit_reason is ExitReason.CHANNEL_EXIT

    def test_regime_flip_closes_a_long(self) -> None:
        # Rally, sharp break, then a quiet shelf: the regime has flipped while
        # price still holds above the recent lows, isolating the flip rule.
        rally = [100.0 + 2.0 * i for i in range(30)]
        break_down = [158.0 - 10.0 * i for i in range(1, 11)]
        shelf = [58.0 + 0.1 * (i % 2) for i in range(8)]
        strategy = make_strategy()
        prepared = strategy.prepare(candles(rally + break_down + shelf))
        assert prepared["regime"].iloc[-1] == Regime.DOWN.value
        assert prepared["close"].iloc[-1] >= prepared["dc_exit_lower"].iloc[-1]

        ctx = BarContext(
            symbol="BTC/USDT",
            timeframe="4h",
            index=len(prepared) - 1,
            history=prepared,
            equity=EQUITY,
            position=open_position(),
        )
        signal = strategy.on_bar(ctx)
        assert signal is not None
        assert signal.exit_reason is ExitReason.REGIME_FLIP

    def test_no_exit_while_the_trend_holds(self) -> None:
        assert evaluate(make_strategy(), rising(), position=open_position()) is None

    def test_exit_signal_carries_the_full_position_size(self) -> None:
        path = [100.0 + 2.0 * i for i in range(40)] + [178.0, 168.0, 158.0]
        position = open_position(size=3.5, initial_size=3.5)
        signal = evaluate(make_strategy(), candles(path), position=position)
        assert signal is not None
        assert signal.size == pytest.approx(3.5)


class TestStopManagement:
    def _context(self, frame: pd.DataFrame, position: Position) -> BarContext:
        prepared = make_strategy().prepare(frame)
        return BarContext(
            symbol="BTC/USDT",
            timeframe="4h",
            index=len(prepared) - 1,
            history=prepared,
            equity=EQUITY,
            position=position,
        )

    def test_no_position_means_no_update(self) -> None:
        frame = rising()
        prepared = make_strategy().prepare(frame)
        ctx = BarContext("BTC/USDT", "4h", len(prepared) - 1, prepared, EQUITY)
        assert make_strategy().update_stop(ctx) is None

    def test_breakeven_after_one_r(self) -> None:
        frame = rising()
        last_close = float(frame["close"].iloc[-1])
        position = open_position(entry_price=last_close - 20.0, stop=last_close - 40.0)
        new_stop = make_strategy().update_stop(self._context(frame, position))
        assert new_stop == pytest.approx(position.entry_price)

    def test_breakeven_does_not_trigger_below_one_r(self) -> None:
        frame = rising()
        last_close = float(frame["close"].iloc[-1])
        position = open_position(entry_price=last_close - 1.0, stop=last_close - 40.0)
        assert make_strategy().update_stop(self._context(frame, position)) is None

    def test_trailing_starts_after_the_first_target(self) -> None:
        frame = rising()
        prepared = make_strategy().prepare(frame)
        atr_value = float(prepared["atr"].iloc[-1])
        position = open_position(
            entry_price=100.0,
            stop=100.0,
            highest_price=200.0,
            filled_take_profits=1,
            breakeven_moved=True,
        )
        new_stop = make_strategy().update_stop(self._context(frame, position))
        assert new_stop == pytest.approx(200.0 - 3.0 * atr_value)

    def test_trailing_never_moves_against_a_long(self) -> None:
        frame = rising()
        position = open_position(
            entry_price=100.0,
            stop=195.0,
            highest_price=200.0,
            filled_take_profits=1,
        )
        assert make_strategy().update_stop(self._context(frame, position)) is None

    def test_trailing_mirrors_for_shorts(self) -> None:
        frame = falling()
        prepared = make_strategy().prepare(frame)
        atr_value = float(prepared["atr"].iloc[-1])
        position = open_position(
            side=Side.SHORT,
            entry_price=200.0,
            stop=220.0,
            lowest_price=120.0,
            filled_take_profits=1,
        )
        new_stop = make_strategy().update_stop(self._context(frame, position))
        assert new_stop == pytest.approx(120.0 + 3.0 * atr_value)

    def test_no_update_during_warmup(self) -> None:
        frame = rising(bars=6)
        prepared = make_strategy().prepare(frame)
        ctx = BarContext(
            "BTC/USDT", "4h", len(prepared) - 1, prepared, EQUITY, position=open_position()
        )
        assert make_strategy().update_stop(ctx) is None


class TestRegistry:
    def test_default_strategy_is_registered(self) -> None:
        assert "donchian_trend" in available()
        assert get_strategy_class("donchian_trend") is DonchianTrendStrategy

    def test_create_from_config_mapping(self) -> None:
        strategy = create_strategy("donchian_trend", {"entry_channel": 30})
        assert isinstance(strategy, DonchianTrendStrategy)
        assert strategy.params.entry_channel == 30

    def test_risk_fraction_is_forwarded(self) -> None:
        strategy = create_strategy("donchian_trend", {}, risk_per_trade_pct=0.02)
        assert strategy.describe()["risk_per_trade_pct"] == 0.02

    def test_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(ConfigError, match="registered strategies: donchian_trend"):
            create_strategy("does_not_exist")

    def test_nameless_strategy_cannot_register(self) -> None:
        class Nameless(Strategy):
            def warmup_period(self) -> int:
                return 1

            def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
                return frame

            def on_bar(self, ctx: BarContext) -> Signal | None:
                return None

        with pytest.raises(ConfigError, match="non-empty class attribute"):
            register(Nameless)

    def test_duplicate_name_is_rejected(self) -> None:
        class Clashing(DonchianTrendStrategy):
            name = "donchian_trend"

        with pytest.raises(ConfigError, match="already registered"):
            register(Clashing)

    def test_describe_exposes_every_parameter(self) -> None:
        described = DonchianTrendStrategy().describe()
        assert described["name"] == "donchian_trend"
        assert described["entry_channel"] == 20

    def test_repr_is_informative(self) -> None:
        assert "donchian_trend" in repr(DonchianTrendStrategy())


class TestInfrastructureIndependence:
    """The strategy package must not reach into infrastructure (DoD of stage 5)."""

    FORBIDDEN = ("tradingbot.data", "tradingbot.storage", "tradingbot.notify", "tradingbot.live")

    def test_no_infrastructure_imports(self) -> None:
        package = Path(__file__).resolve().parents[2] / "src" / "tradingbot" / "strategy"
        offenders: list[str] = []
        for module in package.glob("*.py"):
            tree = ast.parse(module.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                offenders.extend(
                    f"{module.name}: {name}" for name in names if name.startswith(self.FORBIDDEN)
                )
        assert offenders == []
