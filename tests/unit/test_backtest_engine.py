"""Engine behaviour: bar ordering, costs, exits, determinism.

Most tests drive the engine with a scripted strategy rather than the real one,
so a failure points at the engine instead of at the trading rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config.models import AppConfig, CostsConfig, ExchangeConfig, RiskConfig
from tradingbot.core.enums import ExitReason, Side, SignalType
from tradingbot.core.exceptions import BacktestError
from tradingbot.core.models import BarContext, MarketMeta, Signal, TakeProfit
from tradingbot.strategy.base import Strategy

SYMBOL = "BTC/USDT"


def frame_from(rows: list[tuple[float, float, float, float]], atr: float = 1.0) -> pd.DataFrame:
    """Build an OHLCV frame from ``(open, high, low, close)`` tuples."""
    index = pd.date_range("2026-01-01", periods=len(rows), freq="4h", tz="UTC", name="ts")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=index)
    frame["volume"] = 1_000.0
    frame["atr"] = atr
    return frame


def flat_frame(bars: int, price: float = 100.0, atr: float = 1.0) -> pd.DataFrame:
    return frame_from([(price, price, price, price)] * bars, atr=atr)


@dataclass
class EntrySpec:
    """What the scripted strategy should ask for on a given bar."""

    side: Side = Side.LONG
    stop_distance: float = 10.0
    targets: list[tuple[float, float]] = field(default_factory=list)
    size: float = 1.0


class ScriptedStrategy(Strategy):
    """Emits predetermined intents so engine mechanics can be tested in isolation."""

    name = "scripted"

    def __init__(
        self,
        entries: dict[int, EntrySpec] | None = None,
        exits: set[int] | None = None,
        stops: dict[int, float] | None = None,
    ) -> None:
        self.entries = entries or {}
        self.exits = exits or set()
        self.stops = stops or {}

    def warmup_period(self) -> int:
        return 0

    def prepare(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if "atr" not in prepared.columns:
            prepared["atr"] = 1.0
        return prepared

    def update_stop(self, ctx: BarContext) -> float | None:
        return self.stops.get(ctx.index)

    def on_bar(self, ctx: BarContext) -> Signal | None:
        close = float(ctx.bar["close"])
        if ctx.position is not None:
            if ctx.index not in self.exits:
                return None
            return Signal(
                signal_type=SignalType.EXIT,
                side=ctx.position.side,
                symbol=ctx.symbol,
                timeframe=ctx.timeframe,
                bar_ts=ctx.ts,
                price=close,
                stop_loss=ctx.position.current_stop,
                size=ctx.position.size,
                risk_pct=0.0,
                exit_reason=ExitReason.CHANNEL_EXIT,
                reason="scripted exit",
            )

        spec = self.entries.get(ctx.index)
        if spec is None:
            return None
        stop = close - spec.stop_distance if spec.side is Side.LONG else close + spec.stop_distance
        targets = [
            TakeProfit(
                price=close + multiple * spec.stop_distance * spec.side.sign,
                r_multiple=multiple,
                fraction=fraction,
            )
            for multiple, fraction in spec.targets
        ]
        return Signal(
            signal_type=SignalType.ENTRY,
            side=spec.side,
            symbol=ctx.symbol,
            timeframe=ctx.timeframe,
            bar_ts=ctx.ts,
            price=close,
            stop_loss=stop,
            take_profits=targets,
            size=spec.size,
            risk_pct=0.01,
            reason="scripted entry",
        )


def make_config(**sections: object) -> AppConfig:
    """An AppConfig with frictionless defaults, overridable per test."""
    defaults: dict[str, object] = {
        "exchange": ExchangeConfig(symbols=[SYMBOL], timeframe="4h"),
        "costs": CostsConfig(
            taker_fee=0.0,
            maker_fee=0.0,
            slippage_model="fixed_bps",
            slippage_bps=0.0,
            apply_funding=False,
        ),
        "risk": RiskConfig(risk_per_trade_pct=0.01, max_notional_pct=1.0),
    }
    defaults.update(sections)
    return AppConfig(**defaults)  # type: ignore[arg-type]


def run(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
    config: AppConfig | None = None,
    markets: dict[str, MarketMeta] | None = None,
):
    engine = BacktestEngine(config or make_config(), strategy, markets=markets)
    return engine.run(data)


class TestNoLookAhead:
    """The core guarantee of section 8.2."""

    def test_signal_is_filled_at_the_next_open_not_this_close(self) -> None:
        # Bar 1 closes at 100; bar 2 opens 50% higher. A look-ahead bug would
        # fill at 100 and book a windfall.
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (150.0, 155.0, 149.0, 152.0),
            (152.0, 153.0, 151.0, 152.0),
        ]
        result = run(
            ScriptedStrategy(entries={1: EntrySpec(stop_distance=50.0)}), {SYMBOL: frame_from(rows)}
        )
        assert len(result.trades) == 1
        assert result.trades.iloc[0]["entry_price"] == pytest.approx(150.0)

    def test_entry_timestamp_is_the_bar_after_the_signal(self) -> None:
        frame = flat_frame(5)
        result = run(ScriptedStrategy(entries={1: EntrySpec()}), {SYMBOL: frame})
        assert result.trades.iloc[0]["entry_ts"] == frame.index[2]

    def test_a_signal_on_the_last_bar_never_trades(self) -> None:
        frame = flat_frame(4)
        result = run(ScriptedStrategy(entries={3: EntrySpec()}), {SYMBOL: frame})
        assert result.trades.empty
        assert len(result.signals) == 1

    def test_exit_also_waits_for_the_next_open(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (120.0, 120.0, 120.0, 120.0),
            (120.0, 120.0, 120.0, 120.0),
        ]
        strategy = ScriptedStrategy(entries={0: EntrySpec(stop_distance=90.0)}, exits={3})
        result = run(strategy, {SYMBOL: frame_from(rows)})
        assert result.trades.iloc[0]["exit_price"] == pytest.approx(120.0)


class TestExits:
    def test_stop_is_hit_intrabar(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 101.0, 99.0, 100.0),
            (100.0, 101.0, 85.0, 95.0),
            (95.0, 96.0, 94.0, 95.0),
        ]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}), {SYMBOL: frame_from(rows)}
        )
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == ExitReason.STOP.value
        assert trade["exit_price"] == pytest.approx(90.0)

    def test_stop_beats_target_inside_the_same_bar(self) -> None:
        # This bar touches both 120 (target) and 90 (stop). The stop must win.
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 125.0, 85.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        strategy = ScriptedStrategy(
            entries={0: EntrySpec(stop_distance=10.0, targets=[(2.0, 1.0)])}
        )
        result = run(strategy, {SYMBOL: frame_from(rows)})
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == ExitReason.STOP.value
        assert trade["pnl_net"] < 0

    def test_gap_through_the_stop_fills_at_the_open(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (70.0, 72.0, 68.0, 70.0),
        ]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}), {SYMBOL: frame_from(rows)}
        )
        assert result.trades.iloc[0]["exit_price"] == pytest.approx(70.0)

    def test_partial_target_leaves_the_rest_running(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 121.0, 99.0, 120.0),
            (105.0, 105.0, 105.0, 105.0),
        ]
        strategy = ScriptedStrategy(
            entries={0: EntrySpec(stop_distance=10.0, targets=[(2.0, 0.5)])}
        )
        result = run(strategy, {SYMBOL: frame_from(rows)})
        trade = result.trades.iloc[0]
        # Half of the 10 units leaves at the 120 target, the rest is flattened at
        # the final close of 105, so the journal shows the weighted average.
        assert trade["size"] == pytest.approx(10.0)
        assert trade["exit_price"] == pytest.approx(112.5)
        assert trade["pnl_gross"] == pytest.approx(125.0)
        assert trade["exit_reason"] == ExitReason.EOD.value

    def test_a_full_target_closes_the_position_as_tp1(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 121.0, 99.0, 120.0),
            (105.0, 105.0, 105.0, 105.0),
        ]
        strategy = ScriptedStrategy(
            entries={0: EntrySpec(stop_distance=10.0, targets=[(2.0, 1.0)])}
        )
        result = run(strategy, {SYMBOL: frame_from(rows)})
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == ExitReason.TP1.value
        assert trade["exit_price"] == pytest.approx(120.0)
        assert trade["pnl_r"] == pytest.approx(2.0)

    def test_trailing_stop_after_a_partial_is_reported_as_trail(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 121.0, 99.0, 120.0),
            (120.0, 120.0, 104.0, 105.0),
            (105.0, 105.0, 105.0, 105.0),
        ]
        strategy = ScriptedStrategy(
            entries={0: EntrySpec(stop_distance=10.0, targets=[(2.0, 0.5)])},
            stops={2: 110.0},
        )
        result = run(strategy, {SYMBOL: frame_from(rows)})
        assert result.trades.iloc[0]["exit_reason"] == ExitReason.TRAIL.value

    def test_short_position_stop_and_pnl(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 115.0, 99.0, 112.0),
        ]
        strategy = ScriptedStrategy(entries={0: EntrySpec(side=Side.SHORT, stop_distance=10.0)})
        result = run(strategy, {SYMBOL: frame_from(rows)})
        trade = result.trades.iloc[0]
        assert trade["side"] == Side.SHORT.value
        assert trade["exit_price"] == pytest.approx(110.0)
        assert trade["pnl_net"] < 0

    def test_open_position_is_flattened_at_the_end_of_data(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 3 + [(130.0, 130.0, 130.0, 130.0)]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}), {SYMBOL: frame_from(rows)}
        )
        trade = result.trades.iloc[0]
        assert trade["exit_reason"] == ExitReason.EOD.value
        assert trade["pnl_net"] > 0

    def test_stop_update_is_applied_from_the_following_bar(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 96.0, 100.0),
            (100.0, 100.0, 96.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        # The stop moves to 98 at the close of bar 2, so bar 2's own low of 96
        # must not trigger it, while bar 3's identical low must.
        strategy = ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}, stops={2: 98.0})
        result = run(strategy, {SYMBOL: frame_from(rows)})
        trade = result.trades.iloc[0]
        assert trade["exit_ts"] == frame_from(rows).index[3]
        assert trade["exit_price"] == pytest.approx(98.0)


class TestCosts:
    def test_fees_are_charged_on_both_sides(self) -> None:
        config = make_config(
            costs=CostsConfig(
                taker_fee=0.001, slippage_model="fixed_bps", slippage_bps=0.0, apply_funding=False
            )
        )
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
            config,
        )
        # 1% of 10 000 over a 50-wide stop is 2 units, so 200 notional per side.
        trade = result.trades.iloc[0]
        assert trade["fees"] == pytest.approx(0.4)
        assert trade["pnl_net"] == pytest.approx(-0.4)

    def test_slippage_widens_the_round_trip(self) -> None:
        config = make_config(
            costs=CostsConfig(
                taker_fee=0.0, slippage_model="fixed_bps", slippage_bps=10.0, apply_funding=False
            )
        )
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
            config,
        )
        trade = result.trades.iloc[0]
        assert trade["entry_price"] == pytest.approx(100.1)
        assert trade["exit_price"] == pytest.approx(99.9)

    def test_funding_is_charged_to_a_long(self) -> None:
        config = make_config(
            costs=CostsConfig(
                taker_fee=0.0,
                slippage_model="fixed_bps",
                slippage_bps=0.0,
                funding_rate_8h=0.001,
                apply_funding=True,
            )
        )
        rows = [(100.0, 100.0, 100.0, 100.0)] * 8
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
            config,
        )
        assert result.trades.iloc[0]["funding"] > 0

    def test_slippage_actually_paid_is_measured(self) -> None:
        config = make_config(
            costs=CostsConfig(
                taker_fee=0.0, slippage_model="fixed_bps", slippage_bps=10.0, apply_funding=False
            )
        )
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
            config,
        )
        # Two units, 0.1 of price concession on each of the two sides.
        assert result.slippage_cost == pytest.approx(0.4)

    def test_no_slippage_model_costs_nothing(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
        )
        assert result.slippage_cost == pytest.approx(0.0)

    def test_atr_slippage_scales_with_volatility(self) -> None:
        config = make_config(
            costs=CostsConfig(
                taker_fee=0.0, slippage_model="atr", slippage_atr_mult=0.5, apply_funding=False
            )
        )
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows, atr=2.0)},
            config,
        )
        assert result.trades.iloc[0]["entry_price"] == pytest.approx(101.0)


class TestSizingAndRisk:
    def test_size_comes_from_the_risk_manager_not_the_strategy(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0, size=999.0)}),
            {SYMBOL: frame_from(rows)},
        )
        # 1% of 10 000 risked over a 10-wide stop is 10 units.
        assert result.trades.iloc[0]["size"] == pytest.approx(10.0)

    def test_rejected_signals_are_logged_with_a_reason(self) -> None:
        config = make_config(risk=RiskConfig(risk_per_trade_pct=0.01, max_concurrent_positions=1))
        rows = [(100.0, 100.0, 100.0, 100.0)] * 6
        data = {SYMBOL: frame_from(rows), "ETH/USDT": frame_from(rows)}
        strategy = ScriptedStrategy(
            entries={0: EntrySpec(stop_distance=10.0), 2: EntrySpec(stop_distance=10.0)}
        )
        result = run(strategy, data, config)

        rejected = result.signals[result.signals["status"] == "REJECTED"]
        assert not rejected.empty
        assert rejected.iloc[0]["reject_reason"]
        assert result.rejections

    def test_untradable_size_produces_no_position(self) -> None:
        markets = {SYMBOL: MarketMeta(symbol=SYMBOL, lot_step=1_000.0)}
        rows = [(100.0, 100.0, 100.0, 100.0)] * 4
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}),
            {SYMBOL: frame_from(rows)},
            markets=markets,
        )
        assert result.trades.empty


class TestMultiSymbol:
    def test_capital_is_shared_across_instruments(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 3 + [(110.0, 110.0, 110.0, 110.0)]
        data = {SYMBOL: frame_from(rows), "ETH/USDT": frame_from(rows)}
        result = run(ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}), data)
        assert set(result.trades["symbol"]) == {SYMBOL, "ETH/USDT"}
        assert result.final_equity > result.initial_capital

    def test_symbols_with_different_histories_are_aligned(self) -> None:
        long_frame = flat_frame(6)
        short_frame = flat_frame(6).iloc[3:]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: long_frame, "ETH/USDT": short_frame},
        )
        assert len(result.equity) == 6

    def test_empty_frames_are_skipped(self) -> None:
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: flat_frame(4), "ETH/USDT": flat_frame(0)},
        )
        assert result.symbols == [SYMBOL]

    def test_no_data_at_all_is_an_error(self) -> None:
        with pytest.raises(BacktestError, match="no data"):
            run(ScriptedStrategy(), {SYMBOL: flat_frame(0)})


class TestEquityCurve:
    def test_one_point_per_timestamp(self) -> None:
        result = run(ScriptedStrategy(), {SYMBOL: flat_frame(10)})
        assert len(result.equity) == 10
        assert result.equity.index.is_monotonic_increasing

    def test_columns_and_starting_value(self) -> None:
        result = run(ScriptedStrategy(), {SYMBOL: flat_frame(5)})
        assert list(result.equity.columns) == [
            "balance",
            "equity",
            "open_positions",
            "drawdown_pct",
        ]
        assert result.equity["equity"].iloc[0] == pytest.approx(10_000.0)

    def test_drawdown_is_tracked(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 2 + [
            (100.0, 100.0, 80.0, 80.0),
            (80.0, 80.0, 80.0, 80.0),
        ]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
        )
        assert result.equity["drawdown_pct"].max() > 0

    def test_open_position_count_is_recorded(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 5
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}),
            {SYMBOL: frame_from(rows)},
        )
        assert result.equity["open_positions"].max() == 1


class TestDeterminism:
    def test_identical_inputs_produce_identical_journals(self) -> None:
        rows = [(100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i) for i in range(30)]
        data = {SYMBOL: frame_from(rows)}
        strategy_args = {0: EntrySpec(stop_distance=10.0), 10: EntrySpec(stop_distance=10.0)}

        first = run(ScriptedStrategy(entries=dict(strategy_args), exits={5, 15}), data)
        second = run(ScriptedStrategy(entries=dict(strategy_args), exits={5, 15}), data)

        pd.testing.assert_frame_equal(first.trades, second.trades)
        pd.testing.assert_frame_equal(first.equity, second.equity)

    def test_trade_ids_are_reproducible(self) -> None:
        data = {SYMBOL: flat_frame(5)}
        first = run(ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}), data)
        second = run(ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}), data)
        assert first.trades["trade_id"].tolist() == second.trades["trade_id"].tolist()


class TestJournals:
    def test_trade_schema_matches_the_specification(self) -> None:
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}), {SYMBOL: flat_frame(4)}
        )
        assert list(result.trades.columns) == [
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

    def test_mae_and_mfe_are_recorded_in_r(self) -> None:
        rows = [
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
            (100.0, 115.0, 95.0, 100.0),
            (100.0, 100.0, 100.0, 100.0),
        ]
        result = run(
            ScriptedStrategy(entries={0: EntrySpec(stop_distance=10.0)}), {SYMBOL: frame_from(rows)}
        )
        trade = result.trades.iloc[0]
        assert trade["mfe"] == pytest.approx(1.5)
        assert trade["mae"] == pytest.approx(-0.5)

    def test_bars_held_counts_bars_not_rows(self) -> None:
        rows = [(100.0, 100.0, 100.0, 100.0)] * 6
        strategy = ScriptedStrategy(entries={0: EntrySpec(stop_distance=50.0)}, exits={4})
        result = run(strategy, {SYMBOL: frame_from(rows)})
        assert result.trades.iloc[0]["bars_held"] == 4

    def test_empty_run_still_returns_typed_frames(self) -> None:
        result = run(ScriptedStrategy(), {SYMBOL: flat_frame(3)})
        assert result.trades.empty
        assert result.signals.empty
        assert len(result.equity) == 3
        assert result.total_return_pct == pytest.approx(0.0)
