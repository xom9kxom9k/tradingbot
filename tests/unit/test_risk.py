"""Position sizing and every portfolio limit from tech.md section 7.9."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingbot.config.models import RiskConfig
from tradingbot.core.enums import Side, SignalType
from tradingbot.core.models import MarketMeta, Position, Signal
from tradingbot.risk import RiskDecision, RiskManager, position_size
from tradingbot.risk.manager import RejectionLog, open_risk_amount

EQUITY = 10_000.0
TS = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def entry_signal(
    symbol: str = "BTC/USDT",
    side: Side = Side.LONG,
    price: float = 100.0,
    stop: float = 90.0,
) -> Signal:
    return Signal(
        signal_type=SignalType.ENTRY,
        side=side,
        symbol=symbol,
        timeframe="4h",
        bar_ts=TS,
        price=price,
        stop_loss=stop,
        size=1.0,
        risk_pct=0.01,
    )


def held(
    symbol: str = "ETH/USDT",
    side: Side = Side.LONG,
    entry: float = 100.0,
    stop: float = 90.0,
    size: float = 1.0,
) -> Position:
    return Position(
        symbol=symbol,
        side=side,
        entry_ts=TS,
        entry_price=entry,
        size=size,
        initial_size=size,
        initial_stop=stop,
        current_stop=stop,
        highest_price=entry,
        lowest_price=entry,
    )


def manager(**overrides: object) -> RiskManager:
    config = RiskConfig(**overrides)  # type: ignore[arg-type]
    risk = RiskManager(config)
    risk.update_equity(TS, EQUITY)
    return risk


class TestPositionSize:
    def test_matches_a_hand_calculation(self) -> None:
        # 1% of 10 000 = 100 USDT at risk, stop is 10 away, so 10 units.
        result = position_size(equity=EQUITY, entry_price=100.0, stop_price=90.0, risk_pct=0.01)
        assert result.size == pytest.approx(10.0)
        assert result.risk_amount == pytest.approx(100.0)
        assert result.r_value == pytest.approx(10.0)
        assert result.is_tradable

    def test_wider_stop_buys_fewer_units(self) -> None:
        narrow = position_size(equity=EQUITY, entry_price=100.0, stop_price=95.0, risk_pct=0.01)
        wide = position_size(equity=EQUITY, entry_price=100.0, stop_price=80.0, risk_pct=0.01)
        assert narrow.size > wide.size
        assert narrow.risk_amount == pytest.approx(wide.risk_amount)

    def test_short_side_is_symmetric(self) -> None:
        result = position_size(equity=EQUITY, entry_price=100.0, stop_price=110.0, risk_pct=0.01)
        assert result.size == pytest.approx(10.0)

    def test_notional_cap_limits_the_size(self) -> None:
        # Risk-based size would be 100 units (10 000 notional); the cap allows 3 500.
        result = position_size(
            equity=EQUITY,
            entry_price=100.0,
            stop_price=99.0,
            risk_pct=0.01,
            max_notional_pct=0.35,
        )
        assert result.capped_by_notional
        assert result.size == pytest.approx(35.0)
        assert result.notional(100.0) == pytest.approx(3_500.0)

    def test_uncapped_size_is_not_flagged(self) -> None:
        result = position_size(
            equity=EQUITY,
            entry_price=100.0,
            stop_price=90.0,
            risk_pct=0.01,
            max_notional_pct=0.35,
        )
        assert not result.capped_by_notional

    def test_size_is_floored_to_the_lot_step(self) -> None:
        market = MarketMeta(symbol="BTC/USDT", lot_step=0.001)
        result = position_size(
            equity=EQUITY,
            entry_price=63_250.0,
            stop_price=61_100.0,
            risk_pct=0.01,
            market=market,
        )
        assert result.size == pytest.approx(0.046)
        assert result.size * 1000 == pytest.approx(round(result.size * 1000))

    def test_rounding_never_increases_risk(self) -> None:
        market = MarketMeta(symbol="BTC/USDT", lot_step=0.01)
        result = position_size(
            equity=EQUITY, entry_price=100.0, stop_price=90.0, risk_pct=0.01, market=market
        )
        assert result.risk_amount <= EQUITY * 0.01 + 1e-9

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"stop_price": 100.0}, "stop coincides with entry"),
            ({"equity": 0.0}, "equity is not positive"),
            ({"entry_price": 0.0}, "entry price is not positive"),
            ({"risk_pct": 0.0}, "risk_per_trade_pct is not positive"),
        ],
    )
    def test_degenerate_inputs_are_rejected(self, kwargs: dict[str, float], expected: str) -> None:
        base = {"equity": EQUITY, "entry_price": 100.0, "stop_price": 90.0, "risk_pct": 0.01}
        result = position_size(**{**base, **kwargs})  # type: ignore[arg-type]
        assert not result.is_tradable
        assert expected in result.reason

    def test_micro_capital_rounds_away_to_nothing(self) -> None:
        market = MarketMeta(symbol="BTC/USDT", lot_step=0.001)
        result = position_size(
            equity=5.0, entry_price=63_000.0, stop_price=61_000.0, risk_pct=0.01, market=market
        )
        assert not result.is_tradable
        assert result.rounded_away
        assert "lot step" in result.reason

    def test_minimum_notional_is_enforced(self) -> None:
        market = MarketMeta(symbol="BTC/USDT", lot_step=0.001, min_notional=1_000.0)
        result = position_size(
            equity=100.0, entry_price=100.0, stop_price=90.0, risk_pct=0.01, market=market
        )
        assert not result.is_tradable
        assert "min notional" in result.reason

    def test_minimum_size_is_enforced(self) -> None:
        market = MarketMeta(symbol="BTC/USDT", lot_step=0.001, min_size=1.0)
        result = position_size(
            equity=100.0, entry_price=100.0, stop_price=90.0, risk_pct=0.01, market=market
        )
        assert not result.is_tradable
        assert "min size" in result.reason


class TestOpenRisk:
    def test_risk_before_the_stop_moves(self) -> None:
        assert open_risk_amount(held(size=2.0)) == pytest.approx(20.0)

    def test_stop_above_entry_means_no_remaining_risk(self) -> None:
        position = held()
        position.current_stop = 105.0
        assert open_risk_amount(position) == 0.0

    def test_short_position_risk(self) -> None:
        position = held(side=Side.SHORT, entry=100.0, stop=110.0, size=3.0)
        assert open_risk_amount(position) == pytest.approx(30.0)


class TestApproval:
    def test_a_clean_signal_is_approved(self) -> None:
        decision = manager().evaluate(entry_signal(), equity=EQUITY)
        assert decision.approved
        assert decision.size == pytest.approx(10.0)
        assert decision.reason == ""

    def test_the_manager_overrides_the_strategy_size(self) -> None:
        signal = entry_signal()
        decision = manager(risk_per_trade_pct=0.02).evaluate(signal, equity=EQUITY)
        assert signal.size == 1.0
        assert decision.size == pytest.approx(20.0)

    def test_notional_cap_is_reported(self) -> None:
        decision = manager().evaluate(entry_signal(price=100.0, stop=99.0), equity=EQUITY)
        assert decision.approved
        assert decision.reason == "notional cap applied"


class TestLimits:
    """Each limit gets one test that trips it and one that passes just under it."""

    def test_duplicate_symbol_is_blocked(self) -> None:
        decision = manager().evaluate(
            entry_signal(), equity=EQUITY, open_positions=[held(symbol="BTC/USDT")]
        )
        assert not decision.approved
        assert "already open" in decision.reason

    def test_max_concurrent_positions(self) -> None:
        risk = manager(max_concurrent_positions=2)
        two = [held(symbol="ETH/USDT"), held(symbol="SOL/USDT")]
        assert not risk.evaluate(entry_signal(), equity=EQUITY, open_positions=two).approved
        assert risk.evaluate(entry_signal(), equity=EQUITY, open_positions=two[:1]).approved

    def test_max_positions_per_side(self) -> None:
        risk = manager(max_concurrent_positions=5, max_positions_per_side=2)
        longs = [held(symbol="ETH/USDT"), held(symbol="SOL/USDT")]
        blocked = risk.evaluate(entry_signal(), equity=EQUITY, open_positions=longs)
        assert not blocked.approved
        assert "LONG positions" in blocked.reason

        short = entry_signal(side=Side.SHORT, price=100.0, stop=110.0)
        assert risk.evaluate(short, equity=EQUITY, open_positions=longs).approved

    def test_portfolio_risk_ceiling(self) -> None:
        risk = manager(max_concurrent_positions=5, max_portfolio_risk_pct=0.025)
        # Two open positions risking 1% each leave room for only 0.5%.
        positions = [
            held(symbol="ETH/USDT", entry=100.0, stop=90.0, size=10.0),
            held(symbol="SOL/USDT", entry=100.0, stop=90.0, size=10.0),
        ]
        decision = risk.evaluate(entry_signal(), equity=EQUITY, open_positions=positions)
        assert not decision.approved
        assert "portfolio risk would reach" in decision.reason

    def test_portfolio_risk_just_within_the_ceiling(self) -> None:
        risk = manager(max_concurrent_positions=5, max_portfolio_risk_pct=0.03)
        positions = [
            held(symbol="ETH/USDT", entry=100.0, stop=90.0, size=10.0),
            held(symbol="SOL/USDT", entry=100.0, stop=90.0, size=10.0),
        ]
        assert risk.evaluate(entry_signal(), equity=EQUITY, open_positions=positions).approved

    def test_daily_loss_limit_blocks_new_entries(self) -> None:
        risk = manager(daily_loss_limit_pct=0.05)
        risk.register_realized_pnl(-500.0)
        decision = risk.evaluate(entry_signal(), equity=9_500.0)
        assert not decision.approved
        assert "daily loss limit" in decision.reason

    def test_daily_loss_just_under_the_limit_still_trades(self) -> None:
        risk = manager(daily_loss_limit_pct=0.05)
        risk.register_realized_pnl(-499.0)
        assert risk.evaluate(entry_signal(), equity=9_501.0).approved

    def test_daily_loss_resets_on_the_next_day(self) -> None:
        risk = manager(daily_loss_limit_pct=0.05)
        risk.register_realized_pnl(-600.0)
        assert not risk.evaluate(entry_signal(), equity=9_400.0).approved

        risk.update_equity(TS + timedelta(days=1), 9_400.0)
        assert risk.state.day_realized_pnl == 0.0
        assert risk.evaluate(entry_signal(), equity=9_400.0).approved

    def test_max_drawdown_stop_halts_trading(self) -> None:
        risk = manager(max_drawdown_stop_pct=0.25)
        risk.update_equity(TS + timedelta(hours=4), 7_400.0)
        assert risk.state.halted
        decision = risk.evaluate(entry_signal(), equity=7_400.0)
        assert not decision.approved
        assert "max drawdown stop" in decision.reason

    def test_drawdown_just_above_the_stop_keeps_trading(self) -> None:
        risk = manager(max_drawdown_stop_pct=0.25)
        risk.update_equity(TS + timedelta(hours=4), 7_600.0)
        assert not risk.state.halted
        assert risk.evaluate(entry_signal(), equity=7_600.0).approved

    def test_halt_survives_an_equity_recovery(self) -> None:
        risk = manager(max_drawdown_stop_pct=0.25)
        risk.update_equity(TS + timedelta(hours=4), 7_000.0)
        risk.update_equity(TS + timedelta(hours=8), 9_900.0)
        assert risk.state.halted
        risk.resume()
        assert risk.evaluate(entry_signal(), equity=9_900.0).approved

    def test_exhausted_equity_is_blocked(self) -> None:
        risk = manager()
        assert not risk.evaluate(entry_signal(), equity=0.0).approved

    def test_untradable_size_is_rejected_with_its_reason(self) -> None:
        risk = RiskManager(
            RiskConfig(), {"BTC/USDT": MarketMeta(symbol="BTC/USDT", lot_step=1000.0)}
        )
        risk.update_equity(TS, EQUITY)
        decision = risk.evaluate(entry_signal(), equity=EQUITY)
        assert not decision.approved
        assert "lot step" in decision.reason


class TestState:
    def test_peak_equity_and_drawdown(self) -> None:
        risk = manager()
        risk.update_equity(TS + timedelta(hours=4), 12_000.0)
        risk.update_equity(TS + timedelta(hours=8), 10_800.0)
        assert risk.state.peak_equity == pytest.approx(12_000.0)
        assert risk.state.drawdown_pct == pytest.approx(0.10)

    def test_drawdown_without_history_is_zero(self) -> None:
        assert RiskManager(RiskConfig()).state.drawdown_pct == 0.0

    def test_day_rollover_rebases_the_loss_budget(self) -> None:
        risk = manager()
        risk.register_realized_pnl(-100.0)
        risk.update_equity(TS + timedelta(days=1), 9_900.0)
        assert risk.state.day_start_equity == pytest.approx(9_900.0)
        assert risk.state.day_loss_pct == 0.0

    def test_market_lookup_falls_back_to_permissive(self) -> None:
        risk = RiskManager(RiskConfig())
        assert risk.market("DOGE/USDT") == MarketMeta.permissive("DOGE/USDT")


class TestDecisionHelpers:
    def test_reject_carries_the_reason(self) -> None:
        decision = RiskDecision.reject("nope")
        assert not decision.approved
        assert decision.size == 0.0
        assert decision.reason == "nope"

    def test_rejection_log_counts_by_reason(self) -> None:
        log = RejectionLog()
        log.add(TS, "BTC/USDT", "daily loss limit")
        log.add(TS, "ETH/USDT", "daily loss limit")
        log.add(TS, "SOL/USDT", "portfolio risk")
        assert len(log) == 3
        assert log.summary() == {"daily loss limit": 2, "portfolio risk": 1}
