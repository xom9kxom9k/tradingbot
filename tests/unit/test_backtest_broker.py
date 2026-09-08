"""Cost model: slippage, fees, funding and gap-aware fill references."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tradingbot.backtest.broker import PaperBroker, funding_boundaries
from tradingbot.config.models import CostsConfig
from tradingbot.core.enums import Side
from tradingbot.core.models import MarketMeta, Position

TS = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def broker(**overrides: object) -> PaperBroker:
    return PaperBroker(CostsConfig(**overrides))  # type: ignore[arg-type]


def position(side: Side = Side.LONG, size: float = 1.0, entry: float = 100.0) -> Position:
    return Position(
        symbol="BTC/USDT",
        side=side,
        entry_ts=TS,
        entry_price=entry,
        size=size,
        initial_size=size,
        initial_stop=entry * 0.9,
        current_stop=entry * 0.9,
        highest_price=entry,
        lowest_price=entry,
    )


class TestSlippage:
    def test_fixed_bps(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=5)
        assert model.slippage_amount(10_000.0) == pytest.approx(5.0)

    def test_percent(self) -> None:
        model = broker(slippage_model="percent", slippage_percent=0.001)
        assert model.slippage_amount(10_000.0) == pytest.approx(10.0)

    def test_atr_model(self) -> None:
        model = broker(slippage_model="atr", slippage_atr_mult=0.05)
        assert model.slippage_amount(10_000.0, atr=200.0) == pytest.approx(10.0)

    def test_atr_model_without_atr_is_free(self) -> None:
        model = broker(slippage_model="atr")
        assert model.slippage_amount(10_000.0, atr=None) == 0.0

    def test_atr_model_ignores_nan(self) -> None:
        model = broker(slippage_model="atr")
        assert model.slippage_amount(10_000.0, atr=float("nan")) == 0.0

    def test_slippage_always_works_against_the_trade(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=10)
        assert model.fill_price(100.0, buying=True) == pytest.approx(100.1)
        assert model.fill_price(100.0, buying=False) == pytest.approx(99.9)

    def test_slippage_can_be_switched_off(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=10)
        assert model.fill_price(100.0, buying=True, apply_slippage=False) == 100.0


class TestFees:
    def test_taker_and_maker(self) -> None:
        model = broker(taker_fee=0.0005, maker_fee=0.0002)
        assert model.fee_for(10_000.0) == pytest.approx(5.0)
        assert model.fee_for(10_000.0, taker=False) == pytest.approx(2.0)

    def test_entry_fill_charges_on_the_filled_price(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=10, taker_fee=0.001)
        fill = model.open_position_fill(Side.LONG, 100.0, 2.0)
        assert fill.price == pytest.approx(100.1)
        assert fill.notional == pytest.approx(200.2)
        assert fill.fee == pytest.approx(0.2002)

    def test_exit_fill_of_a_long_sells_below_the_reference(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=10)
        assert model.close_position_fill(Side.LONG, 100.0, 1.0).price == pytest.approx(99.9)

    def test_exit_fill_of_a_short_buys_above_the_reference(self) -> None:
        model = broker(slippage_model="fixed_bps", slippage_bps=10)
        assert model.close_position_fill(Side.SHORT, 100.0, 1.0).price == pytest.approx(100.1)


class TestGapHandling:
    def test_long_stop_fills_at_the_stop_when_there_is_no_gap(self) -> None:
        assert broker().stop_fill_reference(Side.LONG, stop=95.0, bar_open=99.0) == 95.0

    def test_long_stop_fills_at_the_gap_open(self) -> None:
        assert broker().stop_fill_reference(Side.LONG, stop=95.0, bar_open=80.0) == 80.0

    def test_short_stop_fills_at_the_gap_open(self) -> None:
        assert broker().stop_fill_reference(Side.SHORT, stop=105.0, bar_open=120.0) == 120.0

    def test_target_fills_at_the_favourable_gap(self) -> None:
        assert broker().target_fill_reference(Side.LONG, target=110.0, bar_open=115.0) == 115.0
        assert broker().target_fill_reference(Side.SHORT, target=90.0, bar_open=85.0) == 85.0


class TestFunding:
    def test_boundaries_inside_a_window(self) -> None:
        marks = funding_boundaries(
            datetime(2026, 9, 8, 7, 0, tzinfo=UTC), datetime(2026, 9, 8, 17, 0, tzinfo=UTC)
        )
        assert marks == [
            datetime(2026, 9, 8, 8, 0, tzinfo=UTC),
            datetime(2026, 9, 8, 16, 0, tzinfo=UTC),
        ]

    def test_no_boundaries_in_a_short_window(self) -> None:
        assert (
            funding_boundaries(
                datetime(2026, 9, 8, 9, 0, tzinfo=UTC), datetime(2026, 9, 8, 12, 0, tzinfo=UTC)
            )
            == []
        )

    def test_reversed_window_is_empty(self) -> None:
        assert funding_boundaries(TS, TS - timedelta(hours=1)) == []

    def test_long_pays_and_short_receives(self) -> None:
        model = broker(funding_rate_8h=0.0001)
        long_cost = model.funding_cost(position(Side.LONG, size=2.0), price=50_000.0, events=1)
        short_cost = model.funding_cost(position(Side.SHORT, size=2.0), price=50_000.0, events=1)
        assert long_cost == pytest.approx(10.0)
        assert short_cost == pytest.approx(-10.0)

    def test_cost_scales_with_the_number_of_events(self) -> None:
        model = broker(funding_rate_8h=0.0001)
        one = model.funding_cost(position(), price=100.0, events=1)
        three = model.funding_cost(position(), price=100.0, events=3)
        assert three == pytest.approx(3 * one)

    def test_zero_events_costs_nothing(self) -> None:
        assert broker().funding_cost(position(), price=100.0, events=0) == 0.0

    def test_funding_can_be_disabled(self) -> None:
        model = broker(apply_funding=False)
        events = model.funding_events(
            position(), datetime(2026, 9, 8, tzinfo=UTC), datetime(2026, 9, 9, tzinfo=UTC)
        )
        assert events == 0

    def test_events_are_counted_from_the_entry(self) -> None:
        model = broker(apply_funding=True)
        held = position()
        held.entry_ts = datetime(2026, 9, 8, 15, 0, tzinfo=UTC)
        events = model.funding_events(
            held, datetime(2026, 9, 8, 12, 0, tzinfo=UTC), datetime(2026, 9, 8, 20, 0, tzinfo=UTC)
        )
        assert events == 1


class TestMarketLookup:
    def test_known_market(self) -> None:
        meta = MarketMeta(symbol="BTC/USDT", lot_step=0.001)
        assert PaperBroker(CostsConfig(), {"BTC/USDT": meta}).market("BTC/USDT") is meta

    def test_unknown_market_is_permissive(self) -> None:
        assert broker().market("DOGE/USDT") == MarketMeta.permissive("DOGE/USDT")
