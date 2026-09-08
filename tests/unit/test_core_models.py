"""Domain models, validators and identifier helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from tradingbot.core.enums import ExitReason, RunStatus, Side, SignalType
from tradingbot.core.models import (
    Bar,
    BarContext,
    Position,
    RunMeta,
    Signal,
    TakeProfit,
    Trade,
    config_hash,
    make_run_id,
    utcnow,
)

TS = datetime(2026, 9, 8, 12, 0, tzinfo=UTC)


def make_bar(**overrides: object) -> Bar:
    values: dict[str, object] = {
        "ts": TS,
        "open": 100.0,
        "high": 110.0,
        "low": 95.0,
        "close": 105.0,
        "volume": 1_000.0,
    }
    values.update(overrides)
    return Bar(**values)  # type: ignore[arg-type]


def make_signal(**overrides: object) -> Signal:
    values: dict[str, object] = {
        "signal_type": SignalType.ENTRY,
        "side": Side.LONG,
        "symbol": "BTC/USDT",
        "timeframe": "4h",
        "bar_ts": TS,
        "price": 63_250.0,
        "stop_loss": 61_100.0,
        "size": 0.145,
        "risk_pct": 0.01,
    }
    values.update(overrides)
    return Signal(**values)  # type: ignore[arg-type]


def make_position(**overrides: object) -> Position:
    values: dict[str, object] = {
        "symbol": "BTC/USDT",
        "side": Side.LONG,
        "entry_ts": TS,
        "entry_price": 100.0,
        "size": 1.0,
        "initial_size": 1.0,
        "initial_stop": 90.0,
        "current_stop": 90.0,
        "highest_price": 100.0,
        "lowest_price": 100.0,
    }
    values.update(overrides)
    return Position(**values)  # type: ignore[arg-type]


class TestBar:
    def test_valid_bar(self) -> None:
        bar = make_bar()
        assert bar.range == pytest.approx(15.0)
        assert bar.is_gap is False

    def test_naive_timestamp_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone-aware"):
            make_bar(ts=datetime(2026, 9, 8, 12, 0))

    def test_timestamp_converted_to_utc(self) -> None:
        moscow = datetime(2026, 9, 8, 15, 0, tzinfo=UTC) + timedelta(0)
        assert make_bar(ts=moscow).ts.tzinfo is UTC

    @pytest.mark.parametrize(
        ("field", "value"),
        [("high", 104.0), ("low", 106.0)],
    )
    def test_ohlc_consistency_enforced(self, field: str, value: float) -> None:
        with pytest.raises(ValidationError):
            make_bar(**{field: value})

    def test_non_positive_price_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_bar(low=0.0)


class TestSignal:
    def test_r_value_and_identifiers(self) -> None:
        signal = make_signal()
        assert signal.r_value == pytest.approx(2_150.0)
        assert signal.display_id == "SIG-20260908-1200-BTCUSDT-L"
        assert signal.uid == "SIG-20260908-1200-BTCUSDT-L-4h-ENTRY"

    def test_uid_differs_per_signal_type(self) -> None:
        entry = make_signal()
        exit_signal = make_signal(signal_type=SignalType.EXIT, exit_reason=ExitReason.TP1, size=0.0)
        assert entry.uid != exit_signal.uid

    def test_uid_differs_per_timeframe(self) -> None:
        assert make_signal().uid != make_signal(timeframe="1h").uid

    def test_long_stop_must_be_below_entry(self) -> None:
        with pytest.raises(ValidationError, match="below the entry price"):
            make_signal(stop_loss=64_000.0)

    def test_short_stop_must_be_above_entry(self) -> None:
        with pytest.raises(ValidationError, match="above the entry price"):
            make_signal(side=Side.SHORT, stop_loss=60_000.0)

    def test_short_signal_is_accepted(self) -> None:
        signal = make_signal(side=Side.SHORT, stop_loss=65_000.0)
        assert signal.side is Side.SHORT
        assert signal.display_id.endswith("-S")

    def test_entry_requires_positive_size(self) -> None:
        with pytest.raises(ValidationError, match="positive size"):
            make_signal(size=0.0)

    def test_exit_signal_may_have_zero_size(self) -> None:
        signal = make_signal(signal_type=SignalType.EXIT, size=0.0, stop_loss=70_000.0)
        assert signal.size == 0.0

    def test_take_profit_fraction_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TakeProfit(price=1.0, r_multiple=2.0, fraction=1.5)

    def test_signal_is_immutable(self) -> None:
        with pytest.raises(ValidationError):
            make_signal().price = 1.0  # type: ignore[misc]

    def test_roundtrips_through_dict(self) -> None:
        signal = make_signal(
            take_profits=[TakeProfit(price=67_550.0, r_multiple=2.0, fraction=0.5)]
        )
        assert Signal.model_validate(signal.model_dump()) == signal


class TestSide:
    def test_sign_and_opposite(self) -> None:
        assert Side.LONG.sign == 1
        assert Side.SHORT.sign == -1
        assert Side.LONG.opposite is Side.SHORT
        assert Side.SHORT.letter == "S"

    def test_serialises_as_plain_string(self) -> None:
        assert f"{Side.LONG}" == "LONG"


class TestPosition:
    def test_unrealized_pnl_long_and_short(self) -> None:
        long = make_position()
        assert long.unrealized_pnl(110.0) == pytest.approx(10.0)
        short = make_position(side=Side.SHORT, initial_stop=110.0, current_stop=110.0)
        assert short.unrealized_pnl(90.0) == pytest.approx(10.0)

    def test_excursion_in_r(self) -> None:
        position = make_position()
        assert position.r_value == pytest.approx(10.0)
        assert position.excursion_r(120.0) == pytest.approx(2.0)
        assert position.excursion_r(95.0) == pytest.approx(-0.5)

    def test_register_extremes_is_monotonic(self) -> None:
        position = make_position()
        position.register_extremes(high=120.0, low=98.0)
        position.register_extremes(high=110.0, low=99.0)
        assert position.highest_price == pytest.approx(120.0)
        assert position.lowest_price == pytest.approx(98.0)

    def test_position_is_mutable(self) -> None:
        position = make_position()
        position.current_stop = 95.0
        assert position.current_stop == 95.0

    def test_notional(self) -> None:
        assert make_position(size=2.0).notional == pytest.approx(200.0)


class TestTrade:
    def _trade(self, **overrides: object) -> Trade:
        values: dict[str, object] = {
            "symbol": "BTC/USDT",
            "side": Side.LONG,
            "entry_ts": TS,
            "entry_price": 100.0,
            "exit_ts": TS + timedelta(hours=36),
            "exit_price": 120.0,
            "size": 1.0,
            "initial_stop": 90.0,
            "r_value": 10.0,
            "pnl_gross": 20.0,
            "fees": 0.2,
            "funding": 0.05,
            "pnl_net": 19.75,
            "pnl_r": 1.975,
            "pnl_pct": 0.0198,
            "bars_held": 9,
            "exit_reason": ExitReason.TP1,
            "mae": -0.35,
            "mfe": 2.41,
        }
        values.update(overrides)
        return Trade(**values)  # type: ignore[arg-type]

    def test_win_flag_and_row_serialisation(self) -> None:
        trade = self._trade()
        assert trade.is_win is True
        row = trade.to_row()
        assert row["side"] == "LONG"
        assert row["exit_reason"] == "TP1"
        assert isinstance(row["entry_ts"], datetime)

    def test_trade_id_is_unique(self) -> None:
        assert self._trade().trade_id != self._trade().trade_id

    def test_exit_before_entry_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must not precede"):
            self._trade(exit_ts=TS - timedelta(hours=1))

    def test_losing_trade(self) -> None:
        assert self._trade(pnl_net=-5.0).is_win is False


class TestBarContext:
    def test_exposes_the_closed_bar_only(self) -> None:
        index = pd.date_range("2026-09-08", periods=5, freq="4h", tz="UTC")
        frame = pd.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=index)
        ctx = BarContext(
            symbol="BTC/USDT",
            timeframe="4h",
            index=2,
            history=frame.iloc[:3],
            equity=10_000.0,
        )
        assert ctx.bar["close"] == 3.0
        assert ctx.ts == index[2].to_pydatetime()
        assert len(ctx.history) == 3

    def test_defaults(self) -> None:
        frame = pd.DataFrame(
            {"close": [1.0]}, index=pd.date_range("2026-09-08", periods=1, tz="UTC")
        )
        ctx = BarContext(symbol="BTC/USDT", timeframe="4h", index=0, history=frame, equity=1.0)
        assert ctx.position is None
        assert ctx.bars_since_exit is None


class TestRunIdentity:
    def test_config_hash_is_stable_and_order_independent(self) -> None:
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})
        assert len(config_hash({"a": 1})) == 8

    def test_run_id_format(self) -> None:
        run_id = make_run_id({"a": 1}, moment=TS)
        assert run_id.startswith("20260908-120000-")
        assert len(run_id) == len("20260908-120000-") + 8

    def test_run_meta_defaults(self) -> None:
        meta = RunMeta(run_id="20260908-120000-abcdef12", code_version="0.1.0")
        assert meta.status is RunStatus.RUNNING
        assert meta.created_at.tzinfo is UTC

    def test_utcnow_is_aware(self) -> None:
        assert utcnow().tzinfo is UTC
