"""Indicator correctness, warmup behaviour and absence of look-ahead."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingbot.core.exceptions import StrategyError
from tradingbot.indicators import (
    adx,
    atr,
    atr_pct,
    directional_movement,
    donchian,
    donchian_width_pct,
    ema,
    keltner_channels,
    rma,
    roc,
    rsi,
    slope,
    slope_pct,
    sma,
    stdev,
    true_range,
)

from .. import reference

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def candles() -> pd.DataFrame:
    """The 200-bar reference dataset."""
    return pd.read_csv(FIXTURES / "ohlcv_200.csv", index_col="ts", parse_dates=["ts"])


@pytest.fixture(scope="module")
def golden() -> pd.DataFrame:
    """Indicator values precomputed by the naive reference implementation."""
    return pd.read_csv(FIXTURES / "indicators_expected.csv", index_col="ts", parse_dates=["ts"])


def assert_matches(actual: pd.Series, expected: pd.Series, tolerance: float = TOLERANCE) -> None:
    """Compare two series including the position of their NaNs."""
    pd.testing.assert_series_equal(
        actual.reset_index(drop=True).rename(None),
        expected.reset_index(drop=True).rename(None),
        atol=tolerance,
        rtol=0.0,
        check_names=False,
    )


class TestAgainstGoldenValues:
    """Vectorised implementations must match the stored reference values."""

    def test_sma(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        assert_matches(sma(candles["close"], 20), golden["sma_20"])

    @pytest.mark.parametrize("period", [50, 200])
    def test_ema(self, candles: pd.DataFrame, golden: pd.DataFrame, period: int) -> None:
        assert_matches(ema(candles["close"], period), golden[f"ema_{period}"])

    def test_true_range(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        actual = true_range(candles["high"], candles["low"], candles["close"])
        assert_matches(actual, golden["true_range"])

    def test_atr(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        actual = atr(candles["high"], candles["low"], candles["close"], 14)
        assert_matches(actual, golden["atr_14"])

    def test_donchian(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        channel = donchian(candles["high"], candles["low"], 20)
        for column in ("dc_upper", "dc_lower", "dc_mid"):
            assert_matches(channel[column], golden[column])

    def test_directional_movement(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        frame = directional_movement(candles["high"], candles["low"], candles["close"], 14)
        for column in ("plus_di", "minus_di", "adx"):
            assert_matches(frame[column], golden[column], tolerance=1e-8)

    def test_adx_helper_matches_the_frame(self, candles: pd.DataFrame) -> None:
        frame = directional_movement(candles["high"], candles["low"], candles["close"], 14)
        assert_matches(adx(candles["high"], candles["low"], candles["close"], 14), frame["adx"])

    def test_rsi(self, candles: pd.DataFrame, golden: pd.DataFrame) -> None:
        assert_matches(rsi(candles["close"], 14), golden["rsi_14"])

    def test_reference_is_independent_of_production_code(self, candles: pd.DataFrame) -> None:
        """Guard against the golden file drifting into a copy of the implementation."""
        naive = reference.ema(candles["close"].tolist(), 50)
        vectorised = ema(candles["close"], 50).tolist()
        assert np.allclose(naive[49:], vectorised[49:], atol=TOLERANCE)


class TestWarmup:
    def test_sma_warmup(self, candles: pd.DataFrame) -> None:
        values = sma(candles["close"], 20)
        assert values.iloc[:19].isna().all()
        assert values.iloc[19:].notna().all()

    def test_ema_warmup_and_seed(self, candles: pd.DataFrame) -> None:
        values = ema(candles["close"], 50)
        assert values.iloc[:49].isna().all()
        assert values.iloc[49] == pytest.approx(candles["close"].iloc[:50].mean())

    def test_atr_warmup(self, candles: pd.DataFrame) -> None:
        values = atr(candles["high"], candles["low"], candles["close"], 14)
        assert values.iloc[:13].isna().all()
        assert values.iloc[13:].notna().all()

    def test_adx_warmup_is_roughly_two_periods(self, candles: pd.DataFrame) -> None:
        values = adx(candles["high"], candles["low"], candles["close"], 14)
        assert values.iloc[:26].isna().all()
        assert values.iloc[27:].notna().all()

    def test_series_shorter_than_the_period(self) -> None:
        short = pd.Series([1.0, 2.0, 3.0])
        assert ema(short, 50).isna().all()
        assert rma(short, 50).isna().all()
        assert sma(short, 50).isna().all()

    def test_index_is_preserved(self, candles: pd.DataFrame) -> None:
        assert ema(candles["close"], 20).index.equals(candles.index)


class TestConstantSeries:
    """A flat market is the simplest case where the right answer is obvious."""

    @pytest.fixture
    def flat(self) -> pd.DataFrame:
        index = pd.date_range("2024-01-01", periods=100, freq="4h", tz="UTC")
        return pd.DataFrame(
            {"open": 50.0, "high": 50.0, "low": 50.0, "close": 50.0, "volume": 1.0}, index=index
        )

    def test_moving_averages_equal_the_constant(self, flat: pd.DataFrame) -> None:
        assert ema(flat["close"], 20).iloc[-1] == pytest.approx(50.0)
        assert sma(flat["close"], 20).iloc[-1] == pytest.approx(50.0)

    def test_atr_is_zero(self, flat: pd.DataFrame) -> None:
        assert atr(flat["high"], flat["low"], flat["close"], 14).iloc[-1] == pytest.approx(0.0)

    def test_rsi_is_neutral_without_movement(self, flat: pd.DataFrame) -> None:
        assert rsi(flat["close"], 14).iloc[-1] == pytest.approx(50.0)

    def test_donchian_collapses_to_the_price(self, flat: pd.DataFrame) -> None:
        channel = donchian(flat["high"], flat["low"], 20)
        assert channel["dc_upper"].iloc[-1] == pytest.approx(50.0)
        assert channel["dc_mid"].iloc[-1] == pytest.approx(50.0)

    def test_adx_has_no_direction(self, flat: pd.DataFrame) -> None:
        assert adx(flat["high"], flat["low"], flat["close"], 14).iloc[-1] == pytest.approx(0.0)

    def test_stdev_is_zero(self, flat: pd.DataFrame) -> None:
        assert stdev(flat["close"], 20).iloc[-1] == pytest.approx(0.0)


class TestMonotoneSeries:
    def test_rsi_saturates_on_an_unbroken_rally(self) -> None:
        rising = pd.Series(np.arange(1.0, 101.0))
        assert rsi(rising, 14).iloc[-1] == pytest.approx(100.0)

    def test_rsi_bottoms_out_on_an_unbroken_selloff(self) -> None:
        falling = pd.Series(np.arange(100.0, 0.0, -1.0))
        assert rsi(falling, 14).iloc[-1] == pytest.approx(0.0)

    def test_roc(self) -> None:
        series = pd.Series([100.0, 110.0, 121.0])
        assert roc(series, 1).iloc[-1] == pytest.approx(10.0)
        assert roc(series, 2).iloc[-1] == pytest.approx(21.0)

    def test_slope(self) -> None:
        series = pd.Series([1.0, 3.0, 5.0, 7.0])
        assert slope(series, 2).iloc[-1] == pytest.approx(2.0)
        assert slope_pct(series, 2).iloc[-1] == pytest.approx(2.0 / 7.0)


class TestDonchianShift:
    """The channel must exclude the bar it is evaluated on (tech.md 7.2)."""

    def test_current_bar_is_not_part_of_its_own_channel(self) -> None:
        index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 99.0], index=index)
        low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0, 1.0], index=index)

        channel = donchian(high, low, 5)
        assert channel["dc_upper"].iloc[5] == pytest.approx(14.0)
        assert channel["dc_lower"].iloc[5] == pytest.approx(9.0)

    def test_a_breakout_is_therefore_detectable(self) -> None:
        index = pd.date_range("2024-01-01", periods=6, freq="4h", tz="UTC")
        high = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 99.0], index=index)
        low = pd.Series([9.0, 10.0, 11.0, 12.0, 13.0, 1.0], index=index)
        close = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0, 90.0], index=index)

        channel = donchian(high, low, 5)
        assert close.iloc[5] > channel["dc_upper"].iloc[5]

    def test_first_valid_value_appears_after_period_bars(self) -> None:
        index = pd.date_range("2024-01-01", periods=10, freq="4h", tz="UTC")
        high = pd.Series(np.arange(10.0, 20.0), index=index)
        low = pd.Series(np.arange(9.0, 19.0), index=index)
        channel = donchian(high, low, 5)
        assert channel["dc_upper"].iloc[:5].isna().all()
        assert not np.isnan(channel["dc_upper"].iloc[5])

    def test_width_pct(self, candles: pd.DataFrame) -> None:
        width = donchian_width_pct(candles["high"], candles["low"], candles["close"], 20)
        assert (width.dropna() > 0).all()


class TestNoLookAhead:
    """Values already computed must never change when future bars arrive."""

    @pytest.mark.parametrize(
        "name",
        ["ema", "sma", "atr", "adx", "rsi", "donchian_upper", "donchian_lower", "atr_pct"],
    )
    def test_past_values_are_stable(self, candles: pd.DataFrame, name: str) -> None:
        def compute(frame: pd.DataFrame) -> pd.Series:
            high, low, close = frame["high"], frame["low"], frame["close"]
            match name:
                case "ema":
                    return ema(close, 50)
                case "sma":
                    return sma(close, 20)
                case "atr":
                    return atr(high, low, close, 14)
                case "adx":
                    return adx(high, low, close, 14)
                case "rsi":
                    return rsi(close, 14)
                case "atr_pct":
                    return atr_pct(high, low, close, 14)
                case "donchian_upper":
                    return donchian(high, low, 20)["dc_upper"]
                case _:
                    return donchian(high, low, 20)["dc_lower"]

        truncated = compute(candles.iloc[:150])
        full = compute(candles).iloc[:150]
        assert_matches(truncated, full)

    def test_appending_a_wild_bar_does_not_rewrite_history(self, candles: pd.DataFrame) -> None:
        spike = candles.iloc[[-1]].copy()
        spike.index = spike.index + pd.Timedelta(hours=4)
        spike[["open", "high", "low", "close"]] *= 100.0
        extended = pd.concat([candles, spike])

        before = atr(candles["high"], candles["low"], candles["close"], 14)
        after = atr(extended["high"], extended["low"], extended["close"], 14).iloc[:-1]
        assert_matches(before, after)


class TestGuards:
    @pytest.mark.parametrize(
        ("function", "period"),
        [(sma, 0), (ema, 0), (rma, 0), (slope, 0)],
    )
    def test_non_positive_periods_are_rejected(self, function: object, period: int) -> None:
        series = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(StrategyError, match="must be >= 1"):
            function(series, period)  # type: ignore[operator]

    def test_stdev_needs_at_least_two_bars(self) -> None:
        with pytest.raises(StrategyError, match="must be >= 2"):
            stdev(pd.Series([1.0, 2.0]), 1)

    def test_donchian_period_is_validated(self) -> None:
        with pytest.raises(StrategyError, match="must be >= 1"):
            donchian(pd.Series([1.0]), pd.Series([1.0]), 0)

    def test_rsi_and_roc_periods_are_validated(self) -> None:
        with pytest.raises(StrategyError, match="must be >= 1"):
            rsi(pd.Series([1.0]), 0)
        with pytest.raises(StrategyError, match="must be >= 1"):
            roc(pd.Series([1.0]), 0)

    def test_adx_period_is_validated(self) -> None:
        series = pd.Series([1.0, 2.0])
        with pytest.raises(StrategyError, match="must be >= 1"):
            directional_movement(series, series, series, 0)


class TestDerivedIndicators:
    def test_atr_pct_is_atr_over_close(self, candles: pd.DataFrame) -> None:
        expected = atr(candles["high"], candles["low"], candles["close"], 14) / candles["close"]
        assert_matches(atr_pct(candles["high"], candles["low"], candles["close"], 14), expected)

    def test_keltner_bands_straddle_the_middle(self, candles: pd.DataFrame) -> None:
        bands = keltner_channels(candles["high"], candles["low"], candles["close"])
        valid = bands.dropna()
        assert (valid["kc_upper"] > valid["kc_middle"]).all()
        assert (valid["kc_lower"] < valid["kc_middle"]).all()

    def test_stdev_matches_pandas(self, candles: pd.DataFrame) -> None:
        expected = candles["close"].rolling(20).std(ddof=1)
        assert_matches(stdev(candles["close"], 20), expected)
