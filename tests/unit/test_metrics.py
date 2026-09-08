"""Metric formulas checked against hand-computed answers (tech.md section 9.1).

Where a metric has a closed form, the expected value in these tests was worked
out on paper rather than captured from the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from tradingbot.analytics.metrics import (
    PerformanceMetrics,
    aggregate_trades,
    annualized_volatility,
    average_drawdown,
    buy_and_hold_return,
    cagr,
    calmar_ratio,
    compute_metrics,
    conditional_var,
    cost_summary,
    drawdown_episodes,
    drawdown_series,
    elapsed_years,
    exposure_pct,
    longest_streak,
    max_drawdown,
    monthly_trade_pnl,
    omega_ratio,
    period_returns,
    periods_per_year,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    trade_stats,
    ulcer_index,
    value_at_risk,
)
from tradingbot.analytics.summary import format_annual_returns, format_metrics
from tradingbot.config.models import CostsConfig
from tradingbot.core.enums import ExitReason, Side


def curve(values: list[float], freq: str = "1D", start: str = "2024-01-01") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq=freq, tz="UTC", name="ts")
    return pd.Series(values, index=index, dtype=float)


def journal(rows: list[dict[str, object]]) -> pd.DataFrame:
    """Build a trade journal, filling in the columns a test does not care about."""
    defaults: dict[str, object] = {
        "trade_id": "t",
        "symbol": "BTC/USDT",
        "side": Side.LONG.value,
        "entry_ts": pd.Timestamp("2024-01-01", tz="UTC"),
        "exit_ts": pd.Timestamp("2024-01-02", tz="UTC"),
        "entry_price": 100.0,
        "exit_price": 110.0,
        "size": 1.0,
        "initial_stop": 90.0,
        "r_value": 10.0,
        "pnl_gross": 10.0,
        "fees": 0.0,
        "funding": 0.0,
        "pnl_net": 10.0,
        "pnl_r": 1.0,
        "pnl_pct": 0.001,
        "bars_held": 6,
        "exit_reason": ExitReason.TP1.value,
        "mae": -0.2,
        "mfe": 1.5,
        "entry_reason": "test",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestAnnualisation:
    def test_known_timeframes(self) -> None:
        assert periods_per_year("1d") == pytest.approx(365.0)
        assert periods_per_year("4h") == pytest.approx(2190.0)
        assert periods_per_year("1h") == pytest.approx(8760.0)

    def test_unknown_timeframe_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown timeframe"):
            periods_per_year("7s")


class TestReturns:
    def test_total_return_is_the_ratio_of_the_endpoints(self) -> None:
        assert total_return(curve([100.0, 50.0, 150.0])) == pytest.approx(0.5)

    def test_total_return_of_a_single_point_is_zero(self) -> None:
        assert total_return(curve([100.0])) == 0.0

    def test_cagr_doubles_over_exactly_one_year(self) -> None:
        index = pd.DatetimeIndex(["2024-01-01", "2024-12-31"], tz="UTC")
        series = pd.Series([100.0, 200.0], index=index)
        # 365 days is one year, so the growth rate is the full doubling.
        assert cagr(series) == pytest.approx(1.0, rel=0.02)

    def test_cagr_over_two_years_is_the_square_root(self) -> None:
        index = pd.DatetimeIndex(["2024-01-01", "2025-12-31"], tz="UTC")
        series = pd.Series([100.0, 400.0], index=index)
        assert cagr(series) == pytest.approx(1.0, rel=0.02)

    def test_cagr_needs_a_time_span(self) -> None:
        assert cagr(curve([100.0])) == 0.0

    def test_elapsed_years(self) -> None:
        index = pd.DatetimeIndex(["2024-01-01", "2024-07-01"], tz="UTC")
        assert elapsed_years(index) == pytest.approx(0.4986, abs=1e-3)

    def test_buy_and_hold(self) -> None:
        assert buy_and_hold_return(curve([100.0, 90.0, 125.0])) == pytest.approx(0.25)

    def test_monthly_returns_start_from_the_opening_equity(self) -> None:
        values = [100.0] * 31 + [110.0] * 29  # January flat, February up 10%
        monthly = period_returns(curve(values), "ME")
        assert list(monthly.round(6)) == [0.0, 0.1]

    def test_period_returns_of_an_empty_curve(self) -> None:
        assert period_returns(pd.Series(dtype=float), "ME").empty


class TestDrawdown:
    def test_drawdown_series_matches_the_definition(self) -> None:
        series = drawdown_series(curve([100.0, 120.0, 90.0, 120.0]))
        assert list(series.round(4)) == [0.0, 0.0, 0.25, 0.0]

    def test_max_drawdown_in_both_units(self) -> None:
        fractional, absolute = max_drawdown(curve([100.0, 200.0, 150.0, 180.0]))
        assert fractional == pytest.approx(0.25)
        assert absolute == pytest.approx(50.0)

    def test_a_rising_curve_has_no_drawdown(self) -> None:
        assert max_drawdown(curve([100.0, 110.0, 120.0])) == (0.0, 0.0)

    def test_episodes_are_split_at_recoveries(self) -> None:
        episodes = drawdown_episodes(curve([100.0, 80.0, 100.0, 90.0, 100.0]))
        assert len(episodes) == 2
        assert [round(episode.depth, 4) for episode in episodes] == [0.2, 0.1]
        assert all(episode.recovered for episode in episodes)

    def test_an_unrecovered_episode_is_still_reported(self) -> None:
        episodes = drawdown_episodes(curve([100.0, 120.0, 60.0]))
        assert len(episodes) == 1
        assert not episodes[0].recovered
        assert episodes[0].depth == pytest.approx(0.5)

    def test_duration_and_recovery_are_measured_in_days(self) -> None:
        series = curve([100.0, 80.0, 70.0, 100.0])
        episode = drawdown_episodes(series)[0]
        end = series.index[-1]
        assert episode.duration_days(end) == pytest.approx(3.0)
        assert episode.recovery_days(end) == pytest.approx(1.0)

    def test_average_drawdown_averages_episodes(self) -> None:
        assert average_drawdown(curve([100.0, 80.0, 100.0, 90.0, 100.0])) == pytest.approx(0.15)

    def test_ulcer_index_of_a_flat_curve_is_zero(self) -> None:
        assert ulcer_index(curve([100.0] * 5)) == pytest.approx(0.0)

    def test_ulcer_index_is_the_rms_drawdown_in_percent(self) -> None:
        # Drawdowns are 0%, 0%, 20%, 0% -> sqrt(400/4) = 10.
        assert ulcer_index(curve([100.0, 120.0, 96.0, 120.0])) == pytest.approx(10.0)


class TestRiskStatistics:
    def test_volatility_scales_with_the_square_root_of_time(self) -> None:
        returns = pd.Series([0.01, -0.01, 0.01, -0.01])
        expected = returns.std(ddof=1) * math.sqrt(365.0)
        assert annualized_volatility(returns, 365.0) == pytest.approx(expected)

    def test_volatility_needs_two_points(self) -> None:
        assert annualized_volatility(pd.Series([0.01]), 365.0) == 0.0

    def test_value_at_risk_is_the_fifth_percentile_loss(self) -> None:
        returns = pd.Series(np.linspace(-0.10, 0.10, 101))
        assert value_at_risk(returns) == pytest.approx(0.09, abs=1e-9)

    def test_var_of_a_profitable_series_is_zero(self) -> None:
        assert value_at_risk(pd.Series([0.01, 0.02, 0.03])) == 0.0

    def test_conditional_var_averages_the_tail(self) -> None:
        returns = pd.Series([-0.10, -0.08, -0.06, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        # The 5% cutoff sits at -0.091, leaving only the worst observation.
        assert conditional_var(returns) == pytest.approx(0.10)

    def test_empty_inputs_are_safe(self) -> None:
        empty = pd.Series(dtype=float)
        assert value_at_risk(empty) == 0.0
        assert conditional_var(empty) == 0.0


class TestRatios:
    def test_sharpe_of_a_constant_return_stream_is_zero_volatility_guarded(self) -> None:
        assert sharpe_ratio(pd.Series([0.01] * 10), 365.0) == 0.0

    def test_sharpe_matches_the_formula(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
        expected = returns.mean() / returns.std(ddof=1) * math.sqrt(365.0)
        assert sharpe_ratio(returns, 365.0) == pytest.approx(expected)

    def test_a_risk_free_rate_lowers_sharpe(self) -> None:
        returns = pd.Series([0.01, -0.005, 0.02, 0.0, -0.01])
        assert sharpe_ratio(returns, 365.0, risk_free=0.05) < sharpe_ratio(returns, 365.0)

    def test_sortino_ignores_upside_volatility(self) -> None:
        calm = pd.Series([0.01, -0.01, 0.01, -0.01])
        spiky = pd.Series([0.05, -0.01, 0.05, -0.01])
        assert sortino_ratio(spiky, 365.0) > sortino_ratio(calm, 365.0)

    def test_sortino_without_losses_is_guarded(self) -> None:
        assert sortino_ratio(pd.Series([0.01, 0.02, 0.03]), 365.0) == 0.0

    def test_calmar_is_growth_over_drawdown(self) -> None:
        assert calmar_ratio(0.30, 0.15) == pytest.approx(2.0)

    def test_calmar_without_a_drawdown_is_zero(self) -> None:
        assert calmar_ratio(0.30, 0.0) == 0.0

    def test_omega_is_the_gain_loss_ratio(self) -> None:
        returns = pd.Series([0.03, -0.01, 0.02, -0.02, 0.01])
        # Gains 0.06 against losses 0.03.
        assert omega_ratio(returns) == pytest.approx(2.0)

    def test_omega_without_losses_is_infinite(self) -> None:
        assert omega_ratio(pd.Series([0.01, 0.02])) == math.inf


class TestTradeStatistics:
    def test_streaks(self) -> None:
        assert longest_streak([True, True, False, True, True, True]) == 3
        assert longest_streak([]) == 0
        assert longest_streak([False, False]) == 0

    def test_profit_factor(self) -> None:
        assert profit_factor(pd.Series([10.0, -5.0, 20.0, -5.0])) == pytest.approx(3.0)

    def test_profit_factor_without_losses_is_infinite(self) -> None:
        assert profit_factor(pd.Series([10.0, 5.0])) == math.inf

    def test_profit_factor_of_an_empty_journal(self) -> None:
        assert profit_factor(pd.Series(dtype=float)) == 0.0

    def test_counts_and_win_rate(self) -> None:
        trades = journal(
            [
                {"pnl_net": 10.0, "pnl_r": 1.0},
                {"pnl_net": -5.0, "pnl_r": -0.5, "side": Side.SHORT.value},
                {"pnl_net": 20.0, "pnl_r": 2.0},
                {"pnl_net": -5.0, "pnl_r": -0.5},
            ]
        )
        stats = trade_stats(trades)
        assert stats.trades == 4
        assert stats.long_trades == 3
        assert stats.short_trades == 1
        assert stats.win_rate == pytest.approx(0.5)
        assert stats.profit_factor == pytest.approx(3.0)

    def test_expectancy_in_both_units(self) -> None:
        trades = journal(
            [
                {"pnl_net": 30.0, "pnl_r": 3.0},
                {"pnl_net": -10.0, "pnl_r": -1.0},
            ]
        )
        stats = trade_stats(trades)
        assert stats.expectancy_r == pytest.approx(1.0)
        assert stats.expectancy_usdt == pytest.approx(10.0)

    def test_average_win_and_loss(self) -> None:
        trades = journal(
            [
                {"pnl_net": 20.0, "pnl_r": 2.0},
                {"pnl_net": 40.0, "pnl_r": 4.0},
                {"pnl_net": -10.0, "pnl_r": -1.0},
            ]
        )
        stats = trade_stats(trades)
        assert stats.avg_win_r == pytest.approx(3.0)
        assert stats.avg_loss_r == pytest.approx(-1.0)

    def test_streaks_are_measured_in_journal_order(self) -> None:
        trades = journal(
            [
                {"pnl_net": 1.0, "pnl_r": 0.1},
                {"pnl_net": 1.0, "pnl_r": 0.1},
                {"pnl_net": -1.0, "pnl_r": -0.1},
                {"pnl_net": -1.0, "pnl_r": -0.1},
                {"pnl_net": -1.0, "pnl_r": -0.1},
            ]
        )
        stats = trade_stats(trades)
        assert stats.max_win_streak == 2
        assert stats.max_loss_streak == 3

    def test_holding_time_comes_from_the_timestamps(self) -> None:
        trades = journal(
            [
                {
                    "entry_ts": pd.Timestamp("2024-01-01 00:00", tz="UTC"),
                    "exit_ts": pd.Timestamp("2024-01-01 12:00", tz="UTC"),
                }
            ]
        )
        assert trade_stats(trades).avg_hours_held == pytest.approx(12.0)

    def test_exit_reason_distribution(self) -> None:
        trades = journal(
            [
                {"exit_reason": ExitReason.STOP.value},
                {"exit_reason": ExitReason.STOP.value},
                {"exit_reason": ExitReason.TRAIL.value},
            ]
        )
        assert trade_stats(trades).exit_reasons == {"STOP": 2, "TRAIL": 1}

    def test_empty_journal_gives_neutral_statistics(self) -> None:
        stats = trade_stats(pd.DataFrame())
        assert stats.trades == 0
        assert stats.win_rate == 0.0
        assert stats.exit_reasons == {}

    def test_aggregation_by_symbol(self) -> None:
        trades = journal(
            [
                {"symbol": "BTC/USDT", "pnl_net": 10.0},
                {"symbol": "ETH/USDT", "pnl_net": -4.0},
                {"symbol": "ETH/USDT", "pnl_net": 6.0},
            ]
        )
        rows = aggregate_trades(trades, "symbol")
        assert [row["symbol"] for row in rows] == ["BTC/USDT", "ETH/USDT"]
        assert rows[1]["trades"] == 2
        assert rows[1]["pnl_net"] == pytest.approx(2.0)

    def test_aggregation_by_side(self) -> None:
        trades = journal(
            [
                {"side": Side.LONG.value, "pnl_net": 10.0},
                {"side": Side.SHORT.value, "pnl_net": -3.0},
            ]
        )
        assert len(aggregate_trades(trades, "side")) == 2

    def test_aggregation_of_a_missing_column(self) -> None:
        assert aggregate_trades(journal([{}]), "venue") == []

    def test_monthly_trade_pnl(self) -> None:
        trades = journal(
            [
                {"exit_ts": pd.Timestamp("2024-01-15", tz="UTC"), "pnl_net": 10.0},
                {"exit_ts": pd.Timestamp("2024-01-20", tz="UTC"), "pnl_net": 5.0},
                {"exit_ts": pd.Timestamp("2024-02-01", tz="UTC"), "pnl_net": -3.0},
            ]
        )
        assert monthly_trade_pnl(trades) == {"2024-01": 15.0, "2024-02": -3.0}


class TestExposure:
    def test_exposure_counts_bars_with_a_position(self) -> None:
        equity = pd.DataFrame({"equity": [1.0] * 4, "open_positions": [0, 1, 1, 0]})
        assert exposure_pct(equity) == pytest.approx(0.5)

    def test_exposure_without_the_column(self) -> None:
        assert exposure_pct(pd.DataFrame({"equity": [1.0]})) == 0.0


class TestCosts:
    def test_costs_are_summed_and_weighed_against_gross_profit(self) -> None:
        trades = journal(
            [
                {"pnl_gross": 100.0, "fees": 2.0, "funding": 1.0, "size": 1.0},
                {"pnl_gross": 100.0, "fees": 2.0, "funding": 1.0, "size": 1.0},
            ]
        )
        costs = CostsConfig(slippage_model="fixed_bps", slippage_bps=10.0)
        summary = cost_summary(trades, costs)
        # Notional is (100 + 110) per trade, so 420 in total at 10 bps.
        assert summary.fees == pytest.approx(4.0)
        assert summary.funding == pytest.approx(2.0)
        assert summary.slippage_estimate == pytest.approx(0.42)
        assert summary.share_of_gross_profit == pytest.approx(6.42 / 200.0)

    def test_measured_slippage_wins_over_the_estimate(self) -> None:
        trades = journal([{"pnl_gross": 100.0, "fees": 2.0}])
        costs = CostsConfig(slippage_model="fixed_bps", slippage_bps=10.0)
        summary = cost_summary(trades, costs, slippage=7.5)
        assert summary.slippage_estimate == pytest.approx(7.5)
        assert summary.total == pytest.approx(9.5)

    def test_the_share_is_measured_against_winning_trades(self) -> None:
        # Costs turned this strategy into a loser; the share must stay meaningful
        # rather than collapsing to zero on a negative denominator.
        trades = journal(
            [
                {"pnl_gross": 100.0, "fees": 60.0},
                {"pnl_gross": -120.0, "fees": 60.0},
            ]
        )
        summary = cost_summary(trades, slippage=0.0)
        assert summary.share_of_gross_profit == pytest.approx(1.2)

    def test_atr_slippage_is_not_guessed_after_the_fact(self) -> None:
        summary = cost_summary(journal([{}]), CostsConfig(slippage_model="atr"))
        assert summary.slippage_estimate == 0.0

    def test_percent_model(self) -> None:
        summary = cost_summary(
            journal([{"size": 2.0}]),
            CostsConfig(slippage_model="percent", slippage_percent=0.001),
        )
        assert summary.slippage_estimate == pytest.approx(0.42)

    def test_empty_journal(self) -> None:
        assert cost_summary(pd.DataFrame()).total == 0.0


class TestCompositeMetrics:
    @pytest.fixture
    def equity(self) -> pd.DataFrame:
        values = [10_000.0, 10_500.0, 10_200.0, 11_000.0, 10_800.0, 12_000.0]
        index = pd.date_range("2024-01-01", periods=6, freq="1D", tz="UTC", name="ts")
        return pd.DataFrame(
            {
                "balance": values,
                "equity": values,
                "open_positions": [0, 1, 1, 1, 0, 0],
                "drawdown_pct": [0.0, 0.0, 0.0286, 0.0, 0.0182, 0.0],
            },
            index=index,
        )

    @pytest.fixture
    def trades(self) -> pd.DataFrame:
        return journal(
            [
                {"pnl_net": 500.0, "pnl_r": 2.0, "fees": 1.0},
                {"pnl_net": -300.0, "pnl_r": -1.0, "fees": 1.0, "side": Side.SHORT.value},
                {"pnl_net": 1_800.0, "pnl_r": 3.0, "fees": 1.0},
            ]
        )

    def test_every_required_group_is_present(
        self, equity: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        metrics = compute_metrics(equity, trades, timeframe="1d")
        assert set(metrics.to_dict()) == {
            "returns",
            "risk",
            "ratios",
            "trading",
            "costs",
            "monthly_returns",
            "annual_returns",
            "by_symbol",
            "by_side",
        }

    def test_all_section_9_1_fields_are_reported(
        self, equity: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        metrics = compute_metrics(equity, trades, timeframe="1d")
        assert {
            "total_return_pct",
            "cagr_pct",
            "buy_hold_pct",
            "alpha_pct",
        } <= set(metrics.returns)
        assert {
            "max_drawdown_pct",
            "max_drawdown_abs",
            "max_drawdown_duration_days",
            "max_drawdown_recovery_days",
            "avg_drawdown_pct",
            "annualized_volatility_pct",
            "var_95_pct",
            "cvar_95_pct",
            "ulcer_index",
        } <= set(metrics.risk)
        assert {"sharpe", "sortino", "calmar", "omega"} <= set(metrics.ratios)
        assert {
            "trades",
            "long_trades",
            "short_trades",
            "win_rate_pct",
            "profit_factor",
            "expectancy_r",
            "expectancy_usdt",
            "avg_win_r",
            "avg_loss_r",
            "max_win_streak",
            "max_loss_streak",
            "avg_bars_held",
            "avg_hours_held",
            "exposure_pct",
            "exit_reasons",
            "avg_mae_r",
            "avg_mfe_r",
        } <= set(metrics.trading)
        assert {"fees", "funding", "slippage_estimate", "share_of_gross_profit"} <= set(
            metrics.costs
        )

    def test_headline_numbers(self, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        metrics = compute_metrics(equity, trades, timeframe="1d")
        assert metrics.returns["total_return_pct"] == pytest.approx(20.0)
        assert metrics.trading["trades"] == 3
        assert metrics.trading["win_rate_pct"] == pytest.approx(200.0 / 3)
        assert metrics.trading["exposure_pct"] == pytest.approx(50.0)

    def test_alpha_is_measured_against_the_benchmark(
        self, equity: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        benchmark = {"BTC/USDT": curve([100.0, 105.0, 110.0, 115.0, 118.0, 130.0])}
        metrics = compute_metrics(equity, trades, timeframe="1d", benchmark=benchmark)
        assert metrics.returns["buy_hold_portfolio_pct"] == pytest.approx(30.0)
        assert metrics.returns["alpha_pct"] == pytest.approx(-10.0)

    def test_an_empty_run_produces_neutral_metrics(self) -> None:
        empty_equity = pd.DataFrame(
            {"equity": [10_000.0, 10_000.0], "open_positions": [0, 0]},
            index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC"),
        )
        metrics = compute_metrics(empty_equity, pd.DataFrame(), timeframe="1d")
        assert metrics.returns["total_return_pct"] == pytest.approx(0.0)
        assert metrics.trading["trades"] == 0
        assert metrics.risk["max_drawdown_pct"] == pytest.approx(0.0)

    def test_round_trip_through_a_dictionary(
        self, equity: pd.DataFrame, trades: pd.DataFrame
    ) -> None:
        metrics = compute_metrics(equity, trades, timeframe="1d")
        restored = PerformanceMetrics.from_dict(metrics.to_dict())
        assert restored.to_dict() == metrics.to_dict()

    def test_headline_selection(self, equity: pd.DataFrame, trades: pd.DataFrame) -> None:
        headline = compute_metrics(equity, trades, timeframe="1d").headline()
        assert set(headline) == {
            "total_return_pct",
            "cagr_pct",
            "max_drawdown_pct",
            "sharpe",
            "profit_factor",
            "win_rate_pct",
            "trades",
        }


class TestConsoleSummary:
    def test_the_table_covers_every_block(self) -> None:
        equity = pd.DataFrame(
            {"equity": [10_000.0, 11_000.0], "open_positions": [0, 1]},
            index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC"),
        )
        text = format_metrics(compute_metrics(equity, journal([{}]), timeframe="1d"))
        for heading in ("Returns", "Risk", "Risk-adjusted", "Trading", "Costs", "Exit reasons"):
            assert heading in text

    def test_infinities_are_rendered_readably(self) -> None:
        equity = pd.DataFrame(
            {"equity": [10_000.0, 11_000.0], "open_positions": [0, 1]},
            index=pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC"),
        )
        # A journal without losses makes the profit factor infinite.
        text = format_metrics(compute_metrics(equity, journal([{}]), timeframe="1d"))
        assert "inf" in text

    def test_annual_returns_block(self) -> None:
        values = [100.0] * 366 + [150.0]
        equity = pd.DataFrame(
            {"equity": values, "open_positions": [0] * len(values)},
            index=pd.date_range("2024-01-01", periods=len(values), freq="1D", tz="UTC"),
        )
        text = format_annual_returns(compute_metrics(equity, pd.DataFrame(), timeframe="1d"))
        assert "2024" in text and "2025" in text

    def test_annual_block_is_empty_without_data(self) -> None:
        empty = pd.DataFrame({"equity": [], "open_positions": []}, index=pd.DatetimeIndex([]))
        assert format_annual_returns(compute_metrics(empty, pd.DataFrame(), timeframe="1d")) == ""
