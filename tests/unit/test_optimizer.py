"""Grid / Optuna search, scoring and plateau selection (tech.md 9.3)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.unit.test_backtest_runner import SYMBOL, make_config, trending_frame
from tradingbot.analytics.metrics import PerformanceMetrics
from tradingbot.analytics.optimizer import (
    REJECTED_SCORE,
    TrialRecord,
    expand_grid,
    heatmap_frame,
    pick_plateau,
    run_grid,
    run_optuna,
    save_optimisation,
    score_metrics,
    with_params,
)
from tradingbot.config.models import AppConfig, OptimizeConfig, StrategyConfig
from tradingbot.core.exceptions import BacktestError

SHORT_PARAMS = {
    "entry_channel": 8,
    "exit_channel": 4,
    "trend_fast": 8,
    "trend_slow": 16,
    "atr_period": 5,
    "adx_period": 5,
    "adx_min": 10.0,
    "cooldown_bars": 1,
}


def _metrics(
    *, trades: int, sharpe: float = 1.0, calmar: float = 2.0, factor: float = 1.5
) -> PerformanceMetrics:
    return PerformanceMetrics.from_dict(
        {
            "ratios": {"sharpe": sharpe, "calmar": calmar},
            "trading": {"trades": trades, "profit_factor": factor},
        }
    )


def _trial(params: dict[str, object], score: float, *, accepted: bool = True) -> TrialRecord:
    return TrialRecord(
        params=params,
        score=score,
        accepted=accepted,
        reason="",
        trades=40,
        total_return_pct=10.0,
        sharpe=1.0,
        calmar=1.0,
        profit_factor=1.2,
        max_drawdown_pct=5.0,
    )


def search_config(tmp_path: Path) -> AppConfig:
    return make_config(
        tmp_path,
        strategy=StrategyConfig(name="donchian_trend", params=dict(SHORT_PARAMS)),
        optimize=OptimizeConfig(
            objective="custom",
            min_trades=1,
            heatmap_x="entry_channel",
            heatmap_y="adx_min",
            grid={"entry_channel": [8, 10], "adx_min": [5.0, 15.0]},
        ),
    )


class TestExpandGrid:
    def test_cartesian_product_is_stable(self) -> None:
        combos = expand_grid({"a": [1, 2], "b": ["x", "y"]})
        assert combos == [
            {"a": 1, "b": "x"},
            {"a": 1, "b": "y"},
            {"a": 2, "b": "x"},
            {"a": 2, "b": "y"},
        ]

    def test_empty_space_is_one_default_trial(self) -> None:
        assert expand_grid({}) == [{}]


class TestScoreMetrics:
    def test_too_few_trades_are_rejected(self) -> None:
        score, accepted, reason = score_metrics(
            _metrics(trades=2), objective="sharpe", min_trades=30
        )
        assert not accepted
        assert score == REJECTED_SCORE
        assert "2 trades" in reason

    def test_custom_is_calmar_times_log1p_trades(self) -> None:
        import math

        score, accepted, _ = score_metrics(
            _metrics(trades=30, calmar=2.0), objective="custom", min_trades=30
        )
        assert accepted
        assert score == pytest.approx(2.0 * math.log1p(30))

    def test_sharpe_objective(self) -> None:
        score, accepted, _ = score_metrics(
            _metrics(trades=30, sharpe=1.7), objective="sharpe", min_trades=10
        )
        assert accepted
        assert score == pytest.approx(1.7)


class TestPlateau:
    def test_picks_the_centre_not_the_peak(self) -> None:
        trials = [
            _trial({"entry_channel": 10, "adx_min": 15.0}, 1.00),
            _trial({"entry_channel": 20, "adx_min": 20.0}, 1.05),
            _trial({"entry_channel": 30, "adx_min": 25.0}, 0.99),
            _trial({"entry_channel": 20, "adx_min": 15.0}, 1.02),
            _trial({"entry_channel": 999, "adx_min": 99.0}, 2.00, accepted=False),
        ]
        chosen = pick_plateau(trials, band=0.10)
        assert chosen is not None
        # Peak is 1.05 at (20, 20); the 10% band keeps the four accepted cells,
        # whose median is (20, 20) — the plateau centre, not a rejected spike.
        assert chosen.params["entry_channel"] == 20
        assert chosen.params["adx_min"] == 20.0

    def test_empty_when_nothing_is_accepted(self) -> None:
        assert pick_plateau([_trial({"a": 1}, 1.0, accepted=False)]) is None


class TestHeatmap:
    def test_averages_duplicate_cells(self) -> None:
        trials = [
            _trial({"x": 1, "y": 10}, 1.0),
            _trial({"x": 1, "y": 10}, 3.0),
            _trial({"x": 2, "y": 10}, 2.0),
        ]
        frame = heatmap_frame(trials, "x", "y")
        assert list(frame.columns) == ["x", "y", "value"]
        cell = frame[(frame["x"] == 1) & (frame["y"] == 10)]
        assert float(cell["value"].iloc[0]) == pytest.approx(2.0)


class TestGridSearch:
    def test_two_by_two_grid_writes_heatmap(self, tmp_path: Path) -> None:
        config = search_config(tmp_path)
        data = {SYMBOL: trending_frame(bars=400)}
        search = run_grid(config, data)
        assert len(search.trials) == 4
        assert search.chosen is not None
        assert {"entry_channel", "adx_min"} <= set(search.chosen.params)
        assert {"x", "y", "value"} <= set(search.heatmap.columns)

        stored = save_optimisation(config, data, search, method="grid")
        assert (stored.path / "sensitivity.parquet").is_file()
        assert (stored.path / "trials.parquet").is_file()
        assert (stored.path / "optimize.json").is_file()
        heat = pd.read_parquet(stored.path / "sensitivity.parquet")
        assert not heat.empty
        assert stored.meta.note == "optimize grid"

    def test_with_params_does_not_mutate_the_original(self, tmp_path: Path) -> None:
        config = search_config(tmp_path)
        updated = with_params(config, {"adx_min": 33.0})
        assert config.strategy.params["adx_min"] == 10.0
        assert updated.strategy.params["adx_min"] == 33.0

    def test_save_without_accepted_trials_fails(self, tmp_path: Path) -> None:
        config = search_config(tmp_path)
        data = {SYMBOL: trending_frame(bars=400)}
        search = run_grid(config, data, min_trades=10_000)
        assert search.chosen is None
        with pytest.raises(BacktestError, match="no accepted"):
            save_optimisation(config, data, search, method="grid")


class TestOptuna:
    def test_is_deterministic_with_a_seed(self, tmp_path: Path) -> None:
        config = search_config(tmp_path)
        data = {SYMBOL: trending_frame(bars=400)}
        first = run_optuna(config, data, n_trials=4, seed=7)
        second = run_optuna(config, data, n_trials=4, seed=7)
        assert first.chosen is not None
        assert second.chosen is not None
        assert first.chosen.params == second.chosen.params
        assert len(first.trials) == 4
