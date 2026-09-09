"""Walk-forward windows, WFE and OOS stitching (tech.md 9.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.unit.test_backtest_runner import SYMBOL, make_config, trending_frame
from tradingbot.analytics.walkforward import (
    HEALTHY_POSITIVE_OOS,
    HEALTHY_WFE,
    iter_windows,
    run_walkforward,
    save_walkforward,
    stitch_equity,
    wfe,
    window_return,
)
from tradingbot.config.models import OptimizeConfig, StrategyConfig, WalkForwardConfig
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


def _index(start: str, bars: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=bars, freq="4h", tz="UTC", name="ts")


class TestWindows:
    def test_rolling_advances_both_edges(self) -> None:
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2023-01-01", tz="UTC")
        windows = iter_windows(
            start, end, mode="rolling", is_months=18, oos_months=6, step_months=6
        )
        assert len(windows) >= 3
        first, second = windows[0], windows[1]
        assert first.is_start == start
        assert first.oos_start == first.is_end
        assert second.is_start == first.is_start + pd.DateOffset(months=6)
        assert first.is_end == first.is_start + pd.DateOffset(months=18)
        assert second.is_end == second.is_start + pd.DateOffset(months=18)
        assert all(window.oos_end <= end for window in windows)

    def test_anchored_keeps_the_start_fixed(self) -> None:
        start = pd.Timestamp("2020-01-01", tz="UTC")
        end = pd.Timestamp("2023-01-01", tz="UTC")
        windows = iter_windows(
            start, end, mode="anchored", is_months=18, oos_months=6, step_months=6
        )
        assert all(window.is_start == start for window in windows)
        assert windows[1].is_end == windows[0].is_end + pd.DateOffset(months=6)

    def test_too_short_a_span_yields_no_windows(self) -> None:
        start = pd.Timestamp("2024-01-01", tz="UTC")
        end = pd.Timestamp("2024-02-01", tz="UTC")
        assert iter_windows(start, end, is_months=18, oos_months=6) == []


class TestWfe:
    def test_ratio_of_oos_to_is(self) -> None:
        assert wfe(0.06, 0.12) == pytest.approx(0.5)

    def test_undefined_when_is_is_zero(self) -> None:
        assert wfe(0.10, 0.0) is None


class TestStitch:
    def test_second_segment_starts_from_the_first_ending_capital(self) -> None:
        idx_a = _index("2024-01-01", 3)
        idx_b = _index("2024-02-01", 3)
        first = pd.DataFrame(
            {"equity": [100.0, 110.0, 120.0], "balance": [100.0, 110.0, 120.0]},
            index=idx_a,
        )
        second = pd.DataFrame(
            {"equity": [50.0, 55.0, 60.0], "balance": [50.0, 55.0, 60.0]},
            index=idx_b,
        )
        stitched = stitch_equity([first, second], initial_capital=10_000.0)
        # First segment 100 → 120 is +20%, so 10_000 → 12_000.
        assert stitched["equity"].iloc[0] == pytest.approx(10_000.0)
        assert stitched["equity"].iloc[2] == pytest.approx(12_000.0)
        # Second segment 50 → 60 is also +20%, applied to 12_000 → 14_400.
        assert stitched["equity"].iloc[3] == pytest.approx(12_000.0)
        assert stitched["equity"].iloc[-1] == pytest.approx(14_400.0)

    def test_window_return_uses_the_clipped_curve(self) -> None:
        idx = _index("2024-01-01", 10)
        frame = pd.DataFrame({"equity": np.linspace(100.0, 110.0, 10)}, index=idx)
        start, end = idx[2], idx[6]
        expected = float(frame["equity"].iloc[5] / frame["equity"].iloc[2] - 1.0)
        assert window_return(frame, start, end) == pytest.approx(expected)


class TestWalkForwardRun:
    def test_end_to_end_saves_windows_and_report_artefacts(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            strategy=StrategyConfig(name="donchian_trend", params=dict(SHORT_PARAMS)),
            optimize=OptimizeConfig(
                objective="custom",
                min_trades=1,
                grid={"entry_channel": [8, 10], "adx_min": [5.0]},
            ),
            walkforward=WalkForwardConfig(mode="rolling", is_months=1, oos_months=1, step_months=1),
        )
        data = {SYMBOL: trending_frame(bars=900)}
        study = run_walkforward(config, data)
        assert len(study.windows) >= 1
        assert study.positive_oos_share >= 0.0
        assert study.healthy is (
            study.wfe is not None
            and study.wfe >= HEALTHY_WFE
            and study.positive_oos_share >= HEALTHY_POSITIVE_OOS
        )

        stored = save_walkforward(config, data, study)
        assert (stored.path / "walkforward.parquet").is_file()
        assert (stored.path / "walkforward_equity.parquet").is_file()
        assert (stored.path / "walkforward.json").is_file()
        table = pd.read_parquet(stored.path / "walkforward.parquet")
        assert "is_return" in table.columns
        assert "oos_return" in table.columns
        assert "wfe" in table.columns

        from tradingbot.reporting.html_report import extras_for, write_report

        extras = extras_for(stored)
        assert extras.walkforward_windows is not None
        report = write_report(stored)
        assert report.is_file()
        html = report.read_text(encoding="utf-8")
        assert "Walk-forward" in html

    def test_refuses_when_history_is_too_short(self, tmp_path: Path) -> None:
        config = make_config(
            tmp_path,
            strategy=StrategyConfig(name="donchian_trend", params=dict(SHORT_PARAMS)),
            optimize=OptimizeConfig(min_trades=1, grid={"adx_min": [10.0]}),
            walkforward=WalkForwardConfig(is_months=18, oos_months=6),
        )
        data = {SYMBOL: trending_frame(bars=80)}
        with pytest.raises(BacktestError, match="not enough history"):
            run_walkforward(config, data)
