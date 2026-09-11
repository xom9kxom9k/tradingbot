"""Dashboard filters, comparison and Streamlit smoke tests."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from tests.unit.test_backtest_runner import SYMBOL, trending_frame
from tradingbot.backtest.runner import BacktestRunner
from tradingbot.config.models import (
    AppConfig,
    BacktestConfig,
    CostsConfig,
    DataConfig,
    ExchangeConfig,
)
from tradingbot.core.enums import Side
from tradingbot.dashboard.compare import comparison_equity, headline_table, normalised_equity
from tradingbot.dashboard.filters import (
    TradeFilters,
    apply_filters,
    day_of_week_stats,
    hour_of_day_stats,
)
from tradingbot.dashboard.pages import extras_for

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None  # type: ignore[misc, assignment]


def _journal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["BTC/USDT", "ETH/USDT", "BTC/USDT", "BTC/USDT"],
            "side": [Side.LONG.value, Side.SHORT.value, Side.LONG.value, Side.LONG.value],
            "entry_ts": pd.to_datetime(
                ["2024-01-01 08:00", "2024-01-02 14:00", "2024-01-15 08:00", "2024-02-01 22:00"],
                utc=True,
            ),
            "exit_ts": pd.to_datetime(
                ["2024-01-01 16:00", "2024-01-02 20:00", "2024-01-16 08:00", "2024-02-02 02:00"],
                utc=True,
            ),
            "pnl_net": [10.0, -4.0, 8.0, -2.0],
            "pnl_r": [1.0, -0.4, 0.8, -0.2],
        }
    )


class TestFilters:
    def test_symbol_and_side(self) -> None:
        trades = _journal()
        filtered = apply_filters(
            trades, TradeFilters(symbols=("BTC/USDT",), sides=(Side.LONG.value,))
        )
        assert len(filtered) == 3
        assert set(filtered["symbol"]) == {"BTC/USDT"}

    def test_empty_symbol_list_matches_nothing(self) -> None:
        filtered = apply_filters(_journal(), TradeFilters(symbols=()))
        assert filtered.empty

    def test_period_uses_the_exit_date(self) -> None:
        filtered = apply_filters(
            _journal(), TradeFilters(start=date(2024, 1, 2), end=date(2024, 1, 16))
        )
        assert len(filtered) == 2

    def test_wins_only(self) -> None:
        filtered = apply_filters(_journal(), TradeFilters(outcome="win"))
        assert list(filtered["pnl_net"]) == [10.0, 8.0]

    def test_empty_journal(self) -> None:
        assert apply_filters(pd.DataFrame(), TradeFilters(outcome="loss")).empty

    def test_hour_buckets_cover_the_day(self) -> None:
        stats = hour_of_day_stats(_journal())
        assert len(stats) == 24
        assert int(stats.loc[stats["hour"] == 8, "trades"].iloc[0]) == 2

    def test_weekday_order(self) -> None:
        stats = day_of_week_stats(_journal())
        assert list(stats["weekday"])[:2] == ["Monday", "Tuesday"]


@pytest.fixture
def stored(tmp_path: Path):
    config = AppConfig(
        exchange=ExchangeConfig(symbols=[SYMBOL], timeframe="4h"),
        data=DataConfig(cache_dir=tmp_path / "cache"),
        backtest=BacktestConfig(runs_dir=tmp_path / "runs", initial_capital=10_000.0),
        costs=CostsConfig(taker_fee=0.0004, slippage_model="fixed_bps", slippage_bps=2.0),
    )
    return BacktestRunner(config).run(data={SYMBOL: trending_frame(bars=400)})


class TestCompare:
    def test_normalised_equity_starts_near_one(self, stored) -> None:
        series = normalised_equity(stored)
        assert series.iloc[0] == pytest.approx(1.0, abs=0.05)

    def test_comparison_figure_has_both_traces(self, stored) -> None:
        fig = comparison_equity(stored, stored)
        assert len(fig.data) == 2

    def test_headline_table_has_both_run_ids(self, stored) -> None:
        table = headline_table(stored, stored)
        assert stored.run_id in table.columns
        assert "Sharpe" in set(table["metric"])


class TestExtras:
    def test_missing_artefacts_are_empty(self, stored) -> None:
        extras = extras_for(stored)
        assert extras.walkforward_windows is None
        assert extras.montecarlo_paths is None

    def test_walkforward_parquet_is_picked_up(self, stored) -> None:
        frame = pd.DataFrame({"window": ["A"], "is_return": [0.1], "oos_return": [0.05]})
        frame.to_parquet(stored.path / "walkforward.parquet")
        extras = extras_for(stored)
        assert extras.walkforward_windows is not None
        assert extras.walkforward_windows.iloc[0]["window"] == "A"


class TestPreviews:
    def test_preview_pngs_are_written(self, stored, tmp_path: Path) -> None:
        from tradingbot.dashboard.previews import export_previews
        from tradingbot.reporting.png_export import PngExportError, kaleido_available

        if not kaleido_available():
            pytest.skip("kaleido is not installed")
        try:
            written = export_previews(stored, tmp_path / "images")
        except PngExportError:
            pytest.skip("kaleido could not render in this environment")
        names = {path.name for path in written}
        assert names == {"overview.png", "chart.png", "analytics.png", "walkforward.png"}
        assert all(path.stat().st_size > 0 for path in written)


@pytest.mark.skipif(AppTest is None, reason="streamlit testing extras are unavailable")
class TestApp:
    def test_empty_runs_dir_explains_itself(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADINGBOT_RUNS_DIR", str(tmp_path / "missing"))
        monkeypatch.setenv("TRADINGBOT_STATE_DB", str(tmp_path / "missing-state.db"))
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[2] / "dashboard" / "app.py"),
            default_timeout=30,
        )
        app.run()
        assert not app.exception
        texts = " ".join(item.value for item in app.info)
        assert "No runs" in texts or "backtest" in texts.lower()

    def test_every_page_renders_on_a_real_run(
        self, stored, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TRADINGBOT_RUNS_DIR", str(stored.path.parent))
        monkeypatch.setenv("TRADINGBOT_STATE_DB", str(tmp_path / "live-state.db"))
        script = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
        app = AppTest.from_file(str(script), default_timeout=60)
        app.run()
        assert not app.exception
        for page in ("Trades", "Chart", "Analytics", "Walk-Forward", "Live"):
            app.radio[0].set_value(page).run()
            assert not app.exception, page
