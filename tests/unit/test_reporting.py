"""Reporting: figures, HTML report and marker ↔ journal alignment."""

from __future__ import annotations

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
from tradingbot.reporting.charts import (
    CHART_IDS,
    build_charts,
    marker_timestamps,
    montecarlo_fan,
    parameter_heatmap,
    walkforward_chart,
)
from tradingbot.reporting.html_report import charts_for_run, render_html, write_report
from tradingbot.reporting.png_export import PngExportError, export_pngs, kaleido_available


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        exchange=ExchangeConfig(symbols=[SYMBOL], timeframe="4h"),
        data=DataConfig(cache_dir=tmp_path / "cache"),
        backtest=BacktestConfig(runs_dir=tmp_path / "runs", initial_capital=10_000.0),
        costs=CostsConfig(taker_fee=0.0004, slippage_model="fixed_bps", slippage_bps=2.0),
    )


@pytest.fixture
def prices() -> dict[str, pd.DataFrame]:
    return {SYMBOL: trending_frame(bars=400)}


@pytest.fixture
def stored(config: AppConfig, prices: dict[str, pd.DataFrame]):
    return BacktestRunner(config).run(data=prices)


class TestChartFactories:
    def test_all_twelve_charts_are_built(self, stored, prices: dict[str, pd.DataFrame]) -> None:
        bundle = build_charts(
            equity=stored.equity,
            trades=stored.trades,
            prices=prices,
            initial_capital=stored.initial_capital,
            strategy_params=stored.meta.config.get("strategy", {}),
        )
        assert list(bundle.figures) == list(CHART_IDS)
        for name, fig in bundle.ordered():
            assert fig.layout.title.text
            assert name in CHART_IDS

    def test_empty_inputs_still_produce_figures(self) -> None:
        empty = pd.DataFrame()
        bundle = build_charts(equity=empty, trades=empty, prices={}, initial_capital=10_000.0)
        assert set(bundle.figures) == set(CHART_IDS)

    def test_entry_and_exit_markers_match_the_journal(
        self, stored, prices: dict[str, pd.DataFrame]
    ) -> None:
        bundle = build_charts(
            equity=stored.equity,
            trades=stored.trades,
            prices=prices,
            initial_capital=stored.initial_capital,
        )
        price = bundle.figures["price"]
        trades = stored.trades[stored.trades["symbol"] == SYMBOL]
        long_entries = pd.to_datetime(trades.loc[trades["side"] == "LONG", "entry_ts"], utc=True)
        short_entries = pd.to_datetime(trades.loc[trades["side"] == "SHORT", "entry_ts"], utc=True)
        exits = pd.to_datetime(trades["exit_ts"], utc=True)

        assert _as_utc(marker_timestamps(price, "Long entry")) == set(long_entries)
        assert _as_utc(marker_timestamps(price, "Short entry")) == set(short_entries)
        assert _as_utc(marker_timestamps(price, "Exit")) == set(exits)

    def test_equity_includes_buy_and_hold_when_prices_are_given(
        self, stored, prices: dict[str, pd.DataFrame]
    ) -> None:
        bundle = build_charts(
            equity=stored.equity,
            trades=stored.trades,
            prices=prices,
            initial_capital=10_000.0,
        )
        names = [trace.name for trace in bundle.figures["equity"].data]
        assert "Strategy" in names
        assert "Buy & hold" in names

    def test_walkforward_placeholder_without_data(self) -> None:
        fig = walkforward_chart(None, None)
        assert "walk-forward" in fig.layout.title.text.lower()

    def test_walkforward_bars_when_windows_are_supplied(self) -> None:
        windows = pd.DataFrame(
            {
                "window": ["2023", "2024"],
                "is_return": [0.20, 0.10],
                "oos_return": [0.08, -0.02],
            }
        )
        fig = walkforward_chart(windows, None)
        assert len(fig.data) == 2
        assert fig.data[0].name == "In-sample"

    def test_montecarlo_fan_from_paths(self) -> None:
        paths = pd.DataFrame({f"s{i}": list(range(10, 20)) for i in range(30)})
        fig = montecarlo_fan(paths)
        assert {trace.name for trace in fig.data} >= {"5th", "Median", "95th"}

    def test_parameter_heatmap_from_grid(self) -> None:
        grid = pd.DataFrame(
            {
                "x": [10, 10, 20, 20],
                "y": [50, 100, 50, 100],
                "value": [0.4, 0.8, 0.2, 1.1],
            }
        )
        fig = parameter_heatmap(grid, x_label="entry", y_label="ema")
        assert fig.data[0].type == "heatmap"


class TestHtmlReport:
    def test_report_contains_every_chart_and_the_trade_ids(self, stored, prices) -> None:
        bundle = charts_for_run(stored, prices=prices)
        document = render_html(stored, bundle)
        for chart_id in CHART_IDS:
            assert f'id="chart-{chart_id}"' in document
        for trade_id in stored.trades["trade_id"].astype(str):
            assert trade_id in document
        assert stored.run_id in document
        assert "Total return" in document

    def test_write_report_creates_the_file(self, stored, prices, tmp_path: Path) -> None:
        path = write_report(stored, prices=prices, path=tmp_path / "report.html")
        assert path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "<html" in text
        assert "plotly" in text.lower()

    def test_empty_trade_journal_still_renders(self, stored, prices) -> None:
        stored.trades = stored.trades.iloc[0:0]
        document = render_html(stored, charts_for_run(stored, prices=prices))
        assert "No closed trades" in document


class TestPngExport:
    def test_export_writes_files_or_explains_the_renderer(
        self, stored, prices, tmp_path: Path
    ) -> None:
        bundle = charts_for_run(stored, prices=prices)
        if not kaleido_available():
            pytest.skip("kaleido is not installed")
        try:
            written = export_pngs(bundle, tmp_path / "charts", names=("equity",))
        except PngExportError:
            pytest.skip("kaleido could not render in this environment")
        assert written and written[0].is_file()
        assert written[0].stat().st_size > 0


def _as_utc(values: list[object]) -> set[pd.Timestamp]:
    return set(pd.to_datetime(pd.Index(values), utc=True))
