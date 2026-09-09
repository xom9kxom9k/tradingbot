"""Streamlit page renderers (FR-5.1 pages 1-5).

Each function assumes ``st`` is already configured. They take a loaded run
rather than looking it up themselves, so tests can drive a page with a fixture.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from tradingbot.analytics.metrics import PerformanceMetrics
from tradingbot.backtest.runner import StoredRun
from tradingbot.dashboard.compare import comparison_equity, headline_table
from tradingbot.dashboard.session import day_of_week_chart, hour_of_day_chart
from tradingbot.reporting.charts import ExtraCharts, build_charts
from tradingbot.reporting.html_report import CARD_KEYS, charts_for_run


def extras_for(run: StoredRun) -> ExtraCharts:
    """Load optional later-stage artefacts sitting next to the run."""
    extras = ExtraCharts()
    windows = run.path / "walkforward.parquet"
    if windows.is_file():
        extras.walkforward_windows = pd.read_parquet(windows)
    equity = run.path / "walkforward_equity.parquet"
    if equity.is_file():
        frame = pd.read_parquet(equity)
        extras.walkforward_equity = frame.iloc[:, 0] if not frame.empty else None
    monte = run.path / "montecarlo.parquet"
    if monte.is_file():
        extras.montecarlo_paths = pd.read_parquet(monte)
    grid = run.path / "sensitivity.parquet"
    if grid.is_file():
        extras.parameter_grid = pd.read_parquet(grid)
    return extras


def metric_cards(run: StoredRun) -> None:
    """Headline numbers across the top of Overview."""
    if not run.metrics:
        st.info("This run has no metrics.json.")
        return
    headline = PerformanceMetrics.from_dict(run.metrics).headline()
    columns = st.columns(len(CARD_KEYS))
    for column, (key, label, suffix) in zip(columns, CARD_KEYS, strict=True):
        raw = headline.get(key, 0)
        column.metric(label, _fmt(raw, suffix))


def page_overview(
    run: StoredRun,
    *,
    prices: dict[str, pd.DataFrame] | None = None,
    compare: StoredRun | None = None,
) -> None:
    """Cards, equity and drawdown for the selected run."""
    if compare is None:
        metric_cards(run)
        bundle = charts_for_run(run, prices=prices)
        st.plotly_chart(bundle.figures["equity"], use_container_width=True)
        st.plotly_chart(bundle.figures["underwater"], use_container_width=True)
        return
    st.caption(f"Comparing **{run.run_id}** with **{compare.run_id}**.")
    st.dataframe(headline_table(run, compare), hide_index=True, use_container_width=True)
    st.plotly_chart(comparison_equity(run, compare), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.subheader(run.run_id)
        metric_cards(run)
        st.plotly_chart(
            charts_for_run(run, prices=prices).figures["underwater"], use_container_width=True
        )
    with right:
        st.subheader(compare.run_id)
        metric_cards(compare)
        st.plotly_chart(charts_for_run(compare).figures["underwater"], use_container_width=True)


def page_trades(run: StoredRun, trades: pd.DataFrame) -> None:
    """Filterable journal with a CSV download."""
    st.caption(f"{len(trades)} of {len(run.trades)} trades after filters")
    if trades.empty:
        st.info("No trades match the current filters.")
        return
    visible = _display_trades(trades)
    st.dataframe(visible, hide_index=True, use_container_width=True, height=520)
    st.download_button(
        "Download CSV",
        data=visible.to_csv(index=False).encode("utf-8"),
        file_name=f"{run.run_id}-trades.csv",
        mime="text/csv",
    )


def page_chart(
    run: StoredRun,
    trades: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    symbol: str | None,
) -> None:
    """Candles, indicators and markers for one instrument."""
    if not prices:
        st.warning(
            "No cached OHLCV for this run. Charts of trades still work; "
            "run `tradingbot data download` to overlay candles."
        )
    bundle = build_charts(
        equity=run.equity,
        trades=trades,
        prices=prices,
        initial_capital=run.initial_capital,
        symbol=symbol,
        strategy_params=run.meta.config.get("strategy", {}),
    )
    st.plotly_chart(bundle.figures["price"], use_container_width=True)


def page_analytics(run: StoredRun, trades: pd.DataFrame) -> None:
    """Distributions, heatmap, MAE/MFE, per-symbol and session stats."""
    bundle = build_charts(
        equity=run.equity,
        trades=trades,
        prices={},
        initial_capital=run.initial_capital,
    )
    left, right = st.columns(2)
    with left:
        st.plotly_chart(bundle.figures["pnl_distribution"], use_container_width=True)
        st.plotly_chart(bundle.figures["mae_mfe"], use_container_width=True)
        st.plotly_chart(hour_of_day_chart(trades), use_container_width=True)
    with right:
        st.plotly_chart(bundle.figures["r_scatter"], use_container_width=True)
        st.plotly_chart(bundle.figures["per_symbol"], use_container_width=True)
        st.plotly_chart(day_of_week_chart(trades), use_container_width=True)
    st.plotly_chart(bundle.figures["monthly"], use_container_width=True)
    st.plotly_chart(bundle.figures["rolling"], use_container_width=True)


def page_walkforward(run: StoredRun) -> None:
    """Walk-forward, Monte Carlo and parameter heatmap when artefacts exist."""
    extras = extras_for(run)
    bundle = build_charts(
        equity=run.equity,
        trades=run.trades,
        prices={},
        initial_capital=run.initial_capital,
        extras=extras,
    )
    st.plotly_chart(bundle.figures["walkforward"], use_container_width=True)
    st.plotly_chart(bundle.figures["montecarlo"], use_container_width=True)
    st.plotly_chart(bundle.figures["parameter_heatmap"], use_container_width=True)


def _display_trades(trades: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "trade_id",
        "symbol",
        "side",
        "entry_ts",
        "exit_ts",
        "entry_price",
        "exit_price",
        "size",
        "pnl_net",
        "pnl_r",
        "exit_reason",
        "bars_held",
        "mae",
        "mfe",
    ]
    present = [column for column in columns if column in trades]
    frame = trades[present].copy()
    if "pnl_net" in frame:
        frame["pnl_net"] = frame["pnl_net"].map(lambda value: round(float(value), 2))
    if "pnl_r" in frame:
        frame["pnl_r"] = frame["pnl_r"].map(lambda value: round(float(value), 3))
    return frame


def _fmt(value: Any, suffix: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return str(value)
