"""Streamlit dashboard for exploring stored backtests (FR-5.1, section 10.3).

Launch with ``tradingbot dashboard`` or ``streamlit run dashboard/app.py``.
The heavy lifting — loading a run, building Plotly figures — is cached so
switching pages does not recompute the equity curve.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from tradingbot.backtest.runner import StoredRun, latest_run_id, list_runs, load_run
from tradingbot.dashboard.filters import (
    TradeFilters,
    apply_filters,
    available_sides,
    available_symbols,
)
from tradingbot.dashboard.pages import (
    page_analytics,
    page_chart,
    page_overview,
    page_trades,
    page_walkforward,
)
from tradingbot.reporting.html_report import load_prices
from tradingbot.reporting.theme import install_template

PAGES = ("Overview", "Trades", "Chart", "Analytics", "Walk-Forward")


def run() -> None:
    """Configure the page and draw the selected view."""
    st.set_page_config(
        page_title="tradingbot",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    install_template()
    runs_dir = Path(os.environ.get("TRADINGBOT_RUNS_DIR", "runs"))
    runs = list_runs(runs_dir)
    if not runs:
        st.title("tradingbot")
        st.info(f"No runs in `{runs_dir}`. Run `tradingbot backtest run` first.")
        return

    default = latest_run_id(runs_dir) or runs[0]
    selected, compare_id, filters, symbol = _sidebar(runs, default, runs_dir)
    primary = _cached_run(selected, str(runs_dir))
    counterpart = _cached_run(compare_id, str(runs_dir)) if compare_id else None
    prices = _cached_prices(selected, str(runs_dir))
    trades = apply_filters(primary.trades, filters)

    st.title("tradingbot")
    st.caption(
        f"{primary.run_id} · {', '.join(primary.meta.symbols) or '—'} · {primary.meta.timeframe}"
    )
    page = st.radio("Page", PAGES, horizontal=True, label_visibility="collapsed")
    if page == "Overview":
        page_overview(primary, prices=prices, compare=counterpart)
    elif page == "Trades":
        page_trades(primary, trades)
    elif page == "Chart":
        page_chart(primary, trades, prices, symbol)
    elif page == "Analytics":
        page_analytics(primary, trades)
    else:
        page_walkforward(primary)


def _sidebar(
    runs: list[str],
    default: str,
    runs_dir: Path,
) -> tuple[str, str | None, TradeFilters, str | None]:
    """Run picker, compare toggle and trade filters."""
    st.sidebar.header("Run")
    index = runs.index(default) if default in runs else 0
    selected = st.sidebar.selectbox("run_id", runs, index=index)
    compare_on = st.sidebar.toggle("Compare with another run")
    compare_id: str | None = None
    if compare_on:
        others = [item for item in runs if item != selected] or runs
        compare_id = st.sidebar.selectbox("compare with", others)

    preview = _cached_run(selected, str(runs_dir))
    symbols = available_symbols(preview.trades)
    sides = available_sides(preview.trades)
    start_default, end_default = _date_bounds(preview.trades)

    st.sidebar.header("Filters")
    chosen_symbols = st.sidebar.multiselect("Symbol", symbols, default=symbols)
    chosen_sides = st.sidebar.multiselect("Side", sides, default=sides)
    outcome = st.sidebar.selectbox("Result", ("all", "win", "loss"))
    start: date | None = None
    end: date | None = None
    if start_default is not None and end_default is not None:
        period = st.sidebar.date_input(
            "Period (exit date)",
            value=(start_default, end_default),
        )
        start, end = _period_bounds(period)

    filters = TradeFilters(
        symbols=_narrow(chosen_symbols, symbols),
        sides=_narrow(chosen_sides, sides),
        start=start if start != start_default else None,
        end=end if end != end_default else None,
        outcome=str(outcome),
    )
    chart_symbol = chosen_symbols[0] if chosen_symbols else (symbols[0] if symbols else None)
    if len(chosen_symbols) > 1:
        chart_symbol = st.sidebar.selectbox("Chart symbol", chosen_symbols)
    return selected, compare_id, filters, chart_symbol


def _date_bounds(trades: pd.DataFrame) -> tuple[date | None, date | None]:
    if trades.empty or "exit_ts" not in trades:
        return None, None
    exits = pd.to_datetime(trades["exit_ts"], utc=True)
    return exits.min().date(), exits.max().date()


@st.cache_resource(show_spinner=False)
def _cached_run(run_id: str, runs_dir: str) -> StoredRun:
    """Load a run once per (id, directory) pair."""
    return load_run(run_id, runs_dir)


@st.cache_data(show_spinner=False)
def _cached_prices(run_id: str, runs_dir: str) -> dict[str, pd.DataFrame]:
    """OHLCV used by the price chart, keyed by symbol."""
    return load_prices(_cached_run(run_id, runs_dir))


def _period_bounds(period: object) -> tuple[date | None, date | None]:
    """Normalise Streamlit's date_input return (one day or a pair)."""
    if isinstance(period, tuple) and len(period) >= 2:
        first, second = period[0], period[1]
        if isinstance(first, date) and isinstance(second, date):
            return first, second
    if isinstance(period, date):
        return period, period
    return None, None


def _narrow(selected: list[str], universe: list[str]) -> tuple[str, ...] | None:
    """``None`` means 'no restriction'; an empty tuple means 'match nothing'."""
    if not universe:
        return None
    if not selected:
        return ()
    if set(selected) == set(universe):
        return None
    return tuple(selected)


if __name__ == "__main__":  # pragma: no cover
    run()
