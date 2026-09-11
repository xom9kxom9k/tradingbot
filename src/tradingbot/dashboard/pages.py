"""Streamlit page renderers (FR-5.1 pages 1-5).

Each function assumes ``st`` is already configured. They take a loaded run
rather than looking it up themselves, so tests can drive a page with a fixture.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from tradingbot.analytics.metrics import PerformanceMetrics
from tradingbot.backtest.runner import StoredRun
from tradingbot.dashboard.compare import comparison_equity, headline_table
from tradingbot.dashboard.session import day_of_week_chart, hour_of_day_chart
from tradingbot.live.state import LiveStore
from tradingbot.reporting.charts import build_charts
from tradingbot.reporting.html_report import CARD_KEYS, charts_for_run, extras_for


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


def page_live() -> None:
    """Open positions, last processed bars and health of the running bot."""
    st.caption("Refreshes about every 30 seconds.")

    def _body() -> None:
        _render_live()

    fragment = getattr(st, "fragment", None)
    if callable(fragment):
        fragment(run_every=30)(_body)()
    else:  # pragma: no cover - Streamlit < 1.33
        _body()


def _render_live() -> None:
    path = Path(os.environ.get("TRADINGBOT_STATE_DB", "data/state.db"))
    if not path.exists():
        st.info(
            f"No live state at `{path}`. Start the bot with `tradingbot live run --mode paper`."
        )
        return

    status = LiveStore.from_path(path).read_status()
    health = status.health
    cards = st.columns(6)
    cards[0].metric("Mode", health.mode)
    cards[1].metric("Running", "yes" if health.running else "no")
    cards[2].metric("Paused", "yes" if health.paused else "no")
    cards[3].metric("Uptime", _fmt_uptime(status.uptime_sec))
    cards[4].metric("Bars", f"{health.bars_processed:,}")
    equity = status.latest_equity
    cards[5].metric("Equity", f"{equity.equity:,.2f}" if equity is not None else "—")

    if health.last_error:
        st.error(health.last_error)
    st.caption(
        f"pid {health.pid or '—'} · started {_fmt_ts(health.started_at)} · "
        f"last cycle {_fmt_ts(health.last_cycle_at)} · "
        f"{status.pending_notifications} queued notifications"
    )

    st.subheader("Last processed bar")
    if status.last_processed:
        st.dataframe(
            pd.DataFrame(
                {
                    "symbol": list(status.last_processed),
                    "bar_ts": [ts.isoformat() for ts in status.last_processed.values()],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No bars processed yet. The runner waits for the next candle close.")

    st.subheader("Open positions")
    if status.open_positions:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "size": pos.size,
                        "entry": pos.entry_price,
                        "stop": pos.current_stop,
                        "entry_ts": pos.entry_ts.isoformat(),
                    }
                    for pos in status.open_positions
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No open positions.")

    st.subheader("Recent signals")
    if status.recent_signals:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "bar_ts": row.bar_ts.isoformat(),
                        "symbol": row.symbol,
                        "type": row.signal_type,
                        "side": row.side,
                        "status": row.status,
                        "reason": row.reason,
                    }
                    for row in status.recent_signals
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No signals stored yet.")


def _fmt_ts(value: datetime | None) -> str:
    if value is None:
        return "—"
    return str(value)


def _fmt_uptime(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


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
