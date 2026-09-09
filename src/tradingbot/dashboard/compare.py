"""Side-by-side comparison of two stored runs."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from tradingbot.analytics.metrics import PerformanceMetrics
from tradingbot.backtest.runner import StoredRun
from tradingbot.reporting.theme import BLUE, GREEN, MUTED, style


def normalised_equity(run: StoredRun) -> pd.Series:
    """Equity as a fraction of starting capital, so two runs can share an axis."""
    if run.equity.empty or "equity" not in run.equity:
        return pd.Series(dtype=float)
    start = run.initial_capital or float(run.equity["equity"].iloc[0])
    if start <= 0:
        return pd.Series(dtype=float)
    series = run.equity["equity"].astype(float) / start
    series.index = pd.DatetimeIndex(run.equity.index)
    series.name = run.run_id
    return series


def comparison_equity(left: StoredRun, right: StoredRun) -> go.Figure:
    """Overlaid equity curves, each rebased to 1.0 at the start of its own run."""
    fig = go.Figure()
    for run, colour in ((left, GREEN), (right, BLUE)):
        series = normalised_equity(run)
        if series.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.to_numpy(dtype=float),
                name=run.run_id,
                line={"color": colour, "width": 2},
                hovertemplate="%{fullData.name}<br>%{x}<br>%{y:.3f}x<extra></extra>",
            )
        )
    fig.add_hline(y=1.0, line={"color": MUTED, "width": 1, "dash": "dot"})
    fig.update_yaxes(title_text="Equity / initial capital")
    return style(fig, "Equity comparison")


def headline_table(left: StoredRun, right: StoredRun) -> pd.DataFrame:
    """One row per headline metric, with both runs as columns."""
    rows = [
        ("Total return, %", "total_return_pct"),
        ("CAGR, %", "cagr_pct"),
        ("Max drawdown, %", "max_drawdown_pct"),
        ("Sharpe", "sharpe"),
        ("Profit factor", "profit_factor"),
        ("Win rate, %", "win_rate_pct"),
        ("Trades", "trades"),
    ]
    left_h = _headline(left)
    right_h = _headline(right)
    return pd.DataFrame(
        {
            "metric": [label for label, _ in rows],
            left.run_id: [left_h.get(key) for _, key in rows],
            right.run_id: [right_h.get(key) for _, key in rows],
        }
    )


def _headline(run: StoredRun) -> dict[str, float]:
    if not run.metrics:
        return {}
    return PerformanceMetrics.from_dict(run.metrics).headline()
