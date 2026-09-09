"""Plotly helpers that only the dashboard uses (session seasonality)."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from tradingbot.dashboard.filters import day_of_week_stats, hour_of_day_stats
from tradingbot.reporting.theme import GREEN, RED, empty_figure, style


def hour_of_day_chart(trades: pd.DataFrame) -> go.Figure:
    """Expectancy in R by UTC hour of the entry."""
    stats = hour_of_day_stats(trades)
    if stats.empty or stats["trades"].sum() == 0:
        return empty_figure("Hour of day", "No trades to group by hour.")
    colours = [GREEN if value >= 0 else RED for value in stats["expectancy_r"]]
    fig = go.Figure(
        data=go.Bar(
            x=stats["hour"],
            y=stats["expectancy_r"],
            marker_color=colours,
            customdata=stats["trades"],
            hovertemplate="Hour %{x}: %{y:+.2f}R · %{customdata} trades<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Hour (UTC)", dtick=1)
    fig.update_yaxes(title_text="Expectancy, R")
    fig.update_layout(hovermode="closest")
    return style(fig, "Hour of day")


def day_of_week_chart(trades: pd.DataFrame) -> go.Figure:
    """Expectancy in R by weekday of the entry."""
    stats = day_of_week_stats(trades)
    if stats.empty or stats["trades"].sum() == 0:
        return empty_figure("Day of week", "No trades to group by weekday.")
    colours = [GREEN if value >= 0 else RED for value in stats["expectancy_r"]]
    fig = go.Figure(
        data=go.Bar(
            x=stats["weekday"],
            y=stats["expectancy_r"],
            marker_color=colours,
            customdata=stats["trades"],
            hovertemplate="%{x}: %{y:+.2f}R · %{customdata} trades<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Expectancy, R")
    fig.update_layout(hovermode="closest")
    return style(fig, "Day of week")
