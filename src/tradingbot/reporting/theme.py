"""Shared Plotly appearance (tech.md section 10.1).

Green is reserved for longs and for money made; red for shorts and for money
lost. Everything else stays quiet so the two colours keep their meaning.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

GREEN = "#26a69a"
RED = "#ef5350"
AMBER = "#ffa726"
BLUE = "#42a5f5"
PURPLE = "#ab47bc"
MUTED = "#8b93a7"
GRID = "#2a2f3a"
PAPER = "#111318"
PLOT = "#161a22"
FONT = "#e6eaf2"

COLOURSCALE_PNL = [
    [0.0, RED],
    [0.5, "#1f2430"],
    [1.0, GREEN],
]


def plotly_template() -> dict[str, Any]:
    """Dark template used by every figure in the report and the dashboard."""
    return {
        "layout": {
            "paper_bgcolor": PAPER,
            "plot_bgcolor": PLOT,
            "font": {"family": "Inter, Helvetica, Arial, sans-serif", "color": FONT, "size": 12},
            "title": {"font": {"size": 16, "color": FONT}, "x": 0.01, "xanchor": "left"},
            "colorway": [GREEN, RED, BLUE, AMBER, PURPLE, MUTED],
            "xaxis": {
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikesnap": "cursor",
                "spikecolor": MUTED,
            },
            "yaxis": {
                "gridcolor": GRID,
                "zerolinecolor": GRID,
                "linecolor": GRID,
                "showspikes": True,
                "spikemode": "across",
                "spikecolor": MUTED,
            },
            "legend": {
                "bgcolor": "rgba(0,0,0,0)",
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "x": 0,
            },
            "hoverlabel": {"bgcolor": PLOT, "font": {"color": FONT}, "bordercolor": GRID},
            "margin": {"l": 56, "r": 24, "t": 64, "b": 48},
            "hovermode": "x unified",
        }
    }


def install_template() -> None:
    """Register the theme as Plotly's default, once per process."""
    pio.templates["tradingbot"] = plotly_template()
    pio.templates.default = "tradingbot"


def style(fig: go.Figure, title: str, *, height: int = 420) -> go.Figure:
    """Apply the theme and a consistent title to a newly built figure."""
    install_template()
    fig.update_layout(template="tradingbot", title=title, height=height)
    return fig


def empty_figure(title: str, message: str) -> go.Figure:
    """A themed placeholder used when a chart has nothing to show yet."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 14, "color": MUTED},
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, title)
