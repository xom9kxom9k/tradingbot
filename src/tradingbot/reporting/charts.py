"""Plotly factories for the twelve charts of tech.md section 10.2.

Each function is a pure transformation of already-computed artefacts. The HTML
report and the Streamlit dashboard call the same factories, so a chart cannot
look different in one place than in the other.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tradingbot.analytics.metrics import drawdown_episodes, drawdown_series, profit_factor
from tradingbot.core.enums import Side
from tradingbot.indicators import atr, donchian, ema
from tradingbot.reporting.theme import (
    AMBER,
    BLUE,
    COLOURSCALE_PNL,
    GREEN,
    MUTED,
    PURPLE,
    RED,
    empty_figure,
    style,
)

ROLLING_WINDOW_DAYS = 90
CHART_IDS: tuple[str, ...] = (
    "equity",
    "underwater",
    "price",
    "monthly",
    "pnl_distribution",
    "r_scatter",
    "mae_mfe",
    "rolling",
    "per_symbol",
    "walkforward",
    "montecarlo",
    "parameter_heatmap",
)


@dataclass(slots=True)
class ExtraCharts:
    """Optional artefacts produced by later stages (walk-forward, Monte Carlo)."""

    walkforward_windows: pd.DataFrame | None = None
    walkforward_equity: pd.Series | None = None
    montecarlo_paths: pd.DataFrame | None = None
    parameter_grid: pd.DataFrame | None = None
    x_label: str = "x"
    y_label: str = "y"
    metric_label: str = "metric"


@dataclass(slots=True)
class ChartBundle:
    """Named figures in the order of section 10.2."""

    figures: dict[str, go.Figure] = field(default_factory=dict)

    def ordered(self) -> list[tuple[str, go.Figure]]:
        """Figures in the specification's numbering, skipping unknown keys."""
        return [(name, self.figures[name]) for name in CHART_IDS if name in self.figures]


def build_charts(
    *,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    prices: Mapping[str, pd.DataFrame] | None = None,
    initial_capital: float = 10_000.0,
    symbol: str | None = None,
    strategy_params: Mapping[str, Any] | None = None,
    extras: ExtraCharts | None = None,
) -> ChartBundle:
    """Build every chart of section 10.2 for one run."""
    extras = extras or ExtraCharts()
    curve = _equity_series(equity)
    chosen = symbol or _first_symbol(trades, prices)
    candles = pd.DataFrame()
    if chosen and prices is not None:
        candles = prices.get(chosen, pd.DataFrame())
    bundle = ChartBundle(
        figures={
            "equity": equity_curve(curve, prices or {}, initial_capital),
            "underwater": underwater(curve),
            "price": price_signals(
                candles,
                trades,
                symbol=chosen,
                params=strategy_params or {},
            ),
            "monthly": monthly_heatmap(curve),
            "pnl_distribution": pnl_distribution(trades),
            "r_scatter": r_multiple_scatter(trades),
            "mae_mfe": mae_mfe_scatter(trades),
            "rolling": rolling_metrics(curve, trades),
            "per_symbol": per_symbol_breakdown(trades),
            "walkforward": walkforward_chart(extras.walkforward_windows, extras.walkforward_equity),
            "montecarlo": montecarlo_fan(extras.montecarlo_paths),
            "parameter_heatmap": parameter_heatmap(
                extras.parameter_grid,
                x_label=extras.x_label,
                y_label=extras.y_label,
                metric_label=extras.metric_label,
            ),
        }
    )
    return bundle


# --------------------------------------------------------------------------- #
# 1. Equity curve
# --------------------------------------------------------------------------- #


def equity_curve(
    equity: pd.Series,
    prices: Mapping[str, pd.DataFrame],
    initial_capital: float,
) -> go.Figure:
    """Capital over time, with an equal-weight buy-and-hold overlay."""
    if equity.empty:
        return empty_figure("Equity curve", "No equity curve was recorded for this run.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.to_numpy(dtype=float),
            name="Strategy",
            line={"color": GREEN, "width": 2},
            hovertemplate="Strategy: %{y:,.2f}<extra></extra>",
        )
    )
    benchmark = buy_and_hold_equity(prices, initial_capital, pd.DatetimeIndex(equity.index))
    if not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark.index,
                y=benchmark.to_numpy(dtype=float),
                name="Buy & hold",
                line={"color": MUTED, "width": 1.5, "dash": "dot"},
                hovertemplate="Buy & hold: %{y:,.2f}<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Equity")
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 1,
                "xanchor": "right",
                "y": 1.16,
                "buttons": [
                    {"label": "Linear", "method": "relayout", "args": [{"yaxis.type": "linear"}]},
                    {"label": "Log", "method": "relayout", "args": [{"yaxis.type": "log"}]},
                ],
            }
        ]
    )
    return style(fig, "Equity curve")


def buy_and_hold_equity(
    prices: Mapping[str, pd.DataFrame],
    initial_capital: float,
    index: pd.DatetimeIndex,
) -> pd.Series:
    """Equal-weight buy-and-hold path aligned to the strategy's timestamps."""
    growth: list[pd.Series] = []
    for frame in prices.values():
        if frame.empty or "close" not in frame:
            continue
        close = frame["close"].astype(float)
        start = float(close.iloc[0])
        if start <= 0:
            continue
        growth.append(close / start)
    if not growth:
        return pd.Series(dtype=float)
    average = pd.concat(growth, axis=1).mean(axis=1)
    aligned = average.reindex(index).ffill().bfill()
    return aligned * initial_capital


# --------------------------------------------------------------------------- #
# 2. Underwater / drawdown
# --------------------------------------------------------------------------- #


def underwater(equity: pd.Series) -> go.Figure:
    """Drawdown through time, with the deepest episode highlighted."""
    if equity.empty:
        return empty_figure("Drawdown", "No equity curve was recorded for this run.")

    dd = drawdown_series(equity) * 100.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=-dd.to_numpy(dtype=float),
            name="Drawdown",
            fill="tozeroy",
            line={"color": RED, "width": 1},
            fillcolor="rgba(239, 83, 80, 0.35)",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        )
    )
    episodes = drawdown_episodes(equity)
    if episodes:
        deepest = max(episodes, key=lambda item: item.depth)
        end = deepest.recovery_ts or equity.index[-1]
        fig.add_vrect(
            x0=deepest.peak_ts,
            x1=end,
            fillcolor=RED,
            opacity=0.12,
            line_width=0,
            annotation_text=f"Max DD {deepest.depth:.1%}",
            annotation_position="top left",
        )
    fig.update_yaxes(title_text="Drawdown, %")
    return style(fig, "Underwater drawdown")


# --------------------------------------------------------------------------- #
# 3. Price + signals
# --------------------------------------------------------------------------- #


def price_signals(
    candles: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    symbol: str | None,
    params: Mapping[str, Any],
) -> go.Figure:
    """Candles, indicators, hold zones and trade markers for one instrument."""
    if candles.empty or not {"open", "high", "low", "close"} <= set(candles.columns):
        return empty_figure(
            "Price + signals",
            "No OHLCV series was supplied, so the price chart cannot be drawn.",
        )

    frame = _with_indicators(candles, params)
    scoped = _trades_for(trades, symbol)
    fig = go.Figure()

    for _, trade in scoped.iterrows():
        colour = GREEN if str(trade["side"]) == Side.LONG.value else RED
        fig.add_vrect(
            x0=trade["entry_ts"],
            x1=trade["exit_ts"],
            fillcolor=colour,
            opacity=0.07,
            line_width=0,
            layer="below",
        )
        fig.add_shape(
            type="line",
            x0=trade["entry_ts"],
            x1=trade["exit_ts"],
            y0=float(trade["initial_stop"]),
            y1=float(trade["initial_stop"]),
            line={"color": RED, "width": 1, "dash": "dot"},
        )

    fig.add_trace(
        go.Candlestick(
            x=frame.index,
            open=frame["open"],
            high=frame["high"],
            low=frame["low"],
            close=frame["close"],
            name="OHLC",
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
        )
    )
    _overlay_line(fig, frame, "ema_fast", f"EMA {params.get('trend_fast', 50)}", BLUE)
    _overlay_line(fig, frame, "ema_slow", f"EMA {params.get('trend_slow', 200)}", PURPLE)
    _overlay_line(fig, frame, "dc_upper", "Donchian upper", AMBER, dash="dash")
    _overlay_line(fig, frame, "dc_lower", "Donchian lower", AMBER, dash="dash")

    _add_trade_markers(fig, scoped)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price")
    title = f"Price + signals — {symbol}" if symbol else "Price + signals"
    return style(fig, title, height=520)


def _add_trade_markers(fig: go.Figure, trades: pd.DataFrame) -> None:
    """Entry triangles and exit crosses whose timestamps match the journal."""
    if trades.empty:
        return
    longs = trades[trades["side"] == Side.LONG.value]
    shorts = trades[trades["side"] == Side.SHORT.value]
    if not longs.empty:
        fig.add_trace(
            go.Scatter(
                x=longs["entry_ts"],
                y=longs["entry_price"],
                mode="markers",
                name="Long entry",
                marker={"symbol": "triangle-up", "size": 11, "color": GREEN},
                customdata=np.column_stack(
                    [longs["trade_id"].astype(str), longs["pnl_r"].astype(float)]
                ),
                hovertemplate=(
                    "Long entry %{customdata[0]}<br>%{x}<br>%{y:.6g}"
                    "<br>%{customdata[1]:+.2f}R<extra></extra>"
                ),
            )
        )
    if not shorts.empty:
        fig.add_trace(
            go.Scatter(
                x=shorts["entry_ts"],
                y=shorts["entry_price"],
                mode="markers",
                name="Short entry",
                marker={"symbol": "triangle-down", "size": 11, "color": RED},
                customdata=np.column_stack(
                    [shorts["trade_id"].astype(str), shorts["pnl_r"].astype(float)]
                ),
                hovertemplate=(
                    "Short entry %{customdata[0]}<br>%{x}<br>%{y:.6g}"
                    "<br>%{customdata[1]:+.2f}R<extra></extra>"
                ),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=trades["exit_ts"],
            y=trades["exit_price"],
            mode="markers",
            name="Exit",
            marker={"symbol": "x", "size": 9, "color": AMBER},
            customdata=np.column_stack(
                [
                    trades["trade_id"].astype(str),
                    trades["exit_reason"].astype(str),
                    trades["pnl_r"].astype(float),
                ]
            ),
            hovertemplate=(
                "Exit %{customdata[0]} (%{customdata[1]})<br>%{x}<br>%{y:.6g}"
                "<br>%{customdata[2]:+.2f}R<extra></extra>"
            ),
        )
    )


def _with_indicators(candles: pd.DataFrame, params: Mapping[str, Any]) -> pd.DataFrame:
    """Add EMA / Donchian / ATR columns when the caller did not precompute them."""
    frame = candles.copy()
    close, high, low = frame["close"], frame["high"], frame["low"]
    fast = int(params.get("trend_fast", 50))
    slow = int(params.get("trend_slow", 200))
    channel = int(params.get("entry_channel", 20))
    atr_period = int(params.get("atr_period", 14))
    if "ema_fast" not in frame:
        frame["ema_fast"] = ema(close, fast)
    if "ema_slow" not in frame:
        frame["ema_slow"] = ema(close, slow)
    if "dc_upper" not in frame or "dc_lower" not in frame:
        bands = donchian(high, low, channel)
        frame["dc_upper"] = bands["dc_upper"]
        frame["dc_lower"] = bands["dc_lower"]
    if "atr" not in frame:
        frame["atr"] = atr(high, low, close, atr_period)
    return frame


def _overlay_line(
    fig: go.Figure,
    frame: pd.DataFrame,
    column: str,
    name: str,
    colour: str,
    *,
    dash: str | None = None,
) -> None:
    if column not in frame:
        return
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame[column],
            name=name,
            line={"color": colour, "width": 1, "dash": dash or "solid"},
            hovertemplate=f"{name}: %{{y:.6g}}<extra></extra>",
        )
    )


# --------------------------------------------------------------------------- #
# 4. Monthly returns heatmap
# --------------------------------------------------------------------------- #


def monthly_heatmap(equity: pd.Series) -> go.Figure:
    """Calendar heatmap of monthly strategy returns, in percent."""
    if len(equity) < 2:
        return empty_figure("Monthly returns", "Not enough equity points to form months.")

    monthly = equity.resample("ME").last().dropna()
    previous = monthly.shift(1)
    previous.iloc[0] = equity.iloc[0]
    returns = (monthly / previous - 1.0) * 100.0
    index = pd.DatetimeIndex(returns.index)
    grid = pd.DataFrame(
        {"year": index.year, "month": index.month, "value": returns.to_numpy(dtype=float)}
    )
    pivot = grid.pivot_table(index="year", columns="month", values="value")
    months = list(range(1, 13))
    pivot = pivot.reindex(columns=months)
    z = pivot.to_numpy(dtype=float)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=pivot.index.astype(str),
            colorscale=COLOURSCALE_PNL,
            zmid=0,
            hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
            colorbar={"title": "%"},
        )
    )
    return style(fig, "Monthly returns")


# --------------------------------------------------------------------------- #
# 5-7. Trade scatter charts
# --------------------------------------------------------------------------- #


def pnl_distribution(trades: pd.DataFrame) -> go.Figure:
    """Histogram of trade outcomes in R, with the expectancy marked."""
    if trades.empty or "pnl_r" not in trades:
        return empty_figure("Trade PnL distribution", "No closed trades in this run.")
    values = trades["pnl_r"].astype(float)
    fig = go.Figure(
        data=go.Histogram(
            x=values,
            name="PnL, R",
            marker_color=BLUE,
            nbinsx=min(40, max(8, len(values))),
            hovertemplate="%{x:.2f}R — %{y} trades<extra></extra>",
        )
    )
    expectancy = float(values.mean())
    fig.add_vline(
        x=expectancy,
        line={"color": AMBER, "width": 2, "dash": "dash"},
        annotation_text=f"E[R] = {expectancy:.2f}",
        annotation_position="top right",
    )
    fig.update_xaxes(title_text="PnL, R")
    fig.update_yaxes(title_text="Trades")
    fig.update_layout(hovermode="closest")
    return style(fig, "Trade PnL distribution")


def r_multiple_scatter(trades: pd.DataFrame) -> go.Figure:
    """Each trade's R-multiple over time, split by side."""
    if trades.empty:
        return empty_figure("R-multiple scatter", "No closed trades in this run.")
    fig = go.Figure()
    for side, colour, symbol in (
        (Side.LONG.value, GREEN, "triangle-up"),
        (Side.SHORT.value, RED, "triangle-down"),
    ):
        subset = trades[trades["side"] == side]
        if subset.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=subset["exit_ts"],
                y=subset["pnl_r"],
                mode="markers",
                name=side,
                marker={"color": colour, "size": 8, "symbol": symbol},
                customdata=subset["trade_id"].astype(str),
                hovertemplate="%{customdata}<br>%{x}<br>%{y:+.2f}R<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": MUTED, "width": 1})
    fig.update_yaxes(title_text="PnL, R")
    fig.update_layout(hovermode="closest")
    return style(fig, "R-multiple scatter")


def mae_mfe_scatter(trades: pd.DataFrame) -> go.Figure:
    """Adverse vs favourable excursion, the usual stop/target quality plot."""
    if trades.empty or not {"mae", "mfe"} <= set(trades.columns):
        return empty_figure("MAE / MFE", "No closed trades in this run.")
    colours = [GREEN if pnl > 0 else RED for pnl in trades["pnl_net"].astype(float)]
    fig = go.Figure(
        data=go.Scatter(
            x=trades["mae"],
            y=trades["mfe"],
            mode="markers",
            name="Trades",
            marker={"color": colours, "size": 8},
            customdata=np.column_stack(
                [trades["trade_id"].astype(str), trades["pnl_r"].astype(float)]
            ),
            hovertemplate=(
                "%{customdata[0]}<br>MAE %{x:.2f}R · MFE %{y:.2f}R"
                "<br>%{customdata[1]:+.2f}R<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="MAE, R")
    fig.update_yaxes(title_text="MFE, R")
    fig.update_layout(hovermode="closest")
    return style(fig, "MAE / MFE scatter")


# --------------------------------------------------------------------------- #
# 8. Rolling metrics
# --------------------------------------------------------------------------- #


def rolling_metrics(
    equity: pd.Series, trades: pd.DataFrame, window_days: int = ROLLING_WINDOW_DAYS
) -> go.Figure:
    """Trailing Sharpe, win rate and profit factor over a calendar window."""
    if equity.empty:
        return empty_figure("Rolling metrics", "No equity curve was recorded for this run.")

    daily = equity.resample("1D").last().dropna()
    returns = daily.pct_change()
    min_obs = max(10, window_days // 6)
    rolling_std = returns.rolling(window_days, min_periods=min_obs).std()
    rolling_mean = returns.rolling(window_days, min_periods=min_obs).mean()
    sharpe = (rolling_mean / rolling_std.replace(0.0, np.nan)) * math.sqrt(365.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sharpe.index,
            y=sharpe,
            name="Sharpe (90d)",
            line={"color": BLUE, "width": 2},
            hovertemplate="Sharpe: %{y:.2f}<extra></extra>",
        )
    )
    if not trades.empty:
        win_rate, factor = _rolling_trade_stats(trades, pd.DatetimeIndex(daily.index), window_days)
        fig.add_trace(
            go.Scatter(
                x=win_rate.index,
                y=win_rate * 100.0,
                name="Win rate, %",
                yaxis="y2",
                line={"color": GREEN, "width": 1.5},
                hovertemplate="Win rate: %{y:.1f}%<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=factor.index,
                y=factor,
                name="Profit factor",
                yaxis="y2",
                line={"color": AMBER, "width": 1.5, "dash": "dot"},
                hovertemplate="Profit factor: %{y:.2f}<extra></extra>",
            )
        )
        fig.update_layout(
            yaxis2={"overlaying": "y", "side": "right", "showgrid": False, "title": "Win rate / PF"}
        )
    fig.update_yaxes(title_text="Sharpe")
    return style(fig, f"Rolling metrics ({window_days}d)")


def _rolling_trade_stats(
    trades: pd.DataFrame, index: pd.DatetimeIndex, window_days: int
) -> tuple[pd.Series, pd.Series]:
    """Win rate and profit factor of trades that closed inside a trailing window."""
    closed = trades.copy()
    exits = pd.to_datetime(closed["exit_ts"], utc=True)
    pnl = closed["pnl_net"].astype(float)
    rates: list[float] = []
    factors: list[float] = []
    for moment in pd.DatetimeIndex(index):
        start = moment - pd.Timedelta(days=window_days)
        sample = pnl[(exits > start) & (exits <= moment)]
        if sample.size == 0:
            rates.append(float("nan"))
            factors.append(float("nan"))
            continue
        rates.append(float((sample > 0).mean()))
        factors.append(profit_factor(pd.Series(sample)))
    return pd.Series(rates, index=index), pd.Series(factors, index=index)


# --------------------------------------------------------------------------- #
# 9. Per-symbol breakdown
# --------------------------------------------------------------------------- #


def per_symbol_breakdown(trades: pd.DataFrame) -> go.Figure:
    """Net PnL contribution of each instrument."""
    if trades.empty or "symbol" not in trades:
        return empty_figure("Per-symbol breakdown", "No closed trades in this run.")
    grouped = trades.groupby("symbol", sort=True)["pnl_net"].sum().sort_values()
    colours = [GREEN if value >= 0 else RED for value in grouped]
    fig = go.Figure(
        data=go.Bar(
            x=grouped.to_numpy(dtype=float),
            y=grouped.index.astype(str),
            orientation="h",
            marker_color=colours,
            hovertemplate="%{y}: %{x:,.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Net PnL")
    fig.update_layout(hovermode="closest")
    return style(fig, "Per-symbol breakdown")


# --------------------------------------------------------------------------- #
# 10-12. Later-stage charts (honest placeholders until those stages land)
# --------------------------------------------------------------------------- #


def walkforward_chart(windows: pd.DataFrame | None, oos_equity: pd.Series | None) -> go.Figure:
    """In-sample vs out-of-sample returns per window, plus the stitched OOS curve."""
    if windows is None or windows.empty:
        return empty_figure(
            "Walk-forward",
            "This run has no walk-forward windows.<br>Produce them with `tradingbot walkforward`.",
        )
    labels = windows["window"].astype(str) if "window" in windows else windows.index.astype(str)
    fig = go.Figure()
    if "is_return" in windows:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=windows["is_return"] * 100.0,
                name="In-sample",
                marker_color=MUTED,
                hovertemplate="IS: %{y:.2f}%<extra></extra>",
            )
        )
    if "oos_return" in windows:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=windows["oos_return"] * 100.0,
                name="Out-of-sample",
                marker_color=GREEN,
                hovertemplate="OOS: %{y:.2f}%<extra></extra>",
            )
        )
    fig.update_yaxes(title_text="Return, %")
    fig.update_layout(barmode="group", hovermode="x unified")
    if oos_equity is not None and not oos_equity.empty:
        fig.add_trace(
            go.Scatter(
                x=oos_equity.index,
                y=oos_equity,
                name="Stitched OOS equity",
                yaxis="y2",
                line={"color": BLUE, "width": 2},
            )
        )
        fig.update_layout(yaxis2={"overlaying": "y", "side": "right", "title": "Equity"})
    return style(fig, "Walk-forward")


def montecarlo_fan(paths: pd.DataFrame | None) -> go.Figure:
    """Fan of shuffled-trade equity paths with 5 / 50 / 95 percentiles."""
    if paths is None or paths.empty:
        return empty_figure(
            "Monte Carlo",
            "This run has no Monte Carlo paths.<br>Produce them with `tradingbot montecarlo`.",
        )
    matrix = paths.to_numpy(dtype=float)
    p05 = np.percentile(matrix, 5, axis=1)
    p50 = np.percentile(matrix, 50, axis=1)
    p95 = np.percentile(matrix, 95, axis=1)
    x = paths.index
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p95,
            name="95th",
            line={"color": GREEN, "width": 1},
            hovertemplate="95th: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p05,
            name="5th",
            line={"color": RED, "width": 1},
            fill="tonexty",
            fillcolor="rgba(38, 166, 154, 0.15)",
            hovertemplate="5th: %{y:,.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=p50,
            name="Median",
            line={"color": BLUE, "width": 2},
            hovertemplate="Median: %{y:,.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Equity")
    return style(fig, "Monte Carlo fan")


def parameter_heatmap(
    grid: pd.DataFrame | None,
    *,
    x_label: str = "x",
    y_label: str = "y",
    metric_label: str = "metric",
) -> go.Figure:
    """Sensitivity of one metric to a pair of parameters."""
    if grid is None or grid.empty or not {"x", "y", "value"} <= set(grid.columns):
        return empty_figure(
            "Parameter heatmap",
            "This run has no sensitivity grid.<br>Produce one with `tradingbot optimize`.",
        )
    pivot = grid.pivot_table(index="y", columns="x", values="value")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="Viridis",
            hovertemplate=(
                f"{x_label}=%{{x}} · {y_label}=%{{y}}<br>{metric_label}: %{{z:.3f}}<extra></extra>"
            ),
            colorbar={"title": metric_label},
        )
    )
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(hovermode="closest")
    return style(fig, "Parameter heatmap")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _equity_series(equity: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(equity, pd.Series):
        return equity.astype(float)
    if equity.empty:
        return pd.Series(dtype=float)
    column = "equity" if "equity" in equity.columns else equity.columns[0]
    series = equity[column].astype(float)
    series.index = pd.DatetimeIndex(equity.index)
    return series


def _first_symbol(trades: pd.DataFrame, prices: Mapping[str, pd.DataFrame] | None) -> str | None:
    if prices:
        return next(iter(prices))
    if not trades.empty and "symbol" in trades:
        return str(trades["symbol"].iloc[0])
    return None


def _trades_for(trades: pd.DataFrame, symbol: str | None) -> pd.DataFrame:
    if trades.empty:
        return trades
    if symbol is None or "symbol" not in trades:
        return trades
    return trades[trades["symbol"] == symbol]


def marker_timestamps(fig: go.Figure, name: str) -> list[Any]:
    """X values of a named scatter, used by tests to check marker ↔ journal alignment."""
    for trace in fig.data:
        if getattr(trace, "name", None) == name:
            return list(trace.x)
    return []


def required_chart_ids() -> Sequence[str]:
    """The twelve identifiers of section 10.2, in order."""
    return CHART_IDS
