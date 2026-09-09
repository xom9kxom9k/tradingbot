"""Self-contained HTML report for a stored backtest run (FR-5.3).

The file is meant to be emailed or opened from a USB stick: Plotly is inlined,
so nothing has to be running locally besides a browser.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from tradingbot.analytics.metrics import PerformanceMetrics
from tradingbot.backtest.runner import StoredRun
from tradingbot.data.cache import ParquetCache
from tradingbot.reporting.charts import ChartBundle, ExtraCharts, build_charts
from tradingbot.reporting.theme import GREEN, PAPER, PLOT, RED

REPORT_FILE = "report.html"

CARD_KEYS = (
    ("total_return_pct", "Total return", "%"),
    ("cagr_pct", "CAGR", "%"),
    ("max_drawdown_pct", "Max drawdown", "%"),
    ("sharpe", "Sharpe", ""),
    ("profit_factor", "Profit factor", ""),
    ("win_rate_pct", "Win rate", "%"),
    ("trades", "Trades", ""),
)

CHART_TITLES = {
    "equity": "1. Equity curve",
    "underwater": "2. Underwater drawdown",
    "price": "3. Price + signals",
    "monthly": "4. Monthly returns",
    "pnl_distribution": "5. Trade PnL distribution",
    "r_scatter": "6. R-multiple scatter",
    "mae_mfe": "7. MAE / MFE",
    "rolling": "8. Rolling metrics",
    "per_symbol": "9. Per-symbol breakdown",
    "walkforward": "10. Walk-forward",
    "montecarlo": "11. Monte Carlo",
    "parameter_heatmap": "12. Parameter heatmap",
}


def write_report(
    run: StoredRun,
    *,
    prices: Mapping[str, pd.DataFrame] | None = None,
    extras: ExtraCharts | None = None,
    path: Path | None = None,
) -> Path:
    """Render ``report.html`` next to the run's other artefacts.

    Returns:
        The path of the written file.
    """
    target = path or (run.path / REPORT_FILE)
    target.parent.mkdir(parents=True, exist_ok=True)
    bundle = charts_for_run(run, prices=prices, extras=extras)
    target.write_text(render_html(run, bundle), encoding="utf-8")
    logger.info("report written to {path}", path=target)
    return target


def extras_for(run: StoredRun) -> ExtraCharts:
    """Load walk-forward, Monte Carlo and heatmap artefacts sitting next to a run."""
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
    optimize = run.meta.config.get("optimize", {})
    if isinstance(optimize, dict):
        extras.x_label = str(optimize.get("heatmap_x", extras.x_label))
        extras.y_label = str(optimize.get("heatmap_y", extras.y_label))
        extras.metric_label = str(optimize.get("objective", extras.metric_label))
    return extras


def charts_for_run(
    run: StoredRun,
    *,
    prices: Mapping[str, pd.DataFrame] | None = None,
    extras: ExtraCharts | None = None,
) -> ChartBundle:
    """Build the twelve charts for a stored run, loading candles when needed."""
    candles = dict(prices) if prices is not None else load_prices(run)
    resolved = extras if extras is not None else extras_for(run)
    return build_charts(
        equity=run.equity,
        trades=run.trades,
        prices=candles,
        initial_capital=run.initial_capital,
        strategy_params=_strategy_params(run),
        extras=resolved,
    )


def render_html(run: StoredRun, bundle: ChartBundle) -> str:
    """Assemble the full HTML document as a string."""
    metrics = PerformanceMetrics.from_dict(run.metrics) if run.metrics else None
    parts: list[str] = [_head(run), _hero(run, metrics)]
    include_js: str | bool = "inline"
    for name, fig in bundle.ordered():
        title = CHART_TITLES.get(name, name)
        div = fig.to_html(full_html=False, include_plotlyjs=include_js, config=_plotly_config())
        include_js = False
        parts.append(
            f'<section id="chart-{html.escape(name)}" class="chart">'
            f"<h2>{html.escape(title)}</h2>{div}</section>"
        )
    parts.append(_trades_table(run.trades))
    parts.append("</main></body></html>")
    return "\n".join(parts)


def load_prices(run: StoredRun) -> dict[str, pd.DataFrame]:
    """Best-effort load of the candles this run was computed on."""
    data_cfg = run.meta.config.get("data", {})
    exchange_cfg = run.meta.config.get("exchange", {})
    cache_dir = data_cfg.get("cache_dir", "data/cache")
    exchange = exchange_cfg.get("name", "binanceusdm")
    timeframe = run.meta.timeframe or exchange_cfg.get("timeframe", "4h")
    cache = ParquetCache(cache_dir, exchange)
    frames: dict[str, pd.DataFrame] = {}
    for symbol in run.meta.symbols:
        frame = cache.read(symbol, timeframe)
        if not frame.empty:
            frames[symbol] = frame
    return frames


def _strategy_params(run: StoredRun) -> dict[str, Any]:
    strategy = run.meta.config.get("strategy", {})
    if isinstance(strategy, dict):
        return dict(strategy)
    return {}


def _plotly_config() -> dict[str, Any]:
    return {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d"]}


def _head(run: StoredRun) -> str:
    title = html.escape(f"Run {run.run_id}")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin: 0; background: {PAPER}; color: #e6eaf2;
         font-family: Inter, Helvetica, Arial, sans-serif; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 24px 20px 64px; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
  h2 {{ font-size: 1.1rem; margin: 32px 0 8px; color: #c5cbe0; }}
  .meta {{ color: #8b93a7; font-size: 0.9rem; margin-bottom: 20px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px; margin: 16px 0 8px; }}
  .card {{ background: {PLOT}; border: 1px solid #2a2f3a;
           border-radius: 10px; padding: 12px 14px; }}
  .card .label {{ color: #8b93a7; font-size: 0.75rem;
                  text-transform: uppercase; letter-spacing: .04em; }}
  .card .value {{ font-size: 1.25rem; margin-top: 4px; font-variant-numeric: tabular-nums; }}
  .pos {{ color: {GREEN}; }}
  .neg {{ color: {RED}; }}
  .chart {{ margin-top: 8px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem;
           font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid #2a2f3a; }}
  th {{ color: #8b93a7; font-weight: 600; position: sticky; top: 0; background: {PAPER}; }}
  .table-wrap {{ max-height: 480px; overflow: auto;
                 border: 1px solid #2a2f3a; border-radius: 8px; }}
</style>
</head>
<body>
<main>
"""


def _hero(run: StoredRun, metrics: PerformanceMetrics | None) -> str:
    symbols = ", ".join(run.meta.symbols) or "—"
    duration = f"{run.meta.duration_sec:.2f}s" if run.meta.duration_sec is not None else "—"
    meta = (
        f'<div class="meta">{html.escape(symbols)} · {html.escape(run.meta.timeframe)} · '
        f"{html.escape(duration)} · {html.escape(run.meta.code_version)}"
        f"{' · ' + html.escape(run.meta.git_sha) if run.meta.git_sha else ''}</div>"
    )
    heading = f"<h1>Run {html.escape(run.run_id)}</h1>"
    if metrics is None:
        return heading + meta
    headline = metrics.headline()
    cards = []
    for key, label, suffix in CARD_KEYS:
        raw = headline.get(key, 0)
        css = ""
        if key in {"total_return_pct", "cagr_pct", "sharpe"} and isinstance(raw, int | float):
            css = " pos" if raw > 0 else " neg" if raw < 0 else ""
        if key == "max_drawdown_pct":
            css = " neg"
        cards.append(
            f'<div class="card"><div class="label">{html.escape(label)}</div>'
            f'<div class="value{css}">{_fmt(raw, suffix)}</div></div>'
        )
    return heading + meta + '<div class="cards">' + "".join(cards) + "</div>"


def _fmt(value: Any, suffix: str) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return html.escape(str(value))


def _trades_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<h2>Trades</h2><p class='meta'>No closed trades.</p>"
    columns = [
        ("trade_id", "id"),
        ("symbol", "symbol"),
        ("side", "side"),
        ("entry_ts", "entry"),
        ("exit_ts", "exit"),
        ("pnl_net", "pnl"),
        ("pnl_r", "R"),
        ("exit_reason", "exit reason"),
        ("bars_held", "bars"),
    ]
    present = [(src, label) for src, label in columns if src in trades]
    header = "".join(f"<th>{html.escape(label)}</th>" for _, label in present)
    rows = []
    for _, trade in trades.iterrows():
        cells = []
        for src, _ in present:
            cells.append(f"<td>{html.escape(_cell(trade[src]))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<h2>Trades</h2><div class='table-wrap'><table>"
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _cell(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M")
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
