"""PNG previews of the dashboard views, used in the README."""

from __future__ import annotations

from pathlib import Path

from tradingbot.backtest.runner import StoredRun
from tradingbot.reporting.html_report import charts_for_run, load_prices
from tradingbot.reporting.png_export import write_png


def export_previews(run: StoredRun, directory: Path) -> list[Path]:
    """Write one PNG per dashboard page into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    bundle = charts_for_run(run, prices=load_prices(run))
    mapping = {
        "overview.png": bundle.figures["equity"],
        "chart.png": bundle.figures["price"],
        "analytics.png": bundle.figures["monthly"],
        "walkforward.png": bundle.figures["walkforward"],
    }
    return [write_png(figure, directory / name) for name, figure in mapping.items()]
