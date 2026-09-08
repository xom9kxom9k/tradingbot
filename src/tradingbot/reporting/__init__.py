"""Plotly charts, HTML reports and PNG exports."""

from __future__ import annotations

from tradingbot.reporting.charts import (
    CHART_IDS,
    ChartBundle,
    ExtraCharts,
    build_charts,
    marker_timestamps,
)
from tradingbot.reporting.html_report import REPORT_FILE, charts_for_run, render_html, write_report
from tradingbot.reporting.png_export import PngExportError, export_pngs, kaleido_available
from tradingbot.reporting.theme import GREEN, RED, install_template

__all__ = [
    "CHART_IDS",
    "GREEN",
    "RED",
    "REPORT_FILE",
    "ChartBundle",
    "ExtraCharts",
    "PngExportError",
    "build_charts",
    "charts_for_run",
    "export_pngs",
    "install_template",
    "kaleido_available",
    "marker_timestamps",
    "render_html",
    "write_report",
]
