"""PNG export of the key charts, for Telegram and for embedding elsewhere (FR-5.4)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import plotly.graph_objects as go
from loguru import logger

from tradingbot.core.exceptions import TradingBotError
from tradingbot.reporting.charts import ChartBundle

DEFAULT_CHARTS: tuple[str, ...] = ("equity", "underwater", "price", "monthly")
WIDTH = 1280
HEIGHT = 720


class PngExportError(TradingBotError):
    """Raised when kaleido cannot render a figure."""


def export_pngs(
    bundle: ChartBundle,
    directory: Path,
    *,
    names: Sequence[str] | None = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> list[Path]:
    """Write selected figures as PNG files.

    Args:
        bundle: Figures produced by :func:`build_charts`.
        directory: Destination folder; created if missing.
        names: Chart identifiers to export; defaults to the four Telegram-facing ones.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Paths of the files that were actually written.

    Raises:
        PngExportError: Kaleido is missing or failed to render.
    """
    directory.mkdir(parents=True, exist_ok=True)
    wanted = list(names) if names is not None else list(DEFAULT_CHARTS)
    written: list[Path] = []
    for name in wanted:
        fig = bundle.figures.get(name)
        if fig is None:
            continue
        path = directory / f"{name}.png"
        write_png(fig, path, width=width, height=height)
        written.append(path)
    return written


def write_png(fig: go.Figure, path: Path, *, width: int = WIDTH, height: int = HEIGHT) -> Path:
    """Render one figure to ``path``.

    Raises:
        PngExportError: The renderer is unavailable or rejected the figure.
    """
    path.write_bytes(png_bytes(fig, width=width, height=height))
    logger.debug("png written to {path}", path=path)
    return path


def png_bytes(fig: go.Figure, *, width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Render one figure to PNG bytes.

    Raises:
        PngExportError: The renderer is unavailable or rejected the figure.
    """
    try:
        payload = fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception as exc:  # kaleido raises a variety of errors across versions
        raise PngExportError(f"could not render PNG: {exc}") from exc
    if not isinstance(payload, (bytes, bytearray)):
        payload = bytes(payload)
    return bytes(payload)


def kaleido_available() -> bool:
    """True when a PNG renderer can be imported."""
    try:
        import kaleido  # noqa: F401

        return True
    except ImportError:
        return False
