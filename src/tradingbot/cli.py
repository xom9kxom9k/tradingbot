"""Command line interface built with typer.

Subcommand groups mirror the workflow: ``data`` -> ``backtest`` -> ``optimize`` /
``walkforward`` / ``montecarlo`` -> ``dashboard``, plus ``live`` and ``telegram``
for the running bot.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tradingbot import __version__
from tradingbot.backtest import BacktestRunner, list_runs, run_summary
from tradingbot.config import AppConfig, load_config
from tradingbot.core.exceptions import TradingBotError
from tradingbot.core.logging import setup_logging
from tradingbot.data import CcxtDataFeed, ParquetCache, validate_ohlcv

ConfigOption = Annotated[
    list[Path] | None,
    typer.Option(
        "--config",
        "-c",
        help="YAML config file; repeat to layer several files (later wins).",
    ),
]

app = typer.Typer(
    name="tradingbot",
    help="Automated trading bot: data, backtests, analytics and Telegram signals.",
    no_args_is_help=True,
    add_completion=False,
)

data_app = typer.Typer(help="Download, validate and inspect OHLCV data.", no_args_is_help=True)
backtest_app = typer.Typer(help="Run backtests and build reports.", no_args_is_help=True)
optimize_app = typer.Typer(help="Search strategy parameters.", no_args_is_help=True)
walkforward_app = typer.Typer(help="Walk-forward validation.", no_args_is_help=True)
montecarlo_app = typer.Typer(help="Monte Carlo robustness checks.", no_args_is_help=True)
live_app = typer.Typer(help="Run and inspect the live/paper bot.", no_args_is_help=True)
telegram_app = typer.Typer(help="Telegram utilities.", no_args_is_help=True)
config_app = typer.Typer(help="Inspect the effective configuration.", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(optimize_app, name="optimize")
app.add_typer(walkforward_app, name="walkforward")
app.add_typer(montecarlo_app, name="montecarlo")
app.add_typer(live_app, name="live")
app.add_typer(telegram_app, name="telegram")
app.add_typer(config_app, name="config")


def build_config(paths: list[Path] | None, *, quiet: bool = False) -> AppConfig:
    """Load the effective config and install logging, or exit with a clear message.

    Args:
        paths: Config files supplied on the command line, if any.
        quiet: Keep the console sink silent (used by commands that print data).

    Returns:
        The validated configuration.
    """
    try:
        config = load_config(paths or None)
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    setup_logging(config.logging, console=not quiet)
    return config


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@config_app.command("show")
def config_show(config: ConfigOption = None) -> None:
    """Print the effective configuration after merging YAML, env and defaults."""
    effective = build_config(config, quiet=True)
    typer.echo(
        yaml.safe_dump(effective.to_yaml_dict(), sort_keys=False, allow_unicode=True).rstrip()
    )


SymbolsOption = Annotated[
    str | None,
    typer.Option("--symbols", help="Comma-separated symbols; defaults to the configured list."),
]
TimeframeOption = Annotated[
    str | None,
    typer.Option("--timeframe", help="Candle size; defaults to the configured one."),
]


def _selected_symbols(config: AppConfig, symbols: str | None) -> list[str]:
    """Resolve the symbol list from the CLI flag, falling back to the config."""
    if not symbols:
        return list(config.exchange.symbols)
    return [item.strip() for item in symbols.split(",") if item.strip()]


@data_app.command("download")
def data_download(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    timeframe: TimeframeOption = None,
    start: Annotated[
        str | None, typer.Option("--start", help="First candle, e.g. 2019-01-01.")
    ] = None,
) -> None:
    """Download missing candles into the local Parquet cache."""
    effective = build_config(config)
    selected = _selected_symbols(effective, symbols)
    candle = timeframe or effective.exchange.timeframe
    begin = datetime.fromisoformat(start).replace(tzinfo=UTC) if start else effective.data.start

    cache = ParquetCache(effective.data.cache_dir, effective.exchange.name)
    with CcxtDataFeed(effective.exchange.name) as feed:
        for symbol in selected:
            try:
                result = cache.sync(feed, symbol, candle, begin, effective.data.end)
            except TradingBotError as exc:
                typer.secho(f"{symbol}: {exc}", fg=typer.colors.RED, err=True)
                continue
            typer.echo(
                f"{symbol} {candle}: {result.total} bars (+{result.added} new) -> {result.path}"
            )


@data_app.command("validate")
def data_validate(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    timeframe: TimeframeOption = None,
) -> None:
    """Check cached data against the quality rules and print a report."""
    effective = build_config(config, quiet=True)
    candle = timeframe or effective.exchange.timeframe
    cache = ParquetCache(effective.data.cache_dir, effective.exchange.name)

    failed = False
    for symbol in _selected_symbols(effective, symbols):
        report = validate_ohlcv(
            cache.read(symbol, candle),
            symbol,
            candle,
            max_gap_bars=effective.data.max_gap_bars,
        )
        typer.echo(report.render())
        failed = failed or not report.is_valid
    if failed:
        raise typer.Exit(code=1)


@backtest_app.command("run")
def backtest_run(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    note: Annotated[str, typer.Option("--note", help="Free-form label stored in meta.json.")] = "",
) -> None:
    """Run a backtest over the cached data and store the artefacts."""
    effective = build_config(config)
    runner = BacktestRunner(effective)
    try:
        stored = runner.run(_selected_symbols(effective, symbols) if symbols else None, note=note)
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(run_summary(stored))


@backtest_app.command("list")
def backtest_list(config: ConfigOption = None) -> None:
    """List stored runs, newest first."""
    effective = build_config(config, quiet=True)
    runs = list_runs(effective.backtest.runs_dir)
    if not runs:
        typer.echo("no runs stored yet")
        return
    for run_id in runs:
        typer.echo(run_id)


@data_app.command("info")
def data_info(config: ConfigOption = None, timeframe: TimeframeOption = None) -> None:
    """List what is currently cached for each configured symbol."""
    effective = build_config(config, quiet=True)
    candle = timeframe or effective.exchange.timeframe
    cache = ParquetCache(effective.data.cache_dir, effective.exchange.name)

    for symbol in effective.exchange.symbols:
        frame = cache.read(symbol, candle)
        if frame.empty:
            typer.echo(f"{symbol} {candle}: not cached")
            continue
        typer.echo(
            f"{symbol} {candle}: {len(frame)} bars, "
            f"{frame.index[0]:%Y-%m-%d} .. {frame.index[-1]:%Y-%m-%d %H:%M} UTC, "
            f"hash {cache.fingerprint(symbol, candle)}"
        )


if __name__ == "__main__":  # pragma: no cover
    app()
