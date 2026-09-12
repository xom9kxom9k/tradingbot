"""Command line interface built with typer.

Subcommand groups mirror the workflow: ``data`` -> ``backtest`` -> ``optimize`` /
``walkforward`` / ``montecarlo`` -> ``dashboard``, plus ``live`` and ``telegram``
for the running bot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
import yaml

from tradingbot import __version__
from tradingbot.backtest import (
    BacktestRunner,
    StoredRun,
    latest_run_id,
    list_runs,
    load_run,
    run_summary,
)
from tradingbot.config import AppConfig, load_config
from tradingbot.config.models import ObjectiveName, WalkForwardMode
from tradingbot.core.exceptions import TradingBotError
from tradingbot.core.logging import setup_logging
from tradingbot.data import CcxtDataFeed, ParquetCache, validate_ohlcv
from tradingbot.reporting.html_report import write_report
from tradingbot.reporting.png_export import PngExportError, export_pngs, kaleido_available

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


@app.command()
def dashboard(
    config: ConfigOption = None,
    port: Annotated[int, typer.Option("--port", help="Port for the Streamlit server.")] = 8501,
    address: Annotated[
        str, typer.Option("--address", help="Bind address; 0.0.0.0 inside Docker.")
    ] = "localhost",
    headless: Annotated[
        bool, typer.Option("--headless", help="Do not open a browser window.")
    ] = False,
) -> None:
    """Open the Streamlit dashboard on stored backtest runs."""
    import tradingbot.dashboard.app as dash_mod

    effective = build_config(config, quiet=True)
    os.environ["TRADINGBOT_RUNS_DIR"] = str(Path(effective.backtest.runs_dir).resolve())
    os.environ["TRADINGBOT_STATE_DB"] = str(Path(effective.live.state_db).resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(Path(dash_mod.__file__).resolve()),
        "--server.port",
        str(port),
        "--server.address",
        address,
    ]
    if headless:
        command.extend(["--server.headless", "true"])
    raise typer.Exit(subprocess.call(command))


@app.command()
def health(config: ConfigOption = None) -> None:
    """Exit 0 when the live runner has marked itself running (Docker healthcheck)."""
    from tradingbot.live.state import LiveStore

    effective = build_config(config, quiet=True)
    status = LiveStore.from_config(effective).read_status()
    if status.health.running:
        typer.echo("ok")
        return
    typer.secho("live runner is not running", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


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


@backtest_app.command("metrics")
def backtest_metrics(
    config: ConfigOption = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run to report on; defaults to the most recent one."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print metrics.json instead.")] = False,
) -> None:
    """Show the metrics of a stored run."""
    effective = build_config(config, quiet=True)
    stored = _load_stored_run(effective, run_id)
    if not stored.metrics:
        typer.secho(f"run {stored.run_id} has no metrics.json", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if as_json:
        typer.echo(json.dumps(stored.metrics, indent=2, sort_keys=True, default=str))
        return
    typer.echo(run_summary(stored))


@backtest_app.command("report")
def backtest_report(
    config: ConfigOption = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run to report on; defaults to the most recent one."),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the report in the default browser.")
    ] = False,
    png: Annotated[
        bool, typer.Option("--png", help="Also export the key charts as PNG files.")
    ] = False,
) -> None:
    """Build a standalone report.html for a stored run."""
    effective = build_config(config)
    stored = _load_stored_run(effective, run_id)
    path = write_report(stored)
    typer.echo(f"report {path}")
    if png:
        from tradingbot.reporting.html_report import charts_for_run

        if not kaleido_available():
            typer.secho("kaleido is not installed; skipping PNG export", fg=typer.colors.YELLOW)
        else:
            try:
                written = export_pngs(charts_for_run(stored), stored.path / "charts")
            except PngExportError as exc:
                typer.secho(str(exc), fg=typer.colors.YELLOW)
            else:
                for file in written:
                    typer.echo(f"png {file}")
    if open_browser:
        import webbrowser

        webbrowser.open(path.resolve().as_uri())


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


def _load_stored_run(config: AppConfig, run_id: str | None) -> StoredRun:
    """Resolve ``run_id`` (or the latest run) and load it, or exit with a message."""
    runs_dir = config.backtest.runs_dir
    target = run_id or latest_run_id(runs_dir)
    if target is None:
        typer.secho(f"no runs found in {runs_dir}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        return load_run(target, runs_dir)
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


ObjectiveOption = Annotated[
    str | None,
    typer.Option(
        "--objective",
        help="Ranking metric: sharpe, calmar, profit_factor or custom.",
    ),
]
JobsOption = Annotated[
    int | None,
    typer.Option("--jobs", help="Parallel trial workers; defaults to the config value."),
]


def _resolve_objective(value: str | None) -> ObjectiveName | None:
    if value is None:
        return None
    allowed: set[str] = {"sharpe", "calmar", "profit_factor", "custom"}
    if value not in allowed:
        typer.secho(
            f"unknown objective {value!r}; choose from {sorted(allowed)}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)
    return value  # type: ignore[return-value]


@optimize_app.command("grid")
def optimize_grid(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    objective: ObjectiveOption = None,
    min_trades: Annotated[
        int | None, typer.Option("--min-trades", help="Reject combinations with fewer trades.")
    ] = None,
    jobs: JobsOption = None,
    note: Annotated[str, typer.Option("--note", help="Free-form label stored in meta.json.")] = "",
) -> None:
    """Exhaustive search over the configured parameter grid; keep the plateau, not the peak."""
    from tradingbot.analytics.optimizer import run_grid, save_optimisation

    effective = build_config(config)
    runner = BacktestRunner(effective)
    try:
        data = runner.load_data(_selected_symbols(effective, symbols) if symbols else None)
        search = run_grid(
            effective,
            data,
            objective=_resolve_objective(objective),
            min_trades=min_trades,
            n_jobs=jobs,
        )
        stored = save_optimisation(
            effective, data, search, method="grid", note=note or "optimize grid"
        )
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _echo_search(search, stored)
    _echo_report(stored)


@optimize_app.command("optuna")
def optimize_optuna(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    trials: Annotated[int | None, typer.Option("--trials", help="Number of TPE trials.")] = None,
    objective: ObjectiveOption = None,
    min_trades: Annotated[
        int | None, typer.Option("--min-trades", help="Reject combinations with fewer trades.")
    ] = None,
    jobs: JobsOption = None,
    note: Annotated[str, typer.Option("--note", help="Free-form label stored in meta.json.")] = "",
) -> None:
    """TPE search over the configured parameter ranges; keep the plateau, not the peak."""
    from tradingbot.analytics.optimizer import run_optuna, save_optimisation

    effective = build_config(config)
    runner = BacktestRunner(effective)
    try:
        data = runner.load_data(_selected_symbols(effective, symbols) if symbols else None)
        search = run_optuna(
            effective,
            data,
            n_trials=trials,
            objective=_resolve_objective(objective),
            min_trades=min_trades,
            n_jobs=jobs,
        )
        stored = save_optimisation(
            effective, data, search, method="optuna", note=note or "optimize optuna"
        )
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _echo_search(search, stored)
    _echo_report(stored)


@walkforward_app.command("run")
def walkforward_run(
    config: ConfigOption = None,
    symbols: SymbolsOption = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="rolling or anchored; defaults to the config value."),
    ] = None,
    is_months: Annotated[
        int | None, typer.Option("--is-months", help="In-sample window length in months.")
    ] = None,
    oos_months: Annotated[
        int | None, typer.Option("--oos-months", help="Out-of-sample window length in months.")
    ] = None,
    step_months: Annotated[
        int | None, typer.Option("--step-months", help="How far to advance each window.")
    ] = None,
    note: Annotated[str, typer.Option("--note", help="Free-form label stored in meta.json.")] = "",
) -> None:
    """Optimise each in-sample window and stitch the frozen-parameter OOS equity."""
    from tradingbot.analytics.walkforward import run_walkforward, save_walkforward

    if mode is not None and mode not in {"rolling", "anchored"}:
        typer.secho("mode must be 'rolling' or 'anchored'", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    wf_mode: WalkForwardMode | None = None if mode is None else mode  # type: ignore[assignment]

    effective = build_config(config)
    runner = BacktestRunner(effective)
    try:
        data = runner.load_data(_selected_symbols(effective, symbols) if symbols else None)
        study = run_walkforward(
            effective,
            data,
            mode=wf_mode,
            is_months=is_months,
            oos_months=oos_months,
            step_months=step_months,
        )
        stored = save_walkforward(effective, data, study, note=note)
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    _echo_walkforward(study, stored)
    _echo_report(stored)


@montecarlo_app.command("run")
def montecarlo_run(
    config: ConfigOption = None,
    run_id: Annotated[
        str | None,
        typer.Option("--run-id", help="Run to shuffle; defaults to the most recent one."),
    ] = None,
    iterations: Annotated[
        int | None, typer.Option("--iterations", help="Number of shuffled paths.")
    ] = None,
) -> None:
    """Shuffle the trade order of a stored run and write a Monte Carlo fan chart."""
    from tradingbot.analytics.monte_carlo import attach_monte_carlo, run_monte_carlo_on_run

    effective = build_config(config)
    stored = _load_stored_run(effective, run_id)
    try:
        result = run_monte_carlo_on_run(stored, effective, iterations=iterations)
        attach_monte_carlo(stored, result)
    except TradingBotError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"monte carlo {stored.run_id}")
    typer.echo(f"  artefacts {stored.path / 'montecarlo.parquet'}")
    finals = result.stats["final_equity"]
    drawdowns = result.stats["max_drawdown"]
    typer.echo(
        f"  final equity  p5={finals['p5']:,.2f}  "
        f"p50={finals['p50']:,.2f}  p95={finals['p95']:,.2f}"
    )
    typer.echo(
        f"  max drawdown  p50={drawdowns['p50']:.2%}  "
        f"P(DD>{result.stats['dd_threshold_pct']:.0%})={result.stats['p_dd_gt_threshold']:.1%}  "
        f"P(ruin)={result.stats['p_ruin']:.1%}"
    )
    _echo_report(stored)


def _echo_search(search: object, stored: StoredRun) -> None:
    from tradingbot.analytics.optimizer import SearchResult

    assert isinstance(search, SearchResult)
    chosen = search.chosen
    typer.echo(run_summary(stored))
    accepted = sum(1 for trial in search.trials if trial.accepted)
    typer.echo(f"  trials    {len(search.trials)} ({accepted} accepted)")
    if search.best and chosen and search.best.params != chosen.params:
        typer.echo(f"  peak      {search.best.params}  score={search.best.score:.3f}")
        typer.echo(f"  plateau   {chosen.params}  score={chosen.score:.3f}")
    elif chosen:
        typer.echo(f"  chosen    {chosen.params}  score={chosen.score:.3f}")


def _echo_walkforward(study: object, stored: StoredRun) -> None:
    from tradingbot.analytics.walkforward import WalkForwardStudy

    assert isinstance(study, WalkForwardStudy)
    typer.echo(run_summary(stored))
    wfe = "n/a" if study.wfe is None else f"{study.wfe:.3f}"
    health = "healthy" if study.healthy else "unhealthy"
    typer.echo(
        f"  walk-forward {study.mode}  windows={len(study.windows)}  "
        f"WFE={wfe}  positive OOS={study.positive_oos_share:.0%}  {health}"
    )
    table = study.windows_frame()
    if not table.empty:
        wanted = ("window", "is_return", "oos_return", "wfe", "is_trades", "oos_trades")
        cols = [column for column in wanted if column in table]
        typer.echo(table[cols].to_string(index=False))


@live_app.command("run")
def live_run(
    config: ConfigOption = None,
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="paper or signal_only; defaults to the config value."),
    ] = None,
) -> None:
    """Start the live loop and wait for candle closes (Ctrl+C to stop)."""
    import asyncio

    from tradingbot.live.runner import LiveRunner

    effective = build_config(config)
    if mode is not None:
        if mode not in {"paper", "signal_only"}:
            typer.secho("mode must be 'paper' or 'signal_only'", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        effective = effective.model_copy(
            update={"live": effective.live.model_copy(update={"mode": mode})}
        )
    asyncio.run(LiveRunner(effective).run())


@live_app.command("status")
def live_status(config: ConfigOption = None) -> None:
    """Print health, last processed bars and open positions from the state database."""
    from tradingbot.live.state import LiveStore, format_status

    effective = build_config(config, quiet=True)
    store = LiveStore.from_config(effective)
    typer.echo(format_status(store.read_status()))


def _echo_report(stored: StoredRun) -> None:
    try:
        path = write_report(stored)
    except Exception as exc:  # pragma: no cover - report must never fail the study
        typer.secho(f"report skipped: {exc}", fg=typer.colors.YELLOW, err=True)
        return
    typer.echo(f"report {path}")


@telegram_app.command("test")
def telegram_test(config: ConfigOption = None) -> None:
    """Send a test message to every whitelisted chat_id."""
    import asyncio

    from tradingbot.config.loader import load_secrets
    from tradingbot.core.exceptions import NotificationError
    from tradingbot.notify.telegram import TelegramNotifier

    effective = build_config(config, quiet=True)
    secrets = load_secrets()
    if not (effective.notify.telegram_enabled and secrets.enabled and secrets.configured):
        typer.secho(
            "Telegram is not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS in .env.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    async def _send() -> None:
        notifier = TelegramNotifier(
            secrets, max_per_minute=effective.notify.max_messages_per_minute
        )
        try:
            await notifier.send_test()
        finally:
            await notifier.close()

    try:
        asyncio.run(_send())
    except NotificationError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc
    typer.echo("test message sent")


if __name__ == "__main__":  # pragma: no cover
    app()
