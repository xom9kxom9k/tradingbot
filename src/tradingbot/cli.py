"""Command line interface built with typer.

Subcommand groups mirror the workflow: ``data`` -> ``backtest`` -> ``optimize`` /
``walkforward`` / ``montecarlo`` -> ``dashboard``, plus ``live`` and ``telegram``
for the running bot.
"""

from __future__ import annotations

import typer

from tradingbot import __version__

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


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
