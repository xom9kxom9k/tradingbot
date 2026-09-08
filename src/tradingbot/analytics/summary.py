"""Console rendering of a metrics set.

Plain text on purpose: the same output has to stay readable in a terminal, in a
CI log and when pasted into a chat message.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from tradingbot.analytics.metrics import PerformanceMetrics

LABEL_WIDTH = 30


def _number(value: Any, suffix: str = "", digits: int = 2) -> str:
    """Format a value for the summary, keeping infinities readable."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}{suffix}"
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "n/a"
        return f"{value:,.{digits}f}{suffix}"
    return str(value)


def _section(title: str, rows: Iterable[tuple[str, str]]) -> list[str]:
    lines = [title]
    lines.extend(f"  {label:<{LABEL_WIDTH}}{value:>16}" for label, value in rows)
    return lines


def format_metrics(metrics: PerformanceMetrics) -> str:
    """Render the headline blocks of a metrics set as an aligned table."""
    returns, risk = metrics.returns, metrics.risk
    ratios, trading, costs = metrics.ratios, metrics.trading, metrics.costs

    blocks = [
        _section(
            "Returns",
            [
                ("Total return", _number(returns["total_return_pct"], "%")),
                ("CAGR", _number(returns["cagr_pct"], "%")),
                ("Buy & hold (portfolio)", _number(returns["buy_hold_portfolio_pct"], "%")),
                ("Alpha", _number(returns["alpha_pct"], "%")),
                ("Final equity", _number(returns["final_equity"])),
            ],
        ),
        _section(
            "Risk",
            [
                ("Max drawdown", _number(risk["max_drawdown_pct"], "%")),
                ("Max drawdown, absolute", _number(risk["max_drawdown_abs"])),
                ("Drawdown duration", _number(risk["max_drawdown_duration_days"], " d", 1)),
                ("Recovery time", _number(risk["max_drawdown_recovery_days"], " d", 1)),
                ("Average drawdown", _number(risk["avg_drawdown_pct"], "%")),
                ("Annualised volatility", _number(risk["annualized_volatility_pct"], "%")),
                ("VaR 95%", _number(risk["var_95_pct"], "%")),
                ("CVaR 95%", _number(risk["cvar_95_pct"], "%")),
                ("Ulcer index", _number(risk["ulcer_index"])),
            ],
        ),
        _section(
            "Risk-adjusted",
            [
                ("Sharpe", _number(ratios["sharpe"])),
                ("Sortino", _number(ratios["sortino"])),
                ("Calmar", _number(ratios["calmar"])),
                ("Omega", _number(ratios["omega"])),
            ],
        ),
        _section(
            "Trading",
            [
                ("Trades", _number(trading["trades"])),
                ("Long / short", f"{trading['long_trades']} / {trading['short_trades']}"),
                ("Win rate", _number(trading["win_rate_pct"], "%")),
                ("Profit factor", _number(trading["profit_factor"])),
                ("Expectancy", _number(trading["expectancy_r"], " R")),
                ("Expectancy", _number(trading["expectancy_usdt"])),
                (
                    "Average win / loss",
                    f"{trading['avg_win_r']:.2f}R / {trading['avg_loss_r']:.2f}R",
                ),
                (
                    "Longest win / loss streak",
                    f"{trading['max_win_streak']} / {trading['max_loss_streak']}",
                ),
                ("Average holding time", _number(trading["avg_hours_held"], " h", 1)),
                ("Exposure", _number(trading["exposure_pct"], "%")),
                ("Average MAE / MFE", f"{trading['avg_mae_r']:.2f}R / {trading['avg_mfe_r']:.2f}R"),
            ],
        ),
        _section(
            "Costs",
            [
                ("Fees", _number(costs["fees"])),
                ("Funding", _number(costs["funding"])),
                ("Slippage", _number(costs["slippage_estimate"])),
                ("Total", _number(costs["total"])),
                ("Share of gross profit", _number(100.0 * costs["share_of_gross_profit"], "%")),
            ],
        ),
    ]

    exits = trading.get("exit_reasons") or {}
    if exits:
        blocks.append(
            _section(
                "Exit reasons",
                [(reason, _number(count)) for reason, count in sorted(exits.items())],
            )
        )

    if metrics.by_symbol and len(metrics.by_symbol) > 1:
        blocks.append(
            _section(
                "By symbol",
                [
                    (
                        str(row["symbol"]),
                        f"{row['trades']} trades, {row['pnl_net']:,.2f}",
                    )
                    for row in metrics.by_symbol
                ],
            )
        )

    return "\n\n".join("\n".join(block) for block in blocks)


def format_annual_returns(metrics: PerformanceMetrics) -> str:
    """One line per calendar year."""
    if not metrics.annual_returns:
        return ""
    rows = [(year, _number(value, "%")) for year, value in sorted(metrics.annual_returns.items())]
    return "\n".join(_section("Annual returns", rows))


__all__ = ["format_annual_returns", "format_metrics"]
