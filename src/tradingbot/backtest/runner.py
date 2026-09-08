"""Run orchestration and on-disk artefacts.

A run is self-describing: alongside the results we store the exact configuration,
the code version and a fingerprint of the input data, so any number in a report
can be traced back to what produced it (tech.md section 8.4).
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from loguru import logger

from tradingbot import __version__
from tradingbot.backtest.engine import BacktestEngine
from tradingbot.config.models import AppConfig
from tradingbot.core.enums import RunStatus
from tradingbot.core.exceptions import BacktestError
from tradingbot.core.models import MarketMeta, RunMeta, make_run_id, utcnow
from tradingbot.data.cache import ParquetCache
from tradingbot.strategy.base import Strategy
from tradingbot.strategy.registry import create_strategy

TRADES_FILE = "trades.parquet"
EQUITY_FILE = "equity.parquet"
SIGNALS_FILE = "signals.parquet"
META_FILE = "meta.json"
CONFIG_FILE = "config.yaml"
METRICS_FILE = "metrics.json"


@dataclass(slots=True)
class StoredRun:
    """A run that has been written to (or read back from) ``runs/``."""

    run_id: str
    path: Path
    meta: RunMeta
    trades: pd.DataFrame
    equity: pd.DataFrame
    signals: pd.DataFrame
    metrics: dict[str, Any]

    @property
    def initial_capital(self) -> float:
        """Starting capital recorded in the run configuration."""
        backtest = self.meta.config.get("backtest", {})
        return float(backtest.get("initial_capital", 0.0))


def build_strategy(config: AppConfig) -> Strategy:
    """Instantiate the configured strategy with risk and cost context."""
    return create_strategy(
        config.strategy.name,
        config.strategy.params,
        risk_per_trade_pct=config.risk.risk_per_trade_pct,
        fee_buffer_pct=2.0 * config.costs.taker_fee,
    )


class BacktestRunner:
    """Loads cached candles, runs the engine and persists the artefacts."""

    def __init__(self, config: AppConfig, markets: Mapping[str, MarketMeta] | None = None) -> None:
        self.config = config
        self.markets = dict(markets or {})
        self.cache = ParquetCache(config.data.cache_dir, config.exchange.name)

    def load_data(self, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """Read cached candles for the requested symbols and window.

        Raises:
            BacktestError: Nothing is cached for any requested symbol.
        """
        timeframe = self.config.exchange.timeframe
        wanted = symbols or list(self.config.exchange.symbols)
        frames: dict[str, pd.DataFrame] = {}
        for symbol in wanted:
            frame = self.cache.load(symbol, timeframe, self.config.data.start, self.config.data.end)
            if frame.empty:
                logger.warning(
                    "{symbol} {timeframe}: nothing cached, run 'tradingbot data download'",
                    symbol=symbol,
                    timeframe=timeframe,
                )
                continue
            frames[symbol] = frame
        if not frames:
            raise BacktestError(
                f"no cached data for {', '.join(wanted)} at {timeframe}; "
                "run 'tradingbot data download' first"
            )
        return frames

    def run(
        self,
        symbols: list[str] | None = None,
        *,
        data: Mapping[str, pd.DataFrame] | None = None,
        save: bool = True,
        note: str = "",
    ) -> StoredRun:
        """Execute a backtest and, unless told otherwise, store its artefacts."""
        frames = dict(data) if data is not None else self.load_data(symbols)
        strategy = build_strategy(self.config)
        engine = BacktestEngine(self.config, strategy, markets=self.markets)

        snapshot = self._config_snapshot(strategy)
        run_id = self._reserve_run_id(snapshot)
        started = time.perf_counter()
        logger.info("run {run_id} started on {count} symbols", run_id=run_id, count=len(frames))

        result = engine.run(frames)
        duration = time.perf_counter() - started

        meta = RunMeta(
            run_id=run_id,
            code_version=__version__,
            git_sha=current_git_sha(),
            config=snapshot,
            data_hash=self._data_fingerprint(frames),
            symbols=result.symbols,
            timeframe=result.timeframe,
            status=RunStatus.COMPLETED,
            duration_sec=duration,
            note=note,
        )
        logger.info(
            "run {run_id} finished in {duration:.2f}s: {trades} trades, final equity {equity:,.2f}",
            run_id=run_id,
            duration=duration,
            trades=len(result.trades),
            equity=result.final_equity,
        )

        path = self.config.backtest.runs_dir / run_id
        stored = StoredRun(
            run_id=run_id,
            path=path,
            meta=meta,
            trades=result.trades,
            equity=result.equity,
            signals=result.signals,
            metrics={},
        )
        if save:
            save_run(stored)
        return stored

    def _reserve_run_id(self, snapshot: dict[str, Any]) -> str:
        """Pick a free identifier in the ``YYYYMMDD-HHMMSS-<hash>`` format of FR-6.4.

        Sweeps and optimisation loops finish several runs inside one second, and
        the format has no sub-second field, so a taken slot moves to the next
        second rather than silently overwriting earlier artefacts.
        """
        moment = utcnow()
        for _ in range(3600):
            run_id = make_run_id(snapshot, moment)
            if not (self.config.backtest.runs_dir / run_id).exists():
                return run_id
            moment += timedelta(seconds=1)
        raise BacktestError("could not allocate a free run id")

    def _config_snapshot(self, strategy: Strategy) -> dict[str, Any]:
        """Effective configuration plus the resolved strategy parameters."""
        snapshot = self.config.to_yaml_dict()
        snapshot["strategy"] = strategy.describe()
        return snapshot

    def _data_fingerprint(self, frames: Mapping[str, pd.DataFrame]) -> str:
        """Hash covering every input series, so results can be tied to their data."""
        parts = []
        for symbol in sorted(frames):
            frame = frames[symbol]
            parts.append(f"{symbol}:{len(frame)}:{frame.index[0]}:{frame.index[-1]}")
        from tradingbot.core.models import config_hash

        return config_hash({"series": parts}, length=16)


def save_run(run: StoredRun) -> Path:
    """Write every artefact of a run into ``runs/<run_id>/``."""
    run.path.mkdir(parents=True, exist_ok=True)
    run.trades.to_parquet(run.path / TRADES_FILE, engine="pyarrow", index=False)
    run.equity.to_parquet(run.path / EQUITY_FILE, engine="pyarrow")
    run.signals.to_parquet(run.path / SIGNALS_FILE, engine="pyarrow", index=False)
    (run.path / META_FILE).write_text(
        json.dumps(run.meta.model_dump(mode="json"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (run.path / CONFIG_FILE).write_text(
        yaml.safe_dump(run.meta.config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if run.metrics:
        write_metrics(run.path, run.metrics)
    logger.info("artefacts written to {path}", path=run.path)
    return run.path


def write_metrics(path: Path, metrics: Mapping[str, Any]) -> Path:
    """Persist the metrics block of a run."""
    target = path / METRICS_FILE
    target.write_text(json.dumps(dict(metrics), indent=2, sort_keys=True, default=str), "utf-8")
    return target


def load_run(run_id: str, runs_dir: Path | str = "runs") -> StoredRun:
    """Read a stored run back from disk without recomputing anything (FR-3.5).

    Raises:
        BacktestError: The run directory or its metadata is missing.
    """
    path = Path(runs_dir) / run_id
    meta_path = path / META_FILE
    if not meta_path.is_file():
        raise BacktestError(f"run {run_id} not found in {runs_dir}")

    meta = RunMeta.model_validate(json.loads(meta_path.read_text(encoding="utf-8")))
    metrics_path = path / METRICS_FILE
    return StoredRun(
        run_id=run_id,
        path=path,
        meta=meta,
        trades=_read_optional(path / TRADES_FILE),
        equity=_read_optional(path / EQUITY_FILE),
        signals=_read_optional(path / SIGNALS_FILE),
        metrics=json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else {},
    )


def list_runs(runs_dir: Path | str = "runs") -> list[str]:
    """Run identifiers present on disk, newest first."""
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return sorted(
        (entry.name for entry in root.iterdir() if (entry / META_FILE).is_file()),
        reverse=True,
    )


def latest_run_id(runs_dir: Path | str = "runs") -> str | None:
    """Identifier of the most recent run, or ``None`` when there are none."""
    runs = list_runs(runs_dir)
    return runs[0] if runs else None


def current_git_sha() -> str | None:
    """Short commit hash of the working tree, when available."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = completed.stdout.strip()
    return sha or None


def _read_optional(path: Path) -> pd.DataFrame:
    """Read a Parquet artefact, tolerating its absence."""
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def run_summary(run: StoredRun) -> str:
    """One-paragraph console summary of a finished run."""
    trades = run.trades
    lines = [
        f"run {run.run_id}",
        f"  symbols   {', '.join(run.meta.symbols)} @ {run.meta.timeframe}",
        f"  trades    {len(trades)}",
    ]
    if not trades.empty:
        wins = int((trades["pnl_net"] > 0).sum())
        lines.append(f"  win rate  {wins / len(trades):.1%}")
        lines.append(f"  net pnl   {trades['pnl_net'].sum():,.2f}")
    if not run.equity.empty:
        final = float(run.equity["equity"].iloc[-1])
        lines.append(f"  equity    {final:,.2f}")
    lines.append(f"  artefacts {run.path}")
    return "\n".join(lines)


def utc_stamp() -> datetime:
    """Current UTC time; kept here so tests can monkeypatch a single symbol."""
    return utcnow()
