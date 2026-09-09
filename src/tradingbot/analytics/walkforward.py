"""Walk-forward analysis: rolling/anchored windows, OOS stitch and WFE (section 9.2).

Each window optimises on the in-sample stretch, freezes those parameters and
applies them to the following out-of-sample stretch. The OOS equity segments
are then rebased onto a running capital so the stitched curve is the one a
trader would actually have lived through — not the in-sample peak.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from tradingbot import __version__
from tradingbot.analytics.metrics import compute_metrics, total_return
from tradingbot.analytics.optimizer import run_grid, run_trial
from tradingbot.backtest.engine import BacktestResult
from tradingbot.backtest.runner import (
    BacktestRunner,
    StoredRun,
    build_strategy,
    current_git_sha,
    save_run,
)
from tradingbot.config.models import AppConfig, ObjectiveName, WalkForwardMode
from tradingbot.core.enums import RunStatus
from tradingbot.core.exceptions import BacktestError
from tradingbot.core.models import RunMeta

WINDOWS_FILE = "walkforward.parquet"
OOS_EQUITY_FILE = "walkforward_equity.parquet"
SUMMARY_FILE = "walkforward.json"
HEALTHY_WFE = 0.5
HEALTHY_POSITIVE_OOS = 0.60


@dataclass(slots=True)
class WalkWindow:
    """One in-sample / out-of-sample pair."""

    index: int
    label: str
    is_start: pd.Timestamp
    is_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass(slots=True)
class WindowResult:
    """Scores and frozen parameters of one walk-forward window."""

    window: WalkWindow
    params: dict[str, Any]
    is_return: float
    oos_return: float
    wfe: float | None
    is_trades: int
    oos_trades: int
    is_score: float
    reason: str = ""
    oos_equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_trades_frame: pd.DataFrame = field(default_factory=pd.DataFrame)
    oos_signals: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_row(self) -> dict[str, Any]:
        """Flat row for the windows table the dashboard already knows how to plot."""
        return {
            "window": self.window.label,
            "is_start": self.window.is_start.isoformat(),
            "is_end": self.window.is_end.isoformat(),
            "oos_start": self.window.oos_start.isoformat(),
            "oos_end": self.window.oos_end.isoformat(),
            "is_return": self.is_return,
            "oos_return": self.oos_return,
            "wfe": self.wfe,
            "is_trades": self.is_trades,
            "oos_trades": self.oos_trades,
            "is_score": self.is_score,
            "params": json.dumps(self.params, default=str),
            "reason": self.reason,
            **{f"param_{key}": value for key, value in self.params.items()},
        }


@dataclass(slots=True)
class WalkForwardStudy:
    """Full walk-forward: per-window rows, stitched OOS equity and a health verdict."""

    windows: list[WindowResult]
    equity: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    wfe: float | None
    positive_oos_share: float
    healthy: bool
    mode: str
    is_months: int
    oos_months: int
    step_months: int

    def windows_frame(self) -> pd.DataFrame:
        """Table of windows for ``walkforward.parquet`` and the report."""
        if not self.windows:
            return pd.DataFrame()
        return pd.DataFrame([row.to_row() for row in self.windows])


def data_span(data: Mapping[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Earliest and latest timestamp across every non-empty series."""
    starts = [frame.index.min() for frame in data.values() if not frame.empty]
    ends = [frame.index.max() for frame in data.values() if not frame.empty]
    if not starts or not ends:
        raise BacktestError("walk-forward needs at least one non-empty price series")
    return pd.Timestamp(min(starts)), pd.Timestamp(max(ends))


def iter_windows(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    mode: WalkForwardMode = "rolling",
    is_months: int = 18,
    oos_months: int = 6,
    step_months: int = 6,
) -> list[WalkWindow]:
    """Build consecutive IS/OOS windows that fit inside ``[start, end]``."""
    windows: list[WalkWindow] = []
    is_start = start
    is_end = start + pd.DateOffset(months=is_months)
    index = 0
    while True:
        oos_start = is_end
        oos_end = oos_start + pd.DateOffset(months=oos_months)
        if oos_end > end:
            break
        windows.append(
            WalkWindow(
                index=index,
                label=f"W{index + 1}",
                is_start=pd.Timestamp(is_start),
                is_end=pd.Timestamp(oos_start),
                oos_start=pd.Timestamp(oos_start),
                oos_end=pd.Timestamp(oos_end),
            )
        )
        index += 1
        if mode == "anchored":
            is_end = is_end + pd.DateOffset(months=step_months)
        else:
            is_start = is_start + pd.DateOffset(months=step_months)
            is_end = is_start + pd.DateOffset(months=is_months)
    return windows


def slice_range(
    data: Mapping[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    warmup_bars: int = 0,
) -> dict[str, pd.DataFrame]:
    """Slice each series to ``[start, end)``, keeping ``warmup_bars`` of lookback."""
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in data.items():
        if frame.empty:
            continue
        start_i = int(frame.index.searchsorted(start))
        end_i = int(frame.index.searchsorted(end))
        begin = max(0, start_i - warmup_bars)
        sliced = frame.iloc[begin:end_i]
        if not sliced.empty:
            out[symbol] = sliced
    return out


def window_return(equity: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Total return of the equity curve restricted to ``[start, end)``."""
    if equity.empty or "equity" not in equity:
        return 0.0
    curve = equity["equity"]
    sliced = curve[(curve.index >= start) & (curve.index < end)]
    if len(sliced) < 2:
        sliced = curve[(curve.index >= start) & (curve.index <= end)]
    return total_return(sliced)


def stitch_equity(segments: list[pd.DataFrame], initial_capital: float) -> pd.DataFrame:
    """Rebase each OOS segment onto the capital left by the previous one."""
    pieces: list[pd.DataFrame] = []
    capital = initial_capital
    for frame in segments:
        if frame.empty or "equity" not in frame:
            continue
        curve = frame["equity"].astype(float)
        origin = float(curve.iloc[0])
        if origin <= 0:
            continue
        scale = capital / origin
        piece = frame.copy()
        piece["equity"] = curve * scale
        if "balance" in piece:
            piece["balance"] = piece["balance"].astype(float) * scale
        peak = piece["equity"].cummax()
        denom = peak.mask(peak <= 0)
        piece["drawdown_pct"] = (peak - piece["equity"]) / denom
        pieces.append(piece)
        capital = float(piece["equity"].iloc[-1])
    if not pieces:
        return pd.DataFrame(
            columns=["balance", "equity", "open_positions", "drawdown_pct"],
            index=pd.DatetimeIndex([], tz="UTC", name="ts"),
        )
    return pd.concat(pieces)


def wfe(oos_return: float, is_return: float) -> float | None:
    """Walk-forward efficiency: OOS return over IS return.

    Undefined when the in-sample return is essentially zero, so a dead IS
    window cannot inflate the headline number through division by noise.
    """
    if abs(is_return) < 1e-12:
        return None
    return oos_return / is_return


def run_walkforward(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    *,
    mode: WalkForwardMode | None = None,
    is_months: int | None = None,
    oos_months: int | None = None,
    step_months: int | None = None,
    objective: ObjectiveName | None = None,
    min_trades: int | None = None,
) -> WalkForwardStudy:
    """Optimise each IS window and freeze the plateau parameters onto the OOS."""
    settings = config.walkforward
    wf_mode = mode or settings.mode
    is_len = is_months if is_months is not None else settings.is_months
    oos_len = oos_months if oos_months is not None else settings.oos_months
    step = step_months if step_months is not None else settings.step_months
    target = objective or config.optimize.objective
    floor = min_trades if min_trades is not None else config.optimize.min_trades
    warmup = build_strategy(config).warmup_period()

    start, end = data_span(data)
    schedule = iter_windows(
        start,
        end,
        mode=wf_mode,
        is_months=is_len,
        oos_months=oos_len,
        step_months=step,
    )
    if not schedule:
        raise BacktestError(
            f"not enough history for a walk-forward window (need IS={is_len}m + OOS={oos_len}m)"
        )
    logger.info(
        "walk-forward {mode}: {n} windows, IS={is_len}m OOS={oos_len}m step={step}m",
        mode=wf_mode,
        n=len(schedule),
        is_len=is_len,
        oos_len=oos_len,
        step=step,
    )

    rows: list[WindowResult] = []
    for window in schedule:
        rows.append(
            _run_window(
                config,
                data,
                window,
                warmup=warmup,
                objective=target,
                min_trades=floor,
            )
        )

    oos_frames = [row.oos_equity for row in rows if not row.oos_equity.empty]
    equity = stitch_equity(oos_frames, config.backtest.initial_capital)
    trades = _concat_frames([row.oos_trades_frame for row in rows])
    signals = _concat_frames([row.oos_signals for row in rows])
    efficiencies = [row.wfe for row in rows if row.wfe is not None]
    mean_wfe = float(sum(efficiencies) / len(efficiencies)) if efficiencies else None
    positive = sum(1 for row in rows if row.oos_return > 0)
    share = positive / len(rows) if rows else 0.0
    healthy = mean_wfe is not None and mean_wfe >= HEALTHY_WFE and share >= HEALTHY_POSITIVE_OOS
    logger.info(
        "walk-forward WFE={wfe} positive OOS={share:.0%} healthy={healthy}",
        wfe=None if mean_wfe is None else round(mean_wfe, 3),
        share=share,
        healthy=healthy,
    )
    return WalkForwardStudy(
        windows=rows,
        equity=equity,
        trades=trades,
        signals=signals,
        wfe=mean_wfe,
        positive_oos_share=share,
        healthy=healthy,
        mode=wf_mode,
        is_months=is_len,
        oos_months=oos_len,
        step_months=step,
    )


def save_walkforward(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    study: WalkForwardStudy,
    *,
    note: str = "",
) -> StoredRun:
    """Persist the stitched OOS run plus the per-window table."""
    stored = _assemble_run(config, data, study, note=note)
    save_run(stored)
    write_walkforward_artefacts(stored.path, study)
    return stored


def write_walkforward_artefacts(path: Path, study: WalkForwardStudy) -> None:
    """Write the windows table, stitched OOS equity and a health summary."""
    path.mkdir(parents=True, exist_ok=True)
    table = study.windows_frame()
    if not table.empty:
        table.to_parquet(path / WINDOWS_FILE, engine="pyarrow", index=False)
    if not study.equity.empty and "equity" in study.equity:
        study.equity[["equity"]].to_parquet(path / OOS_EQUITY_FILE, engine="pyarrow")
    payload = {
        "mode": study.mode,
        "is_months": study.is_months,
        "oos_months": study.oos_months,
        "step_months": study.step_months,
        "n_windows": len(study.windows),
        "wfe": study.wfe,
        "positive_oos_share": study.positive_oos_share,
        "healthy": study.healthy,
        "healthy_wfe": HEALTHY_WFE,
        "healthy_positive_oos": HEALTHY_POSITIVE_OOS,
        "final_equity": (
            float(study.equity["equity"].iloc[-1])
            if not study.equity.empty and "equity" in study.equity
            else None
        ),
    }
    (path / SUMMARY_FILE).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _run_window(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    window: WalkWindow,
    *,
    warmup: int,
    objective: ObjectiveName,
    min_trades: int,
) -> WindowResult:
    is_data = slice_range(data, window.is_start, window.is_end, warmup_bars=warmup)
    oos_data = slice_range(data, window.oos_start, window.oos_end, warmup_bars=warmup)
    empty = WindowResult(
        window=window,
        params={},
        is_return=0.0,
        oos_return=0.0,
        wfe=None,
        is_trades=0,
        oos_trades=0,
        is_score=float("-inf"),
        reason="",
    )
    if not is_data:
        empty.reason = "empty in-sample data"
        return empty
    if not oos_data:
        empty.reason = "empty out-of-sample data"
        return empty

    search = run_grid(config, is_data, objective=objective, min_trades=min_trades)
    chosen = search.chosen
    if chosen is None:
        empty.reason = "no accepted in-sample trial"
        return empty

    _, is_result = run_trial(config, is_data, chosen.params, objective=objective, min_trades=1)
    oos_record, oos_result = run_trial(
        config, oos_data, chosen.params, objective=objective, min_trades=1
    )
    is_return = (
        window_return(is_result.equity, window.is_start, window.is_end) if is_result else 0.0
    )
    oos_return = (
        window_return(oos_result.equity, window.oos_start, window.oos_end) if oos_result else 0.0
    )
    oos_equity = (
        _clip_equity(oos_result, window.oos_start, window.oos_end) if oos_result else pd.DataFrame()
    )
    oos_trades = (
        _clip_trades(oos_result.trades, window.oos_start, window.oos_end)
        if oos_result is not None
        else pd.DataFrame()
    )
    oos_signals = (
        _clip_signals(oos_result.signals, window.oos_start, window.oos_end)
        if oos_result is not None
        else pd.DataFrame()
    )
    return WindowResult(
        window=window,
        params=dict(chosen.params),
        is_return=is_return,
        oos_return=oos_return,
        wfe=wfe(oos_return, is_return),
        is_trades=chosen.trades,
        oos_trades=int(oos_record.trades) if oos_record else 0,
        is_score=chosen.score,
        reason="",
        oos_equity=oos_equity,
        oos_trades_frame=oos_trades,
        oos_signals=oos_signals,
    )


def _clip_equity(result: BacktestResult, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = result.equity
    if frame.empty:
        return frame
    clipped = frame[(frame.index >= start) & (frame.index < end)]
    return clipped if not clipped.empty else frame[(frame.index >= start) & (frame.index <= end)]


def _clip_trades(trades: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if trades.empty or "entry_ts" not in trades:
        return trades
    stamp = pd.to_datetime(trades["entry_ts"], utc=True)
    return trades.loc[(stamp >= start) & (stamp < end)].copy()


def _clip_signals(signals: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if signals.empty or "bar_ts" not in signals:
        return signals
    stamp = pd.to_datetime(signals["bar_ts"], utc=True)
    return signals.loc[(stamp >= start) & (stamp < end)].copy()


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    present = [frame for frame in frames if not frame.empty]
    if not present:
        return pd.DataFrame()
    reset = any(column in present[0].columns for column in ("entry_ts", "bar_ts"))
    return pd.concat(present, ignore_index=reset)


def _assemble_run(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    study: WalkForwardStudy,
    *,
    note: str,
) -> StoredRun:
    runner = BacktestRunner(config)
    strategy = build_strategy(config)
    snapshot = runner._config_snapshot(strategy)
    run_id = runner._reserve_run_id(snapshot)
    equity = study.equity
    symbols = [symbol for symbol, frame in data.items() if not frame.empty]
    metrics = compute_metrics(
        equity if not equity.empty else pd.DataFrame({"equity": []}),
        study.trades,
        timeframe=config.exchange.timeframe,
        risk_free_rate=config.backtest.risk_free_rate,
        benchmark={symbol: frame["close"] for symbol, frame in data.items() if not frame.empty},
        costs=config.costs,
    )
    metrics_dict = metrics.to_dict()
    metrics_dict["walkforward"] = {
        "wfe": study.wfe,
        "positive_oos_share": study.positive_oos_share,
        "healthy": study.healthy,
        "n_windows": len(study.windows),
        "mode": study.mode,
    }
    meta = RunMeta(
        run_id=run_id,
        code_version=__version__,
        git_sha=current_git_sha(),
        config=snapshot,
        data_hash=runner._data_fingerprint(data),
        symbols=symbols,
        timeframe=config.exchange.timeframe,
        status=RunStatus.COMPLETED,
        duration_sec=0.0,
        note=note or f"walk-forward {study.mode} IS={study.is_months} OOS={study.oos_months}",
    )
    return StoredRun(
        run_id=run_id,
        path=config.backtest.runs_dir / run_id,
        meta=meta,
        trades=study.trades,
        equity=equity,
        signals=study.signals,
        metrics=metrics_dict,
    )
