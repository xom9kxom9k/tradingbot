"""Parameter search: grid, Optuna, scoring and plateau selection (section 9.3).

The peak of a grid is almost never the combination to trade. Nearby cells that
score nearly as well form a plateau; the centre of that plateau is a stabler
choice than the single loudest spike, which is usually noise.
"""

from __future__ import annotations

import itertools
import json
import math
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from tradingbot.analytics.metrics import PerformanceMetrics, compute_metrics
from tradingbot.backtest.engine import BacktestEngine, BacktestResult
from tradingbot.backtest.runner import BacktestRunner, StoredRun, build_strategy
from tradingbot.config.models import AppConfig, ObjectiveName
from tradingbot.core.exceptions import BacktestError, ConfigError, TradingBotError

REJECTED_SCORE = float("-inf")
TRIALS_FILE = "trials.parquet"
SENSITIVITY_FILE = "sensitivity.parquet"
OPTIMIZE_FILE = "optimize.json"


@dataclass(slots=True)
class TrialRecord:
    """One evaluated parameter combination."""

    params: dict[str, Any]
    score: float
    accepted: bool
    reason: str
    trades: int
    total_return_pct: float
    sharpe: float
    calmar: float
    profit_factor: float
    max_drawdown_pct: float

    def to_row(self) -> dict[str, Any]:
        """Flat row for a trials table, with params expanded."""
        row = asdict(self)
        params = row.pop("params")
        return {**params, **row}


@dataclass(slots=True)
class SearchResult:
    """Outcome of a grid or Optuna study."""

    trials: list[TrialRecord]
    best: TrialRecord | None
    plateau: TrialRecord | None
    objective: str
    heatmap: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def chosen(self) -> TrialRecord | None:
        """Parameters we would actually take: plateau centre, else the peak."""
        return self.plateau or self.best


def with_params(config: AppConfig, overrides: Mapping[str, Any]) -> AppConfig:
    """Copy of ``config`` whose strategy params include ``overrides``."""
    merged = {**config.strategy.params, **dict(overrides)}
    return config.model_copy(
        update={"strategy": config.strategy.model_copy(update={"params": merged})}
    )


def expand_grid(space: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a parameter space, in stable key order."""
    if not space:
        return [{}]
    keys = list(space)
    values = [list(space[key]) for key in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def score_metrics(
    metrics: PerformanceMetrics,
    *,
    objective: ObjectiveName,
    min_trades: int,
) -> tuple[float, bool, str]:
    """Turn a metrics block into a scalar, rejecting thin samples.

    Returns:
        ``(score, accepted, reason)``. Rejected trials score ``-inf`` so they
        never win a search, while still showing up in the trials table.
    """
    trades = int(metrics.trading.get("trades", 0))
    if trades < min_trades:
        return REJECTED_SCORE, False, f"only {trades} trades (min {min_trades})"

    sharpe = _finite(metrics.ratios.get("sharpe"))
    calmar = _finite(metrics.ratios.get("calmar"))
    factor = _finite(metrics.trading.get("profit_factor"))
    if objective == "sharpe":
        value = sharpe
    elif objective == "calmar":
        value = calmar
    elif objective == "profit_factor":
        value = factor
    else:
        value = calmar * math.log1p(trades)
    if not math.isfinite(value):
        return REJECTED_SCORE, False, "objective is not finite"
    return value, True, ""


def run_trial(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    params: Mapping[str, Any],
    *,
    objective: ObjectiveName,
    min_trades: int,
) -> tuple[TrialRecord, BacktestResult | None]:
    """Run one backtest and score it, returning the engine result when it ran.

    Invalid parameter combinations (ones the strategy schema rejects) are
    recorded as rejected trials rather than crashing the search.
    """
    rejected = TrialRecord(
        params=dict(params),
        score=REJECTED_SCORE,
        accepted=False,
        reason="",
        trades=0,
        total_return_pct=0.0,
        sharpe=0.0,
        calmar=0.0,
        profit_factor=0.0,
        max_drawdown_pct=0.0,
    )
    try:
        cfg = with_params(config, params)
        strategy = build_strategy(cfg)
        result = BacktestEngine(cfg, strategy).run(data)
    except (ConfigError, BacktestError, ValueError) as exc:
        logger.warning("trial {params} skipped: {exc}", params=dict(params), exc=exc)
        rejected.reason = str(exc)
        return rejected, None

    metrics = _metrics_for(cfg, result, data)
    score, accepted, reason = score_metrics(metrics, objective=objective, min_trades=min_trades)
    record = TrialRecord(
        params=dict(params),
        score=score,
        accepted=accepted,
        reason=reason,
        trades=int(metrics.trading.get("trades", 0)),
        total_return_pct=float(metrics.returns.get("total_return_pct", 0.0)),
        sharpe=_finite(metrics.ratios.get("sharpe")),
        calmar=_finite(metrics.ratios.get("calmar")),
        profit_factor=_finite(metrics.trading.get("profit_factor")),
        max_drawdown_pct=float(metrics.risk.get("max_drawdown_pct", 0.0)),
    )
    return record, result


def evaluate(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    params: Mapping[str, Any],
    *,
    objective: ObjectiveName,
    min_trades: int,
) -> TrialRecord:
    """Run one backtest and score it."""
    record, _ = run_trial(config, data, params, objective=objective, min_trades=min_trades)
    return record


def run_grid(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    *,
    space: Mapping[str, Sequence[Any]] | None = None,
    objective: ObjectiveName | None = None,
    min_trades: int | None = None,
    n_jobs: int | None = None,
) -> SearchResult:
    """Exhaustive search over ``space`` (or the configured grid)."""
    settings = config.optimize
    combos = expand_grid(space or settings.grid)
    target = objective or settings.objective
    floor = min_trades if min_trades is not None else settings.min_trades
    workers = n_jobs if n_jobs is not None else settings.n_jobs
    logger.info(
        "grid search: {n} combinations, objective={obj}, jobs={jobs}",
        n=len(combos),
        obj=target,
        jobs=workers,
    )
    trials = _map_trials(config, data, combos, target, floor, workers)
    return _finalise(trials, target, settings.heatmap_x, settings.heatmap_y)


def run_optuna(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    *,
    n_trials: int | None = None,
    space: Mapping[str, Sequence[Any]] | None = None,
    objective: ObjectiveName | None = None,
    min_trades: int | None = None,
    seed: int | None = None,
    n_jobs: int | None = None,
) -> SearchResult:
    """TPE search. Each trial is a full backtest; there are no intermediate steps to prune."""
    import threading

    import optuna
    from optuna.samplers import TPESampler

    settings = config.optimize
    domain = dict(space or settings.grid)
    target = objective or settings.objective
    floor = min_trades if min_trades is not None else settings.min_trades
    trials_n = n_trials if n_trials is not None else settings.optuna_trials
    sampler_seed = config.backtest.seed if seed is None else seed
    workers = n_jobs if n_jobs is not None else settings.n_jobs
    records: list[TrialRecord] = []
    lock = threading.Lock()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=sampler_seed),
        pruner=optuna.pruners.MedianPruner(),
    )

    def objective_fn(trial: optuna.Trial) -> float:
        params = {name: _suggest(trial, name, values) for name, values in domain.items()}
        record = evaluate(config, data, params, objective=target, min_trades=floor)
        with lock:
            records.append(record)
        if not record.accepted:
            raise optuna.TrialPruned(record.reason)
        return record.score

    logger.info("optuna search: {n} trials, objective={obj}", n=trials_n, obj=target)
    study.optimize(
        objective_fn,
        n_trials=trials_n,
        n_jobs=workers,
        catch=(TradingBotError, ValueError),
    )
    return _finalise(records, target, settings.heatmap_x, settings.heatmap_y)


def pick_plateau(trials: Sequence[TrialRecord], *, band: float = 0.10) -> TrialRecord | None:
    """Centre of the cluster of trials within ``band`` of the best score.

    ``band`` is a fraction of the peak. Among those near-best cells we pick the
    combination whose parameters sit closest to the component-wise median, which
    is the geometric centre of the plateau rather than its noisiest corner.
    """
    accepted = [trial for trial in trials if trial.accepted]
    if not accepted:
        return None
    peak = max(trial.score for trial in accepted)
    if peak > 0:
        floor = peak * (1.0 - band)
    elif peak < 0:
        floor = peak * (1.0 + band)
    else:
        floor = peak
    cluster = [trial for trial in accepted if trial.score >= floor]
    if len(cluster) == 1:
        return cluster[0]
    keys = sorted({key for trial in cluster for key in trial.params})
    numeric = [key for key in keys if all(_is_number(trial.params.get(key)) for trial in cluster)]
    if not numeric:
        return max(cluster, key=lambda trial: trial.score)
    medians = {key: _median([float(trial.params[key]) for trial in cluster]) for key in numeric}
    spans = {
        key: (
            max(float(trial.params[key]) for trial in cluster)
            - min(float(trial.params[key]) for trial in cluster)
        )
        or 1.0
        for key in numeric
    }

    def distance(trial: TrialRecord) -> float:
        return sum(((float(trial.params[key]) - medians[key]) / spans[key]) ** 2 for key in numeric)

    return min(cluster, key=distance)


def heatmap_frame(
    trials: Sequence[TrialRecord],
    x_name: str,
    y_name: str,
) -> pd.DataFrame:
    """Mean objective on the ``(x, y)`` plane, for the parameter heatmap."""
    rows = []
    for trial in trials:
        if x_name not in trial.params or y_name not in trial.params:
            continue
        if not trial.accepted:
            continue
        rows.append({"x": trial.params[x_name], "y": trial.params[y_name], "value": trial.score})
    if not rows:
        return pd.DataFrame(columns=["x", "y", "value"])
    frame = pd.DataFrame(rows)
    return frame.groupby(["x", "y"], as_index=False).agg(value=("value", "mean"))


def trials_frame(trials: Sequence[TrialRecord]) -> pd.DataFrame:
    """All trials as a table, rejected ones included."""
    if not trials:
        return pd.DataFrame()
    return pd.DataFrame([trial.to_row() for trial in trials])


def save_optimisation(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    search: SearchResult,
    *,
    method: str,
    note: str = "",
) -> StoredRun:
    """Re-run the plateau parameters as a full backtest and store search artefacts."""
    chosen = search.chosen
    if chosen is None:
        raise BacktestError("no accepted trials; relax min_trades or widen the grid")
    cfg = with_params(config, chosen.params)
    stored = BacktestRunner(cfg).run(data=data, note=note or f"optimize {method}")
    write_search_artefacts(stored.path, search, method=method)
    return stored


def write_search_artefacts(path: Path, search: SearchResult, *, method: str) -> None:
    """Write trials, the sensitivity heatmap and a JSON summary next to a run."""
    path.mkdir(parents=True, exist_ok=True)
    table = trials_frame(search.trials)
    if not table.empty:
        table.to_parquet(path / TRIALS_FILE, engine="pyarrow", index=False)
    if not search.heatmap.empty:
        search.heatmap.to_parquet(path / SENSITIVITY_FILE, engine="pyarrow", index=False)
    payload = {
        "method": method,
        "objective": search.objective,
        "n_trials": len(search.trials),
        "n_accepted": sum(1 for trial in search.trials if trial.accepted),
        "peak": _trial_payload(search.best),
        "plateau": _trial_payload(search.plateau),
        "chosen": _trial_payload(search.chosen),
    }
    (path / OPTIMIZE_FILE).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _map_trials(
    config: AppConfig,
    data: Mapping[str, pd.DataFrame],
    combos: Sequence[Mapping[str, Any]],
    objective: ObjectiveName,
    min_trades: int,
    n_jobs: int,
) -> list[TrialRecord]:
    if n_jobs <= 1 or len(combos) <= 1:
        return [
            evaluate(config, data, combo, objective=objective, min_trades=min_trades)
            for combo in combos
        ]
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = [
            pool.submit(evaluate, config, data, combo, objective=objective, min_trades=min_trades)
            for combo in combos
        ]
        return [future.result() for future in futures]


def _finalise(
    trials: list[TrialRecord],
    objective: str,
    heatmap_x: str,
    heatmap_y: str,
) -> SearchResult:
    accepted = [trial for trial in trials if trial.accepted]
    best = max(accepted, key=lambda trial: trial.score) if accepted else None
    plateau = pick_plateau(trials)
    heat = heatmap_frame(trials, heatmap_x, heatmap_y)
    if plateau:
        logger.info(
            "chosen plateau {params} (score {score:.3f}); peak was {peak}",
            params=plateau.params,
            score=plateau.score,
            peak=best.params if best else None,
        )
    return SearchResult(
        trials=trials, best=best, plateau=plateau, objective=objective, heatmap=heat
    )


def _metrics_for(
    config: AppConfig,
    result: BacktestResult,
    data: Mapping[str, pd.DataFrame],
) -> PerformanceMetrics:
    return compute_metrics(
        result.equity,
        result.trades,
        timeframe=result.timeframe,
        risk_free_rate=config.backtest.risk_free_rate,
        benchmark={symbol: frame["close"] for symbol, frame in data.items() if not frame.empty},
        costs=config.costs,
        slippage=result.slippage_cost,
    )


def _suggest(trial: Any, name: str, values: Sequence[Any]) -> Any:
    """Map a discrete grid axis onto an Optuna suggestion."""
    if not values:
        raise ValueError(f"empty domain for {name}")
    unique = list(dict.fromkeys(values))
    if all(isinstance(item, bool) for item in unique):
        return trial.suggest_categorical(name, unique)
    if all(isinstance(item, int) and not isinstance(item, bool) for item in unique):
        lo, hi = min(unique), max(unique)
        if lo == hi:
            return lo
        step = min(abs(a - b) for a, b in itertools.pairwise(sorted(set(unique))))
        return trial.suggest_int(name, lo, hi, step=max(step, 1))
    if all(isinstance(item, int | float) and not isinstance(item, bool) for item in unique):
        lo, hi = float(min(unique)), float(max(unique))
        if lo == hi:
            return lo
        return trial.suggest_float(name, lo, hi)
    return trial.suggest_categorical(name, unique)


def _trial_payload(trial: TrialRecord | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    return {
        "params": trial.params,
        "score": trial.score,
        "trades": trial.trades,
        "total_return_pct": trial.total_return_pct,
        "sharpe": trial.sharpe,
        "calmar": trial.calmar,
        "profit_factor": trial.profit_factor,
        "max_drawdown_pct": trial.max_drawdown_pct,
        "accepted": trial.accepted,
    }


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return number


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])
