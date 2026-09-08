"""Run orchestration: artefacts, reload and byte-level reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingbot.backtest.runner import (
    BacktestRunner,
    build_strategy,
    latest_run_id,
    list_runs,
    load_run,
    run_summary,
)
from tradingbot.config.models import (
    AppConfig,
    BacktestConfig,
    CostsConfig,
    DataConfig,
    ExchangeConfig,
    RiskConfig,
    StrategyConfig,
)
from tradingbot.core.exceptions import BacktestError

SYMBOL = "BTC/USDT"


def trending_frame(bars: int = 400, seed: int = 7) -> pd.DataFrame:
    """A synthetic series with enough structure to trigger real breakouts."""
    rng = np.random.default_rng(seed)
    drift = np.linspace(0.0, 0.6, bars)
    noise = np.cumsum(rng.normal(0.0, 0.01, bars))
    close = 100.0 * np.exp(drift + noise)
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.004, bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.004, bars)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    index = pd.date_range("2024-01-01", periods=bars, freq="4h", tz="UTC", name="ts")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum.reduce([high, open_, close]),
            "low": np.minimum.reduce([low, open_, close]),
            "close": close,
            "volume": rng.uniform(100.0, 200.0, bars),
        },
        index=index,
    )


def make_config(tmp_path: Path, **overrides: object) -> AppConfig:
    """A self-contained config whose cache and runs live inside ``tmp_path``."""
    sections: dict[str, object] = {
        "exchange": ExchangeConfig(symbols=[SYMBOL], timeframe="4h"),
        "data": DataConfig(cache_dir=tmp_path / "cache"),
        "backtest": BacktestConfig(runs_dir=tmp_path / "runs", initial_capital=10_000.0),
        "costs": CostsConfig(taker_fee=0.0004, slippage_model="fixed_bps", slippage_bps=2.0),
    }
    sections.update(overrides)
    return AppConfig(**sections)  # type: ignore[arg-type]


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return make_config(tmp_path)


@pytest.fixture
def data() -> dict[str, pd.DataFrame]:
    return {SYMBOL: trending_frame()}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestArtifacts:
    def test_every_required_file_is_written(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        names = {entry.name for entry in stored.path.iterdir()}
        assert names == {
            "meta.json",
            "config.yaml",
            "trades.parquet",
            "equity.parquet",
            "signals.parquet",
            "metrics.json",
        }

    def test_metrics_are_computed_and_stored(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        saved = json.loads((stored.path / "metrics.json").read_text())
        assert saved["trading"]["trades"] == len(stored.trades)
        assert saved["returns"]["buy_hold_pct"][SYMBOL] != 0.0
        assert "sharpe" in saved["ratios"]

    def test_meta_records_provenance(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data, note="smoke")
        meta = json.loads((stored.path / "meta.json").read_text())
        assert meta["run_id"] == stored.run_id
        assert meta["symbols"] == [SYMBOL]
        assert meta["timeframe"] == "4h"
        assert meta["status"] == "COMPLETED"
        assert meta["note"] == "smoke"
        assert meta["data_hash"]
        assert meta["duration_sec"] > 0

    def test_config_snapshot_includes_resolved_strategy_params(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        assert stored.meta.config["strategy"]["name"] == "donchian_trend"
        assert stored.meta.config["strategy"]["entry_channel"] == 20

    def test_run_directory_is_named_after_the_run_id(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        assert stored.path.name == stored.run_id
        assert stored.path.parent == config.backtest.runs_dir

    def test_saving_can_be_skipped(self, config: AppConfig, data: dict[str, pd.DataFrame]) -> None:
        stored = BacktestRunner(config).run(data=data, save=False)
        assert not stored.path.exists()


class TestReproducibility:
    def test_two_runs_produce_identical_trade_files(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        first = BacktestRunner(config).run(data=data)
        second = BacktestRunner(config).run(data=data, note="second attempt")
        assert digest(first.path / "trades.parquet") == digest(second.path / "trades.parquet")
        assert digest(first.path / "equity.parquet") == digest(second.path / "equity.parquet")

    def test_the_run_id_carries_a_stable_config_fingerprint(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        # The timestamp prefix keeps runs from overwriting each other; the
        # suffix identifies the configuration that produced them.
        first = BacktestRunner(config).run(data=data)
        second = BacktestRunner(config).run(data=data)
        assert first.run_id != second.run_id
        assert first.run_id.split("-")[-1] == second.run_id.split("-")[-1]
        assert first.path.exists() and second.path.exists()

    def test_changing_a_parameter_changes_the_fingerprint(
        self, tmp_path: Path, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        baseline = BacktestRunner(config).run(data=data, save=False)
        tweaked = make_config(tmp_path, strategy=StrategyConfig(params={"entry_channel": 33}))
        altered = BacktestRunner(tweaked).run(data=data, save=False)
        assert altered.run_id.split("-")[-1] != baseline.run_id.split("-")[-1]

    def test_the_data_hash_follows_the_input(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        baseline = BacktestRunner(config).run(data=data, save=False)
        shorter = {SYMBOL: data[SYMBOL].iloc[:-10]}
        altered = BacktestRunner(config).run(data=shorter, save=False)
        assert altered.meta.data_hash != baseline.meta.data_hash


class TestReload:
    def test_a_stored_run_round_trips(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        reloaded = load_run(stored.run_id, config.backtest.runs_dir)
        assert reloaded.meta.run_id == stored.run_id
        pd.testing.assert_frame_equal(reloaded.trades, stored.trades)
        assert reloaded.initial_capital == pytest.approx(10_000.0)

    def test_missing_run_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(BacktestError, match="not found"):
            load_run("nope", tmp_path)

    def test_listing_and_latest(
        self, tmp_path: Path, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        first = BacktestRunner(config).run(data=data)
        tweaked = make_config(tmp_path, strategy=StrategyConfig(params={"entry_channel": 25}))
        second = BacktestRunner(tweaked).run(data=data)

        runs = list_runs(config.backtest.runs_dir)
        assert set(runs) == {first.run_id, second.run_id}
        assert latest_run_id(config.backtest.runs_dir) == runs[0]

    def test_listing_an_absent_directory(self, tmp_path: Path) -> None:
        assert list_runs(tmp_path / "missing") == []
        assert latest_run_id(tmp_path / "missing") is None


class TestDataLoading:
    def test_missing_cache_is_a_clear_error(self, config: AppConfig) -> None:
        with pytest.raises(BacktestError, match="data download"):
            BacktestRunner(config).load_data()


class TestSummary:
    def test_summary_mentions_the_essentials(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        text = run_summary(stored)
        assert stored.run_id in text
        assert SYMBOL in text
        for heading in ("Returns", "Risk", "Trading", "Costs"):
            assert heading in text

    def test_the_short_form_omits_the_metric_table(
        self, config: AppConfig, data: dict[str, pd.DataFrame]
    ) -> None:
        stored = BacktestRunner(config).run(data=data)
        text = run_summary(stored, detailed=False)
        assert stored.run_id in text
        assert "Sharpe" not in text


class TestStrategyConstruction:
    def test_fee_buffer_is_derived_from_the_cost_model(self, config: AppConfig) -> None:
        strategy = build_strategy(config)
        assert strategy.describe()["fee_buffer_pct"] == pytest.approx(0.0008)

    def test_risk_per_trade_is_passed_through(self, tmp_path: Path) -> None:
        config = make_config(tmp_path, risk=RiskConfig(risk_per_trade_pct=0.02))
        assert build_strategy(config).describe()["risk_per_trade_pct"] == pytest.approx(0.02)
