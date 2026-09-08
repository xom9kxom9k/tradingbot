"""Configuration loading, merging and validation."""

from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path

import pytest
import yaml

from tradingbot.config.loader import (
    deep_merge,
    env_overrides,
    load_config,
    load_secrets,
    load_yaml,
)
from tradingbot.config.models import AppConfig, ExchangeConfig, RiskConfig
from tradingbot.core.exceptions import ConfigError

REPO_CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """A minimal but valid pair of config files."""
    base = {
        "exchange": {"name": "binanceusdm", "symbols": ["BTC/USDT"], "timeframe": "4h"},
        "data": {"start": "2020-01-01", "cache_dir": "data/ohlcv"},
        "backtest": {"initial_capital": 5000},
    }
    strategy = {"strategy": {"name": "donchian_trend", "params": {"entry_channel": 20}}}
    (tmp_path / "base.yaml").write_text(yaml.safe_dump(base), encoding="utf-8")
    (tmp_path / "strategy_donchian_trend.yaml").write_text(
        yaml.safe_dump(strategy), encoding="utf-8"
    )
    return tmp_path


class TestDeepMerge:
    def test_nested_dicts_are_merged_not_replaced(self) -> None:
        merged = deep_merge({"a": {"x": 1, "y": 2}}, {"a": {"y": 3, "z": 4}})
        assert merged == {"a": {"x": 1, "y": 3, "z": 4}}

    def test_lists_are_replaced_wholesale(self) -> None:
        merged = deep_merge({"symbols": ["A", "B"]}, {"symbols": ["C"]})
        assert merged == {"symbols": ["C"]}

    def test_inputs_are_not_mutated(self) -> None:
        base = {"a": {"x": 1}}
        deep_merge(base, {"a": {"x": 2}})
        assert base == {"a": {"x": 1}}


class TestEnvOverrides:
    def test_nested_key_and_type_coercion(self) -> None:
        overrides = env_overrides(
            {
                "TRADINGBOT_EXCHANGE__TIMEFRAME": "1h",
                "TRADINGBOT_RISK__MAX_CONCURRENT_POSITIONS": "5",
                "TRADINGBOT_COSTS__APPLY_FUNDING": "false",
                "PATH": "/usr/bin",
            }
        )
        assert overrides == {
            "exchange": {"timeframe": "1h"},
            "risk": {"max_concurrent_positions": 5},
            "costs": {"apply_funding": False},
        }

    def test_unprefixed_variables_are_ignored(self) -> None:
        assert env_overrides({"TIMEFRAME": "1h"}) == {}


class TestLoadYaml:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(tmp_path / "nope.yaml")

    def test_malformed_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("a: [1, 2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid YAML"):
            load_yaml(path)

    def test_non_mapping_root(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping at the top level"):
            load_yaml(path)

    def test_empty_file_is_an_empty_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_yaml(path) == {}


class TestLoadConfig:
    def test_repository_configs_are_valid(self) -> None:
        config = load_config(config_dir=REPO_CONFIGS, use_env=False)
        assert config.exchange.timeframe == "4h"
        assert len(config.exchange.symbols) == 5
        assert config.strategy.name == "donchian_trend"
        assert config.strategy.params["entry_channel"] == 20
        assert config.notify.daily_report_utc == time(9, 0)

    def test_defaults_fill_missing_sections(self, config_dir: Path) -> None:
        config = load_config(config_dir=config_dir, use_env=False)
        assert config.risk.risk_per_trade_pct == 0.01
        assert config.costs.slippage_model == "atr"
        assert config.backtest.initial_capital == 5000

    def test_env_overrides_yaml(self, config_dir: Path) -> None:
        config = load_config(
            config_dir=config_dir,
            environ={"TRADINGBOT_EXCHANGE__TIMEFRAME": "1d"},
        )
        assert config.exchange.timeframe == "1d"

    def test_cli_overrides_beat_env(self, config_dir: Path) -> None:
        config = load_config(
            config_dir=config_dir,
            environ={"TRADINGBOT_EXCHANGE__TIMEFRAME": "1d"},
            overrides={"exchange": {"timeframe": "15m"}},
        )
        assert config.exchange.timeframe == "15m"

    def test_later_files_win(self, tmp_path: Path) -> None:
        first = tmp_path / "a.yaml"
        second = tmp_path / "b.yaml"
        first.write_text(
            yaml.safe_dump({"exchange": {"symbols": ["BTC/USDT"], "timeframe": "4h"}}),
            encoding="utf-8",
        )
        second.write_text(yaml.safe_dump({"exchange": {"timeframe": "1h"}}), encoding="utf-8")
        config = load_config([first, second], use_env=False)
        assert config.exchange.timeframe == "1h"
        assert config.exchange.symbols == ["BTC/USDT"]

    def test_invalid_value_reports_field_path(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError) as excinfo:
            load_config(
                config_dir=config_dir,
                use_env=False,
                overrides={"risk": {"risk_per_trade_pct": 1.5}},
            )
        assert "risk.risk_per_trade_pct" in str(excinfo.value)

    def test_unknown_key_is_rejected(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError, match="typo_here"):
            load_config(config_dir=config_dir, use_env=False, overrides={"risk": {"typo_here": 1}})

    def test_missing_default_files(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="missing"):
            load_config(config_dir=tmp_path, use_env=False)

    def test_snapshot_roundtrips_through_yaml(self, config_dir: Path) -> None:
        config = load_config(config_dir=config_dir, use_env=False)
        dumped = yaml.safe_dump(config.to_yaml_dict())
        assert AppConfig.model_validate(yaml.safe_load(dumped)) == config


class TestModels:
    def test_start_is_normalised_to_utc(self, config_dir: Path) -> None:
        config = load_config(config_dir=config_dir, use_env=False)
        assert config.data.start == datetime(2020, 1, 1, tzinfo=UTC)

    def test_end_before_start_is_rejected(self, config_dir: Path) -> None:
        with pytest.raises(ConfigError, match=r"data\.end"):
            load_config(
                config_dir=config_dir,
                use_env=False,
                overrides={"data": {"end": "2019-01-01"}},
            )

    def test_duplicate_symbols_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            ExchangeConfig(symbols=["BTC/USDT", "BTC/USDT"])

    def test_symbol_format_is_checked(self) -> None:
        with pytest.raises(ValueError, match="BASE/QUOTE"):
            ExchangeConfig(symbols=["BTCUSDT"])

    def test_timeframe_minutes(self) -> None:
        assert ExchangeConfig(symbols=["BTC/USDT"], timeframe="4h").timeframe_minutes == 240

    def test_portfolio_risk_below_per_trade_risk_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_portfolio_risk_pct"):
            RiskConfig(risk_per_trade_pct=0.05, max_portfolio_risk_pct=0.03)

    def test_config_is_immutable(self, config_dir: Path) -> None:
        config = load_config(config_dir=config_dir, use_env=False)
        with pytest.raises(ValueError, match="frozen"):
            config.exchange.timeframe = "1h"  # type: ignore[misc]


class TestSecrets:
    def test_chat_ids_parsed_from_csv(self) -> None:
        secrets = load_secrets(
            {
                "TELEGRAM_BOT_TOKEN": "123:abc",
                "TELEGRAM_CHAT_IDS": "111, 222",
                "TELEGRAM_ENABLED": "true",
            }
        )
        assert secrets.chat_ids == [111, 222]
        assert secrets.bot_token is not None
        assert secrets.bot_token.get_secret_value() == "123:abc"
        assert secrets.configured is True

    def test_not_configured_without_token(self) -> None:
        assert load_secrets({"TELEGRAM_CHAT_IDS": "111"}).configured is False

    def test_token_is_not_printed(self) -> None:
        secrets = load_secrets({"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_IDS": "1"})
        assert "123:abc" not in repr(secrets)
