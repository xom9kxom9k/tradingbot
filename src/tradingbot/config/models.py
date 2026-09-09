"""Pydantic models describing every configuration section.

The whole application is driven by :class:`AppConfig`. Anything that can differ
between runs lives here; secrets never do — those come from the environment via
:class:`TelegramSecrets`.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

Timeframe = Literal["15m", "1h", "4h", "1d"]
SlippageModel = Literal["fixed_bps", "atr", "percent"]
LiveMode = Literal["paper", "signal_only"]
LogLevel = Literal["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]

Fraction = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]

TIMEFRAME_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240, "1d": 1440}


class StrictModel(BaseModel):
    """Base model that rejects unknown keys so typos fail loudly at startup."""

    model_config = ConfigDict(extra="forbid", frozen=True, validate_assignment=True)


class ExchangeConfig(StrictModel):
    """Which venue, instruments and candle size the system works with."""

    name: str = "binanceusdm"
    symbols: list[str] = Field(min_length=1)
    timeframe: Timeframe = "4h"

    @field_validator("symbols")
    @classmethod
    def _unique_symbols(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("symbols must not contain duplicates")
        for symbol in value:
            if "/" not in symbol:
                raise ValueError(f"symbol {symbol!r} must use the BASE/QUOTE form")
        return value

    @property
    def timeframe_minutes(self) -> int:
        """Length of one candle in minutes."""
        return TIMEFRAME_MINUTES[self.timeframe]


class DataConfig(StrictModel):
    """History window and on-disk cache settings."""

    start: datetime = datetime(2019, 1, 1, tzinfo=UTC)
    end: datetime | None = None
    cache_dir: Path = Path("data/ohlcv")
    max_gap_bars: int = Field(default=3, ge=0)

    @field_validator("start", "end")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @model_validator(mode="after")
    def _ordered(self) -> DataConfig:
        if self.end is not None and self.end <= self.start:
            raise ValueError("data.end must be later than data.start")
        return self


class BacktestConfig(StrictModel):
    """Account settings for a simulated run."""

    initial_capital: PositiveFloat = 10_000.0
    currency: str = "USDT"
    seed: int = 42
    runs_dir: Path = Path("runs")
    # Annual rate used by Sharpe and Sortino; zero is the honest default for a
    # strategy whose capital sits in a stablecoin.
    risk_free_rate: Fraction = 0.0


class CostsConfig(StrictModel):
    """Trading frictions applied by the paper broker."""

    taker_fee: Fraction = 0.0005
    maker_fee: Fraction = 0.0002
    slippage_model: SlippageModel = "atr"
    slippage_bps: float = Field(default=5.0, ge=0.0)
    slippage_atr_mult: float = Field(default=0.05, ge=0.0)
    slippage_percent: float = Field(default=0.0005, ge=0.0)
    funding_rate_8h: float = 0.0001
    apply_funding: bool = True


class RiskConfig(StrictModel):
    """Per-trade and portfolio level risk limits."""

    risk_per_trade_pct: Fraction = 0.01
    max_concurrent_positions: int = Field(default=3, ge=1)
    max_positions_per_side: int = Field(default=3, ge=1)
    max_notional_pct: Fraction = 0.35
    max_portfolio_risk_pct: Fraction = 0.03
    daily_loss_limit_pct: Fraction = 0.05
    max_drawdown_stop_pct: Fraction = 0.25

    @model_validator(mode="after")
    def _portfolio_risk_reachable(self) -> RiskConfig:
        if self.max_portfolio_risk_pct < self.risk_per_trade_pct:
            raise ValueError(
                "risk.max_portfolio_risk_pct must be at least risk.risk_per_trade_pct, "
                "otherwise no trade can ever be opened"
            )
        return self


class StrategyConfig(StrictModel):
    """Strategy selector plus its free-form parameter block.

    Parameters stay untyped here on purpose: each strategy validates its own
    schema, which keeps the registry open for new strategies.
    """

    name: str = "donchian_trend"
    params: dict[str, Any] = Field(default_factory=dict)


def _default_optimize_grid() -> dict[str, list[Any]]:
    return {
        "entry_channel": [15, 20, 25],
        "adx_min": [15.0, 20.0, 25.0],
    }


ObjectiveName = Literal["sharpe", "calmar", "profit_factor", "custom"]
WalkForwardMode = Literal["rolling", "anchored"]


class OptimizeConfig(StrictModel):
    """Search space and ranking rules for parameter studies (section 9.3)."""

    objective: ObjectiveName = "custom"
    min_trades: int = Field(default=30, ge=1)
    optuna_trials: int = Field(default=40, ge=1)
    heatmap_x: str = "entry_channel"
    heatmap_y: str = "adx_min"
    n_jobs: int = Field(default=1, ge=1)
    grid: dict[str, list[Any]] = Field(default_factory=_default_optimize_grid)


class WalkForwardConfig(StrictModel):
    """In-sample / out-of-sample windowing (section 9.2)."""

    mode: WalkForwardMode = "rolling"
    is_months: int = Field(default=18, ge=1)
    oos_months: int = Field(default=6, ge=1)
    step_months: int = Field(default=6, ge=1)


class MonteCarloConfig(StrictModel):
    """Bootstrap settings for shuffled-trade paths (section 9.4)."""

    iterations: int = Field(default=1000, ge=10)
    ruin_equity_pct: Fraction = 0.0
    dd_threshold_pct: float = Field(default=0.20, ge=0.0, le=1.0)


class LiveConfig(StrictModel):
    """Behaviour of the live/paper trading loop."""

    mode: LiveMode = "paper"
    poll_interval_sec: int = Field(default=30, ge=1)
    bar_close_grace_sec: int = Field(default=10, ge=0)
    state_db: Path = Path("data/state.db")


class NotifyConfig(StrictModel):
    """Notification policy (transport credentials live in the environment)."""

    telegram_enabled: bool = True
    daily_report_utc: time = time(9, 0)
    notify_rejected_signals: bool = False
    max_messages_per_minute: int = Field(default=20, ge=1)

    @field_validator("daily_report_utc", mode="before")
    @classmethod
    def _parse_time(cls, value: object) -> object:
        """Accept both ``HH:MM`` and full ISO times so snapshots round-trip."""
        if isinstance(value, str):
            try:
                return time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"{value!r} is not a valid HH:MM time") from exc
        return value


class LoggingConfig(StrictModel):
    """Sinks and verbosity for loguru."""

    level: LogLevel = "INFO"
    file: Path = Path("logs/tradingbot.log")
    json_file: Path = Path("logs/tradingbot.jsonl")
    rotation: str = "10 MB"
    retention: str = "14 days"


class AppConfig(StrictModel):
    """Effective configuration for a single process."""

    exchange: ExchangeConfig
    strategy: StrategyConfig = StrategyConfig()
    data: DataConfig = DataConfig()
    backtest: BacktestConfig = BacktestConfig()
    costs: CostsConfig = CostsConfig()
    risk: RiskConfig = RiskConfig()
    live: LiveConfig = LiveConfig()
    notify: NotifyConfig = NotifyConfig()
    logging: LoggingConfig = LoggingConfig()
    optimize: OptimizeConfig = OptimizeConfig()
    walkforward: WalkForwardConfig = WalkForwardConfig()
    montecarlo: MonteCarloConfig = MonteCarloConfig()

    def to_yaml_dict(self) -> dict[str, Any]:
        """Plain, YAML-serialisable snapshot of the effective configuration."""
        snapshot = _jsonify(self.model_dump(mode="json"))
        assert isinstance(snapshot, dict)
        return snapshot


class TelegramSecrets(BaseSettings):
    """Telegram credentials, sourced exclusively from the environment/`.env`."""

    model_config = SettingsConfigDict(
        env_prefix="TELEGRAM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    bot_token: SecretStr | None = None
    chat_ids: list[int] = Field(default_factory=list)
    enabled: bool = True

    @field_validator("chat_ids", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(part) for part in value.replace(" ", "").split(",") if part]
        return value

    @property
    def configured(self) -> bool:
        """True when the bot can actually talk to somebody."""
        return bool(self.bot_token and self.chat_ids)


def _jsonify(value: Any) -> Any:
    """Recursively convert pydantic dump output into YAML-friendly primitives."""
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
