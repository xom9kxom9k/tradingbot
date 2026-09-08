"""Exception hierarchy shared by every layer of the application."""

from __future__ import annotations


class TradingBotError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(TradingBotError):
    """Configuration is missing, malformed or fails validation."""


class DataError(TradingBotError):
    """Base class for data acquisition and storage problems."""


class FeedError(DataError):
    """The exchange feed could not deliver the requested candles."""


class DataValidationError(DataError):
    """Cached or downloaded OHLCV data violates the quality contract."""


class StrategyError(TradingBotError):
    """A strategy failed while preparing data or evaluating a bar."""


class RiskError(TradingBotError):
    """Position sizing or risk accounting produced an impossible result."""


class BacktestError(TradingBotError):
    """The backtest engine reached an inconsistent state."""


class StorageError(TradingBotError):
    """Persistence layer failure."""


class NotificationError(TradingBotError):
    """A notification could not be delivered.

    Callers inside the trading loop must never let this propagate; see
    :mod:`tradingbot.notify.base` for the safety wrapper.
    """
