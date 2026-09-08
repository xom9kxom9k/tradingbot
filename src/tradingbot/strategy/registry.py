"""Name-to-class registry so strategies can be selected from configuration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from tradingbot.core.exceptions import ConfigError
from tradingbot.strategy.base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}

StrategyT = TypeVar("StrategyT", bound=Strategy)


def register(cls: type[StrategyT]) -> type[StrategyT]:
    """Class decorator adding a strategy to the registry under its ``name``.

    Raises:
        ConfigError: The name is missing or already taken.
    """
    name = getattr(cls, "name", "")
    if not name or name == "abstract":
        raise ConfigError(f"{cls.__name__} must define a non-empty class attribute 'name'")
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not cls:
        raise ConfigError(f"strategy name {name!r} is already registered by {existing.__name__}")
    _REGISTRY[name] = cls
    return cls


def available() -> list[str]:
    """Sorted list of registered strategy names."""
    return sorted(_REGISTRY)


def get_strategy_class(name: str) -> type[Strategy]:
    """Look up a strategy class by name.

    Raises:
        ConfigError: No strategy is registered under that name.
    """
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(available()) or "none"
        raise ConfigError(f"unknown strategy {name!r}; registered strategies: {known}") from exc


def create_strategy(name: str, params: dict[str, Any] | None = None, **kwargs: Any) -> Strategy:
    """Instantiate a registered strategy from a plain parameter mapping.

    Args:
        name: Registered strategy name.
        params: Strategy-specific parameters, validated by the strategy itself.
        **kwargs: Extra constructor arguments, such as ``risk_per_trade_pct``.

    Raises:
        ConfigError: The name is unknown or the parameters fail validation.
    """
    factory: Callable[..., Strategy] = get_strategy_class(name).from_params  # type: ignore[attr-defined]
    return factory(params or {}, **kwargs)
