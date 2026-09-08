"""Loading and merging of configuration from YAML, environment and CLI.

Precedence, from strongest to weakest: CLI overrides, environment variables,
YAML files (later files win over earlier ones), model defaults.

Environment variables use the ``TRADINGBOT_`` prefix and ``__`` as the nesting
separator, so ``TRADINGBOT_EXCHANGE__TIMEFRAME=1h`` sets ``exchange.timeframe``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from tradingbot.config.models import AppConfig, TelegramSecrets
from tradingbot.core.exceptions import ConfigError

ENV_PREFIX = "TRADINGBOT_"
NESTING_SEPARATOR = "__"

DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_CONFIG_FILES: tuple[str, ...] = (
    "base.yaml",
    "strategy_donchian_trend.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Read a single YAML file into a dictionary.

    Args:
        path: File to read.

    Returns:
        Parsed mapping; an empty file yields an empty dict.

    Raises:
        ConfigError: The file is missing, unreadable or is not a mapping.
    """
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(raw).__name__}"
        )
    return raw


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` without mutating either."""
    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = deepcopy(value)
    return merged


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build a nested override mapping from ``TRADINGBOT_*`` variables.

    Values are parsed as YAML scalars, so ``true``, ``42`` and ``[a, b]`` arrive
    as the corresponding Python types.
    """
    source = os.environ if environ is None else environ
    overrides: dict[str, Any] = {}
    for raw_key, raw_value in source.items():
        if not raw_key.startswith(ENV_PREFIX):
            continue
        path = [
            part.lower() for part in raw_key[len(ENV_PREFIX) :].split(NESTING_SEPARATOR) if part
        ]
        if not path:
            continue
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError:
            value = raw_value
        _assign(overrides, path, value)
    return overrides


def _assign(target: dict[str, Any], path: list[str], value: Any) -> None:
    """Set ``value`` at the nested ``path`` inside ``target``, creating levels."""
    cursor = target
    for part in path[:-1]:
        nested = cursor.get(part)
        if not isinstance(nested, dict):
            nested = {}
            cursor[part] = nested
        cursor = nested
    cursor[path[-1]] = value


def resolve_config_paths(
    paths: Iterable[Path | str] | None = None,
    config_dir: Path | None = None,
) -> list[Path]:
    """Return the ordered list of YAML files to merge.

    When ``paths`` is omitted the default set from ``configs/`` is used, skipping
    optional files that do not exist.
    """
    if paths is not None:
        return [Path(item) for item in paths]
    directory = config_dir or DEFAULT_CONFIG_DIR
    resolved = [directory / name for name in DEFAULT_CONFIG_FILES]
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise ConfigError(
            "default config files are missing: " + ", ".join(str(path) for path in missing)
        )
    return resolved


def load_config(
    paths: Iterable[Path | str] | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
    config_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
    use_env: bool = True,
) -> AppConfig:
    """Build the effective :class:`AppConfig`.

    Args:
        paths: Explicit list of YAML files, merged left to right. Defaults to the
            standard set inside ``configs/``.
        overrides: Highest-priority nested mapping, typically built from CLI flags.
        config_dir: Directory holding the default config files.
        environ: Environment mapping to read overrides from; defaults to ``os.environ``.
        use_env: Set to ``False`` to ignore the environment entirely.

    Returns:
        A validated configuration object.

    Raises:
        ConfigError: Any file is missing or the merged result fails validation.
    """
    merged: dict[str, Any] = {}
    for path in resolve_config_paths(paths, config_dir):
        merged = deep_merge(merged, load_yaml(path))
    if use_env:
        merged = deep_merge(merged, env_overrides(environ))
    if overrides:
        merged = deep_merge(merged, overrides)

    try:
        return AppConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(f"invalid configuration:\n{_format_errors(exc)}") from exc


def load_secrets(environ: Mapping[str, str] | None = None) -> TelegramSecrets:
    """Read Telegram credentials from the environment and ``.env``."""
    if environ is None:
        return TelegramSecrets()
    values: dict[str, Any] = {
        key[len("TELEGRAM_") :].lower(): value
        for key, value in environ.items()
        if key.startswith("TELEGRAM_")
    }
    return TelegramSecrets(**values)


def _format_errors(exc: ValidationError) -> str:
    """Render pydantic errors as one readable ``section.field: message`` per line."""
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        lines.append(f"  - {location}: {error['msg']}")
    return "\n".join(lines)
