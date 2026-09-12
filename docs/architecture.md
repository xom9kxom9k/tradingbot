# Архитектура

tradingbot — исследовательский контур: одни и те же правила стратегии считаются
на истории (бэктест) и на закрытых свечах в paper/signal-режиме. Домен не ходит
в сеть, не пишет в SQLite и не шлёт Telegram — поэтому бэктест и live нельзя
случайно разъехать.

## Слои

```
┌──────────────────────────────────────────────────────────┐
│  Интерфейсы:  CLI (typer) │ Streamlit Dashboard │ Telegram │
└───────────────┬──────────────────┬───────────────┬────────┘
                │                  │               │
┌───────────────▼──────────────────▼───────────────▼────────┐
│  Приложение: BacktestRunner │ LiveRunner │ Optimizer │ WFA │
└───────────────┬───────────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────┐
│  Домен: Strategy │ Indicators │ RiskManager │ Portfolio    │
│         Signal / Position / Trade                          │
└───────────────┬───────────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────────┐
│  Инфраструктура: DataFeed(ccxt) │ ParquetCache │ SQLite    │
│                  Notifier(Telegram) │ Logger │ Config      │
└────────────────────────────────────────────────────────────┘
```

| Слой | Пакет | Что знает |
|---|---|---|
| Интерфейсы | `cli.py`, `dashboard/`, `notify/bot.py` | Как вызвать приложение, не как считать сигнал |
| Приложение | `backtest/runner.py`, `live/runner.py`, `analytics/` | Оркестрация: загрузить данные, прогнать движок, сохранить артефакты |
| Домен | `strategy/`, `indicators/`, `risk/`, `core/models.py` | Правила и инварианты. Вход — `DataFrame` / `BarContext`, выход — `Signal` |
| Инфраструктура | `data/`, `storage/`, `notify/telegram.py`, `config/` | Биржа, диск, транспорт сообщений, YAML/env |

`Strategy` получает закрытые бары и возвращает `Signal | None`. Она не знает про
ccxt, SQLite и aiogram. `BacktestEngine` и `LiveRunner` вызывают один и тот же
класс из реестра (`strategy/registry.py`).

## Поток данных

### История → бэктест → отчёт

```
биржа (публичный OHLCV)
    → CcxtDataFeed.fetch_range
    → ParquetCache (data/ohlcv/<exchange>/<symbol>/<tf>.parquet)
    → BacktestRunner.load_data + validate_ohlcv
    → Strategy.prepare (индикаторы без look-ahead)
    → BacktestEngine: open → стопы/тейки → funding → on_bar → mark
    → RiskManager одобряет или режет размер
    → PaperBroker (комиссия, проскальзывание, гэп)
    → runs/<run_id>/{meta,config,trades,equity,signals,metrics,report.html}
    → Streamlit / report.html / Telegram /optimize|walkforward|montecarlo
```

Порядок бара зафиксирован в [`docs/backtest.md`](backtest.md) и тестом
`tests/unit/test_backtest_engine.py::TestNoLookAhead`. Сигнал с закрытия `t`
исполняется только по открытию `t+1`.

### Live / paper

```
планировщик (закрытие свечи + bar_close_grace_sec)
    → догрузка хвоста в ParquetCache
    → тот же BacktestEngine на новых закрытых барах
    → persist: signals, trades, positions, equity, pending, cash, risk halt
    → очередь уведомлений (SQLite) → Telegram
    → health в key/value (tradingbot health / live status / страница Live)
```

Первый старт не переигрывает всю историю: курсоры встают на последний готовый
бар (`arm_cursors`). После SIGINT/SIGTERM состояние восстанавливается из
`data/state.db`.

`paper` ведёт виртуальный портфель. `signal_only` пишет те же сигналы, но не
открывает позицию.

## Хранение

| Что | Где | Зачем |
|---|---|---|
| OHLCV | `data/ohlcv/…/*.parquet` | Инкрементальный кэш; формирующаяся свеча не пишется |
| Прогоны | `runs/<id>/` | Воспроизводимые артефакты, дашборд, сравнение |
| Live-состояние | `data/state.db` (SQLite) | Позиции, очередь Telegram, health, пауза |
| Логи | `logs/tradingbot.log` + `.jsonl` | Ротация, контекст `run_id` / `symbol` / `bar_ts` |
| Секреты | `.env` (`TELEGRAM_*`) | Никогда не в YAML и не в образ |

Идентификатор прогона: `YYYYMMDD-HHMMSS-<config hash>`. `signal_uid` уникален:
повторная вставка того же сигнала после рестарта не создаёт второе уведомление.

## Конфигурация

Приоритет (сильнее → слабее): флаги CLI → `TRADINGBOT_*` → YAML (поздний файл
побеждает) → дефолты Pydantic.

Типичный набор файлов:

- `configs/base.yaml` — биржа, риск, издержки, пути
- `configs/strategy_donchian_trend.yaml` — параметры стратегии
- `configs/backtest.yaml` или `configs/live.yaml` — профиль запуска

Секреты только из окружения / `.env` (`TelegramSecrets`). В контейнере они
приходят через `env_file: .env`.

## Точки расширения

Новая **стратегия**

1. Подкласс `strategy.base.Strategy` (`prepare`, `on_bar`, `warmup_period`).
2. Декоратор `@register`, уникальный `name`.
3. YAML: `strategy.name` + `strategy.params`.
4. Юнит-тесты на синтетических свечах (вход, выход, нет look-ahead).

Новый **индикатор** — чистая функция в `indicators/`, только по данным `≤ t`.

Другая **биржа** — реализация `data.feed.DataFeed` (как `CcxtDataFeed`).

Другой **канал уведомлений** — `notify.base.Notifier`. Live всегда зовёт
`SafeNotifier`: исключение транспорта не рвёт торговый цикл.

Другая **целевая функция** оптимизации — имя в `optimize.objective`
(`sharpe` / `calmar` / `profit_factor` / `custom`).

## Процессы в проде

`docker compose` поднимает один образ дважды:

- `bot` — `tradingbot live run --mode paper` + healthcheck `tradingbot health`
- `dashboard` — Streamlit на `0.0.0.0:8501` (иначе в контейнере не достучаться)

Данные, прогоны и логи — bind-mounts. Контейнер работает от uid 1000.
Подробности — раздел «Развёртывание» в [README](../README.md).
