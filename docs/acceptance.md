# Приёмочный отчёт

Чеклист раздела 18 `tech.md`. Дата прогона: **2026-09-12**.
Код на момент отчёта: тег `v1.0.0` (после коммита документации).
Исполнитель прогона: локальная машина разработчика (macOS, Python 3.13 в
`.venv`, `uv`).

Дисклеймер — в [README](../README.md). Сигналы не являются инвестиционной
рекомендацией. Цифры бэктеста не обещают будущую доходность.

## Функциональные

- [x] **Загрузка ≥5 символов за ≥5 лет, кэш**
  `tradingbot data download` + `data validate`, 2026-09-12.
  BTC 15 364 бара с 2019-09-08; ETH с 2019-11-27; XRP с 2020-01-06;
  BNB с 2020-02-10; SOL с 2020-09-14. Все пять рядов `OK`, Parquet в
  `data/ohlcv/binanceusdm/`.
- [x] **Long и short, логика §7**
  Прогон `20260912-101825-806dd076`: 452 long / 349 short.
  Правила и формулы — [`docs/strategy.md`](strategy.md);
  юнит-тесты `tests/unit/test_strategy_donchian.py`.
- [x] **Бэктест одной командой, полный набор артефактов**
  `tradingbot backtest run --note "acceptance v1.0.0 default params"`.
  В `runs/20260912-101825-806dd076/`: `meta.json`, `config.yaml`,
  `trades.parquet`, `equity.parquet`, `signals.parquet`, `metrics.json`,
  `report.html`, плюс `montecarlo.json` / `montecarlo.parquet`.
- [x] **Нет look-ahead**
  `tests/unit/test_backtest_engine.py::TestNoLookAhead` (гэп 50% на открытии)
  и `tests/unit/test_indicators.py::TestNoLookAhead` (сдвиг Дончиана).
  Входят в `pytest -m "not integration"`.
- [x] **Все метрики §9.1**
  `metrics.json` того же прогона: доходность, риск, Sharpe/Sortino/Calmar/Omega,
  торговая статистика, MAE/MFE, издержки. Сводка — в `docs/strategy.md`.
- [x] **Walk-forward, IS/OOS без приукрашивания**
  `runs/20260912-102224-40d1f54a`. 11 окон, WFE **0.262**, положительный OOS
  **64%**, `healthy: false`. Таблица окон — в `docs/strategy.md`.
- [x] **Monte Carlo, доверительные интервалы**
  1 000 итераций на журнале 801 сделки. MaxDD p5/p50/p95 = 19.3% / 29.2% / 52.7%.
  P(DD>20%)=93.1%, P(ruin)=0%. Итоговый капитал от перестановки не меняется
  (сумма PnL).
- [x] **Дашборд, 6 страниц**
  `PAGES` в `src/tradingbot/dashboard/app.py`: Overview, Trades, Chart,
  Analytics, Walk-Forward, Live. Скриншоты — `docs/images/*.png`.
- [x] **12 графиков §10.2**
  Фабрики в `reporting/charts.py`, заголовки в `html_report.CHART_TITLES`.
  Юнит-тесты `tests/unit/test_reporting.py`. В `report.html` графики вшиты.
- [x] **Автономный `report.html`**
  `runs/20260912-101825-806dd076/report.html` (~10 МБ) и WFA-отчёт
  `runs/20260912-102224-40d1f54a/report.html`. Plotly inline.
- [x] **Выбор прошлого прогона и сравнение двух**
  Сайдбар дашборда: `run_id` + «сравнить с». Тесты `tests/unit/test_dashboard.py`.
- [ ] **Live paper ≥24 часа без падений**
  В этой сессии **не выдерживалось**. Код цикла, восстановления и health —
  `src/tradingbot/live/`, тесты `test_live_runner.py` / `test_live_scheduler.py`.
  Как проверить: `make bot` → через сутки `docker compose logs bot` /
  `tradingbot live status` без traceback и с растущим `bars_processed`.
- [x] **Telegram вход/выход в формате §11.2**
  `notify/formatters.py` + `tests/unit/test_notify.py`. Макеты сообщений —
  `docs/images/telegram_entry.svg`, `telegram_exit.svg`.
  Живая доставка в этой сессии **не отправлялась** (секреты не трогались).
  Проверка у заказчика: `make telegram-test`.
- [x] **Команды бота и whitelist**
  Все команды §11.3 в `notify/bot.py` / `CommandService`.
  `TestCommands.test_help_lists_every_command`, `test_whitelist`
  (чужой `chat_id` отсекается).
- [x] **Нет дублей после рестарта**
  Уникальный `signal_uid`; `test_second_insert_of_the_same_uid_does_not_create_a_duplicate`.

## Технические

- [x] **ruff check / ruff format --check**
  Локально 2026-09-12: без нарушений (см. `make lint` в этой сессии).
- [x] **mypy --strict src/tradingbot**
  Локально: «Success: no issues found in 66 source files».
- [x] **pytest -m "not integration", покрытие ≥80%**
  Предыдущий полный прогон этапа 15: 545 passed, 1 skipped, **82%**.
  Повтор перед тегом — в логе `make test` этой сессии.
- [ ] **CI зелёный на `main`**
  До этапа 16 Actions падал на `uv pip install --system` (~1 с, шаг Install).
  В этом релизе workflow переведён на `setup-uv@v5` + `uv sync --extra dev`.
  Статус после пуша: смотреть
  https://github.com/xom9kxom9k/tradingbot/actions
- [ ] **`docker compose up -d` поднимает bot и dashboard**
  На машине прогона **нет Docker**. Артефакты (`Dockerfile`, compose,
  `.dockerignore`) и юнит-тесты `tests/unit/test_docker.py` есть.
  Проверка: `cp .env.example .env && mkdir -p data runs logs && docker compose up -d`
  → http://localhost:8501 и `docker compose logs bot`.
- [x] **Секретов в git нет**
  `.env` в `.gitignore` и `.dockerignore`. Хук `detect-private-key` в
  pre-commit. В `.env.example` только плейсхолдеры BotFather.
- [x] **Линейная история, Conventional Commits**
  `main` без merge-коммитов, сообщения `feat|chore|docs|fix(scope): …`.
- [x] **Нет Cursor / AI / Co-authored-by в истории**
  `git log` прогнан фильтром `co-authored|generated with|made with ai|cursor`:
  совпадений нет.

## Документационные

- [x] **README ≤15 минут до запуска**
  Пять шагов: `install` → `.env` → `download` → `backtest` → `dashboard`.
- [x] **`docs/strategy.md` можно повторить вручную**
  Режим, вход, стоп/R/TP, выходы, плюс таблица IS/OOS.
- [x] **`docs/architecture.md` — диаграмма и слои**
  Слои, потоки backtest/live, хранение, точки расширения.
- [x] **Дисклеймер**
  README, раздел в конце; повторён в шапке этого файла.

## Команды, которыми снимались цифры

```bash
tradingbot data download
tradingbot data validate
tradingbot data info
tradingbot backtest run --note "acceptance v1.0.0 default params"
# → 20260912-101825-806dd076   +92.06%   801 trades
tradingbot backtest report --run-id 20260912-101825-806dd076
tradingbot montecarlo run --run-id 20260912-101825-806dd076 --iterations 1000
TRADINGBOT_OPTIMIZE__N_JOBS=4 tradingbot walkforward run \
    --note "acceptance v1.0.0 rolling 18/6/6"
# → 20260912-102224-40d1f54a   WFE=0.262   healthy=false
make lint && make typecheck && make test
```

## Открытые пункты для заказчика

1. Прогон paper ≥24 ч (`make bot` или `docker compose up -d`) и скрин
   живого Telegram, если нужен не макет, а доставка.
2. `docker compose up -d` на машине с Docker Engine.
3. Дождаться зелёного CI после пуша `v1.0.0`.
