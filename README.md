# tradingbot

Исследовательский трейдинг-бот: загрузка исторических данных с биржи, трендследящая
стратегия на пробое канала Дончиана, честный событийный бэктест, интерактивная
аналитика и уведомления о сигналах в Telegram.

Полное техническое задание — [`tech.md`](tech.md).
Стратегия и цифры прогонов — [`docs/strategy.md`](docs/strategy.md).
Слои и потоки данных — [`docs/architecture.md`](docs/architecture.md).
Приёмка — [`docs/acceptance.md`](docs/acceptance.md).

## Возможности

- Загрузка и инкрементальное кэширование OHLCV с Binance USDT-M Futures (`ccxt`, Parquet).
- Стратегия `donchian_trend`: пробой канала + фильтры тренда, ADX и волатильности, ATR-стопы,
  частичная фиксация и chandelier-трейлинг. Работает в обе стороны — long и short.
- Событийный бэктест с комиссиями, проскальзыванием и funding, с защитой от look-ahead.
- Полный набор метрик доходности и риска, walk-forward валидация, Monte Carlo.
- Интерактивные графики Plotly: Streamlit-дашборд и автономный `report.html`.
- Live/paper-режим с персистентным состоянием и уведомлениями в Telegram.

## Быстрый старт

Пять шагов до первого отчёта (около 10–15 минут при уже установленном Python 3.11+):

```bash
make install                          # 1. виртуальное окружение и зависимости
cp .env.example .env                  # 2. секреты Telegram (можно заполнить позже)
make download                         # 3. история OHLCV с 2019-01-01
make backtest                         # 4. прогон, артефакты в runs/<id>/
make dashboard                        # 5. Streamlit на http://localhost:8501
```

После шага 4: `make report` пишет автономный `runs/<id>/report.html`.
После шага 2: `make telegram-test` проверяет доставку, `make bot` стартует paper-режим.

Либо `tradingbot dashboard` — тот же сервер. В сайдбаре выбирается `run_id`,
фильтры по символу/направлению/периоду и сравнение двух прогонов.

## Дашборд

Страницы Overview, Trades, Chart, Analytics, Walk-Forward и Live.

![Overview](docs/images/overview.png)

![Price + signals](docs/images/chart.png)

![Monthly returns](docs/images/analytics.png)

Walk-Forward и Monte Carlo читают артефакты соответствующего прогона; страница Live
показывает состояние paper-бота из SQLite (`tradingbot live status` / `make bot`).

![Walk-forward](docs/images/walkforward.png)

## Telegram

1. Напишите `@BotFather` в Telegram, команда `/newbot` — получите токен.
2. Напишите своему боту любое сообщение, затем откройте
   `https://api.telegram.org/bot<TOKEN>/getUpdates` и скопируйте `chat.id`.
3. Положите значения в `.env`:

```
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_IDS=111111111
TELEGRAM_ENABLED=true
```

Несколько чатов — через запятую. Только эти `chat_id` могут вызывать команды;
остальные запросы игнорируются.

Формат входа и выхода (раздел 11.2 ТЗ, рендерит `notify/formatters.py`):

![Сигнал на вход](docs/images/telegram_entry.svg)

![Сигнал на выход](docs/images/telegram_exit.svg)

Команды бота: `/start`, `/status`, `/positions`, `/equity`, `/stats [7d|30d|all]`,
`/last [n]`, `/chart <SYMBOL>`, `/backtest`, `/pause`, `/resume`, `/config`, `/help`.

```bash
make telegram-test              # проверить доставку
make bot                        # paper-режим: сигналы уходят в Telegram
```

## CLI

```bash
tradingbot version
tradingbot health                          # Docker healthcheck: live runner running?
tradingbot config show                     # эффективный YAML после merge

tradingbot data download [--symbols …] [--timeframe 4h] [--start 2019-01-01]
tradingbot data validate
tradingbot data info

tradingbot backtest run [--symbols …] [--note …]
tradingbot backtest list
tradingbot backtest metrics [--run-id …] [--json]
tradingbot backtest report [--run-id …] [--open] [--png]

tradingbot optimize grid [--objective calmar] [--jobs N]
tradingbot optimize optuna [--trials 200] [--objective calmar]
tradingbot walkforward run [--mode rolling] [--is-months 18] [--oos-months 6]
tradingbot montecarlo run [--run-id …] [--iterations 1000]

tradingbot live run [--mode paper|signal_only]
tradingbot live status

tradingbot dashboard [--port 8501] [--address localhost] [--headless]
tradingbot telegram test
```

Повторяющийся `--config` / `-c` наслаивает YAML (поздний файл побеждает).
Переменные `TRADINGBOT_SECTION__KEY` перекрывают YAML.

## Развёртывание

Один образ, два сервиса: `bot` (live/paper) и `dashboard` (порт 8501). Данные,
прогоны и логи лежат в bind-mounts `data/`, `runs/`, `logs/` и переживают
перезапуск контейнера.

### Локально

```bash
cp .env.example .env            # токен и chat_id Telegram
mkdir -p data runs logs
make docker-build
make docker-up                  # или: docker compose up -d
```

Дашборд: http://localhost:8501  
Логи бота: `docker compose logs -f bot`

Остановка: `docker compose down`. Состояние paper-бота остаётся в `data/state.db`.

### VPS

На чистой машине с Docker Engine и Docker Compose v2:

```bash
sudo apt-get update && sudo apt-get install -y git docker.io docker-compose-v2
sudo usermod -aG docker "$USER"   # затем перелогиниться
git clone https://github.com/xom9kxom9k/tradingbot.git
cd tradingbot
cp .env.example .env
# заполнить TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_IDS
mkdir -p data runs logs
# контейнер работает от uid 1000 и пишет в эти каталоги
sudo chown -R 1000:1000 data runs logs
docker compose up -d --build
```

Секреты только в `.env`, в образ они не копируются.

Откройте порт **8501** только для себя (firewall / SSH-туннель
`ssh -L 8501:127.0.0.1:8501 user@vps`). Сам бот наружу портов не публикует.

Обновление:

```bash
git pull
docker compose up -d --build
```

Первый старт без кэша OHLCV: бот подтянет свечи с биржи в `data/ohlcv`.
На слабом VPS удобнее один раз сделать `make download` локально и скопировать
`data/` на сервер.

## FAQ

**Нужен ли API-ключ биржи?** Нет. Используются только публичные свечи Binance USDT-M.

**Почему бэктест не совпал с «идеальным» пробоем на графике?** Сигнал с закрытия
бара исполняется по открытию следующего. Это защита от look-ahead, не баг.

**Можно ли торговать реальными деньгами?** Нет. В объёме проекта — paper и
signal_only. Реальный execution out of scope.

**Куда смотреть, если paper-бот «молчит»?** `tradingbot live status` и логи.
На 4h следующий сигнал возможен только после закрытия свечи (+ `bar_close_grace_sec`).

**Как добавить свою стратегию?** Подкласс `Strategy`, `@register`, имя в YAML.
См. [`docs/architecture.md`](docs/architecture.md) → «Точки расширения».

**Почему дашборд в Docker недоступен?** Streamlit по умолчанию слушает
`localhost`. В compose передаётся `--address 0.0.0.0`.

**Где цифры walk-forward?** В `docs/strategy.md` и в `runs/<id>/walkforward.json`.
Если WFE < 0.5 или положительный OOS реже 60% окон — это пишется как есть.

## Разработка

```bash
make lint        # ruff check + ruff format --check
make typecheck   # mypy --strict
make test        # pytest без интеграционных тестов, с покрытием
make check       # всё вместе
```

## Дисклеймер

Проект является исследовательским инструментом. Сигналы не являются индивидуальной
инвестиционной рекомендацией. Результаты бэктестов не гарантируют будущей доходности.
Автор не несёт ответственности за финансовые потери, возникшие при использовании
этого программного обеспечения.
