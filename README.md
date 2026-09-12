# tradingbot

Исследовательский трейдинг-бот: загрузка исторических данных с биржи, трендследящая
стратегия на пробое канала Дончиана, честный событийный бэктест, интерактивная
аналитика и уведомления о сигналах в Telegram.

Полное техническое задание — [`tech.md`](tech.md).

## Возможности

- Загрузка и инкрементальное кэширование OHLCV с Binance USDT-M Futures (`ccxt`, Parquet).
- Стратегия `donchian_trend`: пробой канала + фильтры тренда, ADX и волатильности, ATR-стопы,
  частичная фиксация и chandelier-трейлинг. Работает в обе стороны — long и short.
- Событийный бэктест с комиссиями, проскальзыванием и funding, с защитой от look-ahead.
- Полный набор метрик доходности и риска, walk-forward валидация, Monte Carlo.
- Интерактивные графики Plotly: Streamlit-дашборд и автономный `report.html`.
- Live/paper-режим с персистентным состоянием и уведомлениями в Telegram.

## Быстрый старт

```bash
make install                 # виртуальное окружение и зависимости
cp .env.example .env         # заполнить токен Telegram
make download                # исторические данные
make backtest                # бэктест
make report                  # report.html
make dashboard               # интерактивный дашборд
make telegram-test           # проверка Telegram
```

Либо `tradingbot dashboard` — тот же Streamlit-сервер. В сайдбаре выбирается `run_id`,
фильтры по символу/направлению/периоду и сравнение двух прогонов.

## Дашборд

Страницы Overview, Trades, Chart, Analytics, Walk-Forward и Live.

![Overview](docs/images/overview.png)

![Price + signals](docs/images/chart.png)

![Monthly returns](docs/images/analytics.png)

Walk-Forward и Monte Carlo читают артефакты соответствующего прогона; страница Live
показывает состояние paper-бота из SQLite (`tradingbot live status` / `make bot`).

![Walk-forward placeholder](docs/images/walkforward.png)

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

Несколько чатов — через запятую. Только эти `chat_id` могут вызывать команды
(`/status`, `/positions`, `/pause`, …); остальные запросы игнорируются.

```bash
make telegram-test              # проверить доставку
make bot                        # paper-режим: сигналы уходят в Telegram
```

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
