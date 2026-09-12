.DEFAULT_GOAL := help
UV ?= .venv/bin/uv
PY ?= .venv/bin/python
RUN := $(UV) run --

.PHONY: help install lint format typecheck test test-all check download backtest metrics report \
        optimize walkforward montecarlo dashboard bot telegram-test docker-build docker-up clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install all dependencies
	@test -d .venv || python3 -m venv .venv
	@$(PY) -m pip install --quiet --upgrade pip uv
	$(UV) pip install --python $(PY) -e ".[dev]"

lint: ## Run ruff check and format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Auto-format the codebase
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

typecheck: ## Run mypy in strict mode
	$(RUN) mypy src/tradingbot

test: ## Run unit tests with coverage
	$(RUN) pytest -m "not integration" --cov=src/tradingbot --cov-report=term-missing

test-all: ## Run every test, including integration ones
	$(RUN) pytest

check: lint typecheck test ## Run the full quality gate

download: ## Download OHLCV history for the configured symbols
	$(RUN) tradingbot data download

backtest: ## Run a backtest with the default config
	$(RUN) tradingbot backtest run

metrics: ## Show the metrics of the latest run
	$(RUN) tradingbot backtest metrics

report: ## Build report.html for the latest run
	$(RUN) tradingbot backtest report --open

optimize: ## Run a grid parameter search
	$(RUN) tradingbot optimize grid

walkforward: ## Run walk-forward analysis
	$(RUN) tradingbot walkforward run

montecarlo: ## Run monte carlo simulation on the latest run
	$(RUN) tradingbot montecarlo run

dashboard: ## Launch the streamlit dashboard
	$(RUN) streamlit run dashboard/app.py

bot: ## Start the live runner in paper mode
	$(RUN) tradingbot live run --mode paper

telegram-test: ## Send a test message to the configured Telegram chats
	$(RUN) tradingbot telegram test

docker-build: ## Build the docker image
	docker compose build

docker-up: ## Start bot and dashboard containers
	mkdir -p data runs logs
	docker compose up -d

clean: ## Remove caches and build artefacts
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
