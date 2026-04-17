# MLB Predictor – local dev helpers
# ==================================
PYTHON   ?= python
PG_URL   ?= postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb
VENV     := .venv
PYTHON_BIN := $(VENV)\Scripts\python.exe
PIP_BIN := $(VENV)\Scripts\pip.exe
PYTEST_BIN := $(VENV)\Scripts\pytest.exe
BLACK_BIN := $(VENV)\Scripts\black.exe
ISORT_BIN := $(VENV)\Scripts\isort.exe
UVICORN_BIN := $(VENV)\Scripts\uvicorn.exe

TARGET_DATE ?= $(END_DATE)

.PHONY: help venv install db-up db-down db-migrate db-reset \
	load-reference-data prepare-slate-inputs ingest-today refresh-aggregates features-today predict-today pipeline \
	train-totals train-first5-totals train-hits train-strikeouts backtest compare-prior-modes start-app start-app-stable run-desktop install-desktop build-desktop build-installer build-release test lint clean pipeline-range product-surfaces \
	rollout-strikeouts

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────
venv: ## Create virtual environment
	$(PYTHON) -m venv $(VENV)
	"$(PIP_BIN)" install --upgrade pip

install: ## Install dependencies into venv
	"$(PIP_BIN)" install -r requirements.txt

install-desktop: ## Install desktop shell dependencies into venv
	"$(PIP_BIN)" install -r requirements-desktop.txt

# ── Database ─────────────────────────────────────────────
db-up: ## Start Postgres + Adminer via Docker
	docker compose up -d

db-down: ## Stop Docker services
	docker compose down

db-migrate: ## Run all SQL migrations
	"$(PYTHON_BIN)" -m src.utils.db_migrate

db-reset: ## Drop and re-create all tables (DESTRUCTIVE)
	@echo Dropping and recreating database...
	docker compose exec db psql -U mlbuser -d mlb -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
	$(MAKE) db-migrate

# ── Pipeline ─────────────────────────────────────────────
load-reference-data: ## Load park factors and bootstrap current-season reference rows
	"$(PYTHON_BIN)" -m src.ingestors.park_factors

prepare-slate-inputs: ## Generate editable lineup and market template CSVs for today's slate
	"$(PYTHON_BIN)" -m src.ingestors.prepare_slate_inputs

ingest-today: ## Run all ingestors for today
	"$(PYTHON_BIN)" -m src.ingestors.games
	"$(PYTHON_BIN)" -m src.ingestors.park_factors
	"$(PYTHON_BIN)" -m src.ingestors.starters
	"$(PYTHON_BIN)" -m src.ingestors.boxscores
	"$(PYTHON_BIN)" -m src.ingestors.player_batting
	"$(PYTHON_BIN)" -m src.ingestors.lineups_backfill
	"$(PYTHON_BIN)" -m src.ingestors.matchup_splits
	"$(PYTHON_BIN)" -m src.ingestors.pitcher_statcast
	"$(PYTHON_BIN)" -m src.ingestors.lineups
	"$(PYTHON_BIN)" -m src.ingestors.player_status
	"$(PYTHON_BIN)" -m src.ingestors.weather
	"$(PYTHON_BIN)" -m src.ingestors.market_totals --require-player-prop-coverage pitcher_strikeouts

refresh-aggregates: ## Refresh daily shared-backbone aggregate tables
	"$(PYTHON_BIN)" -m src.transforms.offense_daily
	"$(PYTHON_BIN)" -m src.transforms.bullpens_daily

features-today: ## Build features for today's slate
	"$(PYTHON_BIN)" -m src.features.totals_builder
	"$(PYTHON_BIN)" -m src.features.first5_totals_builder
	"$(PYTHON_BIN)" -m src.features.inning1_nrfi_builder
	"$(PYTHON_BIN)" -m src.features.hits_builder
	"$(PYTHON_BIN)" -m src.features.hr_builder
	"$(PYTHON_BIN)" -m src.features.strikeouts_builder

predict-today: ## Generate predictions for today
	"$(PYTHON_BIN)" -m src.models.predict_totals
	"$(PYTHON_BIN)" -m src.models.predict_first5_totals
	"$(PYTHON_BIN)" -m src.models.predict_inning1_nrfi
	"$(PYTHON_BIN)" -m src.models.predict_hits
	"$(PYTHON_BIN)" -m src.models.predict_hr
	"$(PYTHON_BIN)" -m src.models.predict_strikeouts

train-hr: ## Train home-run probability model from HR feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_hr

train-first5-totals: ## Train first-five totals baseline from feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_first5_totals

train-inning1-nrfi: ## Train inning-1 NRFI/YRFI classifier from feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_inning1_nrfi

product-surfaces: ## Build trend tables and model scorecards for today
	"$(PYTHON_BIN)" -m src.transforms.product_surfaces

rollout-strikeouts: ## Apply migrations and rebuild strikeout/trend surfaces: make rollout-strikeouts START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD [TARGET_DATE=YYYY-MM-DD]
	@if "$(START_DATE)"=="" (echo START_DATE is required & exit /b 1)
	@if "$(END_DATE)"=="" (echo END_DATE is required & exit /b 1)
	@if "$(TARGET_DATE)"=="" (echo TARGET_DATE is required & exit /b 1)
	$(MAKE) db-migrate
	"$(PYTHON_BIN)" -m src.utils.backfill_dim_players --missing-only
	"$(PYTHON_BIN)" -m src.features.strikeouts_builder --start-date $(START_DATE) --end-date $(END_DATE)
	"$(PYTHON_BIN)" -m src.models.train_strikeouts
	"$(PYTHON_BIN)" -m src.models.predict_strikeouts --target-date $(TARGET_DATE)
	"$(PYTHON_BIN)" -m src.transforms.product_surfaces --start-date $(START_DATE) --end-date $(END_DATE)

pipeline: ingest-today refresh-aggregates features-today predict-today ## Full daily pipeline

pipeline-range: ## Backfill a date range: make pipeline-range START_DATE=YYYY-MM-DD END_DATE=YYYY-MM-DD
	"$(PYTHON_BIN)" -m src.utils.pipeline_range --start-date $(START_DATE) --end-date $(END_DATE)

train-totals: ## Train totals baseline from feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_totals

train-hits: ## Train hits baseline from feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_hits

train-strikeouts: ## Train pitcher strikeout baseline from feature snapshots
	"$(PYTHON_BIN)" -m src.models.train_strikeouts

backtest: ## Run walk-forward backtest
	"$(PYTHON_BIN)" -m src.backtest.run

compare-prior-modes: ## Compare standard_tuned, reduced, and current_only prior modes
	"$(PYTHON_BIN)" -m src.backtest.compare_prior_modes --train-start-date 2025-04-01 --train-end-date 2025-04-12 --target-date 2026-03-30

MLB_PREDICTOR_HOST ?= 127.0.0.1
MLB_PREDICTOR_PORT ?= 8000

start-app: ## Run FastAPI + dashboard locally with uvicorn --reload (loopback; MLB_PREDICTOR_HOST=0.0.0.0 for LAN)
	"$(UVICORN_BIN)" src.api.app:app --reload --reload-exclude "data/**" --reload-exclude "db/**" --host $(MLB_PREDICTOR_HOST) --port $(MLB_PREDICTOR_PORT)

start-app-stable: ## Same API as start-app but WITHOUT --reload — use for long “Update Lineups & Markets” / pipeline jobs
	"$(UVICORN_BIN)" src.api.app:app --host $(MLB_PREDICTOR_HOST) --port $(MLB_PREDICTOR_PORT)

run-desktop: ## Launch the desktop shell locally
	"$(PYTHON_BIN)" -m src.desktop.launcher

build-desktop: ## Build the Windows desktop shell with PyInstaller
	"$(PYTHON_BIN)" scripts\build_windows_app.py

build-installer: ## Build a Windows installer or portable installer bundle from dist/MLBPredictor
	"$(PYTHON_BIN)" scripts\build_windows_installer.py

build-release: ## Build the Windows desktop app and installer bundle in one command
	"$(PYTHON_BIN)" scripts\build_windows_release.py

# ── Quality ──────────────────────────────────────────────
test: ## Run tests
	"$(PYTEST_BIN)" tests/ -v

lint: ## Format and lint
	"$(BLACK_BIN)" src/ tests/
	"$(ISORT_BIN)" src/ tests/

clean: ## Remove caches and temp files
	del /s /q __pycache__ 2>nul
	del /s /q *.pyc 2>nul
	del /s /q .pytest_cache 2>nul

