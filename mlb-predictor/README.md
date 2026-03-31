# MLB Predictor Rebuild

This folder is the clean rebuild lane for the MLB prediction app. It keeps the new work isolated from the legacy codebase while preserving the parts of the current stack that are still useful: local Python, local Postgres, honest feature cutoffs, and model-ready parquet snapshots.

## Product Scope

The rebuild is intentionally narrow:

- One shared data backbone.
- One totals model for full-game over/under runs.
- One player hits model for 1+ hit props.
- One starter strikeout model for pitcher K props.
- Prior-season statistics default to 2025.
- 2026 rolling samples can run in standard blend mode, lighter-prior mode, or current-season-only mode.

## Product Roadmap

For the longer-term product and UI roadmap, including compact game cards, expandable game detail,
historical outcome review, hitter heat and streak tracking, richer starter context, and a future
pitcher strikeout prediction lane, see `PRODUCT_EXCELLENCE_PLAN.md`.

## Opinionated Rules

- Every feature row must be time-safe.
- `feature_cutoff_ts` must be at or before first pitch.
- Market data is an overlay, not the heart of the model.
- Lineups, weather, park, starter form, and bullpen usage are first-class inputs.
- H2H and streaks stay lightweight.

## Project Layout

```text
mlb-predictor/
  config/
    .env.example
  db/
    migrations/
    seeds/
  data/
    raw/
    staged/
    features/
    models/
    reports/
  src/
    api/
    backtest/
    features/
    ingestors/
    models/
    transforms/
    utils/
  notebooks/
  tests/
  docker-compose.yml
  Makefile
  pyproject.toml
  requirements.txt
```

## Current Backbone

The first-pass rebuild includes:

- Canonical Postgres schema in [db/migrations/001_schema.sql](S:/Projects/AI_Predictions/mlb-predictor/db/migrations/001_schema.sql)
- Shared settings, logging, and DB utilities
- Canonical ingestors for schedule, starters, boxscores, batting, weather, market totals, and manual lineups
- Daily aggregate refresh jobs for offense and bullpen state
- Historical feature builders for totals and player hits with prior shrinkage
- Historical feature builders for totals, player hits, and starter strikeouts with prior shrinkage
- Product-surface transforms for player trends, pitcher trends, prediction outcomes, and daily model scorecards
- Baseline training and prediction entrypoints for all three lanes

## Setup

1. Create a venv.
2. Install dependencies.
3. Copy `config/.env.example` to `config/.env` and adjust values.
4. Start Postgres with Docker Compose or point `DATABASE_URL` at a local instance.
5. Apply migrations.

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item config\.env.example config\.env
docker compose up -d
make db-migrate
```

## Daily Flow

```text
ingest raw -> refresh daily aggregates -> build feature snapshots -> train or score -> persist predictions -> backfill outcomes
```

Recommended order:

1. `make prepare-slate-inputs`
2. Fill or confirm the generated lineup and market CSVs in `data/raw/` as needed.
3. `make ingest-today`
4. `make refresh-aggregates`
5. `make features-today`
6. `make predict-today`
7. `make product-surfaces`

For the strikeout lane rollout and trend/scorecard rebuild in one command:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make rollout-strikeouts START_DATE=2025-04-01 END_DATE=2026-03-31 TARGET_DATE=2026-03-31
```

For historical refreshes or dates that are missing review data, run the range pipeline:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make pipeline-range START_DATE=2025-04-01 END_DATE=2025-04-30
```

## App

The rebuild now ships with one local FastAPI app that serves both the API and a simple dashboard.

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make start-app
```

Open `http://localhost:8000` for the dashboard.

Useful endpoints:

- `GET /health` for service and database status
- `GET /api/status?date=YYYY-MM-DD` for model and prediction availability
- `GET /api/predictions/totals?date=YYYY-MM-DD` for latest totals rows
- `GET /api/predictions/hits?date=YYYY-MM-DD` for latest 1+ hit rows
- `GET /api/model-scorecards?target_date=YYYY-MM-DD` for trailing scorecard rows by market
- `GET /api/trends/players/{player_id}?target_date=YYYY-MM-DD` for hitter trend history
- `GET /api/trends/pitchers/{pitcher_id}?target_date=YYYY-MM-DD` for pitcher trend history
- `POST /api/pipeline/run` to rebuild features and rescore a target date

## Historical Training Flow

1. Backfill games, starters, boxscores, batting, pitching, weather, and markets.
2. Refresh `team_offense_daily` and `bullpens_daily`.
3. Build historical totals, hits, and strikeout features for a date range.
4. Train baselines with chronological splits.
5. Save artifacts and evaluation summaries under `data/models` and `data/reports`.

## Data Inputs

- MLB StatsAPI for schedules, games, probable starters, and boxscores
- pybaseball for starter Statcast summaries
- Open-Meteo for forecast and archive weather snapshots
- Park factors loaded from `db/seeds/park_factors.csv` when present, with automatic current-season bootstrap when the table is empty
- Generated lineup templates in `data/raw/manual_lineups.csv`, with historical inference as the default starting point
- Odds API totals snapshots when `THE_ODDS_API_KEY` is set, with `data/raw/manual_market_totals.csv` as a manual fallback or override

## Near-Term Build Order

1. Load park factors and confirm venue coordinates.
2. Backfill 2025 priors and 2026 daily operational tables.
3. Train totals baseline and validate calibration.
4. Train hits baseline once lineup coverage is stable.
5. Dial in lineup and streak features after the baseline app loop is stable.