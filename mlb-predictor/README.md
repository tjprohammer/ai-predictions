# MLB Predictor Rebuild

This folder is the clean rebuild lane for the MLB prediction app. It keeps the new work isolated from the legacy codebase while preserving the parts of the current stack that are still useful: local Python, **SQLite by default** (optional Postgres for power-user dev), honest feature cutoffs, and model-ready parquet snapshots.

## Product Scope

The rebuild is intentionally narrow:

- One shared data backbone.
- One totals model for full-game over/under runs.
- One first-five totals model for first-five over/under runs.
- One player hits model for 1+ hit props.
- One starter strikeout model for pitcher K props.
- Prior-season statistics default to 2025.
- 2026 rolling samples can run in standard blend mode, lighter-prior mode, or current-season-only mode.

## Product Roadmap

For the longer-term product and UI roadmap, including compact game cards, expandable game detail,
historical outcome review, hitter heat and streak tracking, richer starter context, and a future
pitcher strikeout prediction lane, see [docs/PRODUCT_EXCELLENCE_PLAN.md](docs/PRODUCT_EXCELLENCE_PLAN.md).

## Architecture Docs

- [docs/PRODUCT_EXCELLENCE_PLAN.md](docs/PRODUCT_EXCELLENCE_PLAN.md) for long-term product and UX direction
- [docs/MODEL_REWORK_PLAN.md](docs/MODEL_REWORK_PLAN.md) for the certainty-aware modeling plan across totals, hits, first-five totals, and strikeouts
- [docs/FEATURE_DICTIONARY.md](docs/FEATURE_DICTIONARY.md) for the seeded feature inventory and contract scaffold used by the rework audit
- [docs/CALIBRATION_AND_PRODUCT.md](docs/CALIBRATION_AND_PRODUCT.md) for trust buckets, best-bet history verification, and Phase 5–6 operational checks
- [docs/TOTALS_NARRATIVE.md](docs/TOTALS_NARRATIVE.md) for how board “totals lean” copy relates to model lane vs heuristics
- [docs/TODOS.md](docs/TODOS.md) for the current engineering backlog (API routing cleanup, experimental markets, etc.)

### API application layout

- **`src/api/app_logic.py`** — Request helpers, SQL-backed fetchers, recommendation formatting, and shared constants used by routes and jobs.
- **`src/api/app.py`** — Instantiates `FastAPI`, applies `security` middleware, re-exports `app_logic` symbols into `src.api.app` (so tests and direct helper calls keep one namespace), includes the aggregated router from `src/api/routers/http_routes.py` (which composes `meta`, `html`, `api_feed`, `games`, `jobs`, and `ops` sub-routers), and copies each route endpoint onto `src.api.app` for tests that call handlers by name.
- **`src/api/security.py`** — Local middleware (host/CORS/API key) for the dev server.
- **Static UI** — `src/api/static/` (HTML/JS/CSS) served by route handlers in the route definitions file.

## Opinionated Rules

- Every feature row must be time-safe.
- `feature_cutoff_ts` must be at or before first pitch.
- Market data is an overlay, not the heart of the model.
- Lineups, weather, park, starter form, and bullpen usage are first-class inputs.
- H2H and streaks stay lightweight.

## Project Layout

```text
mlb-predictor/
  docs/
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
- Historical feature builders for full-game totals, first-five totals, player hits, and starter strikeouts with prior shrinkage
- Product-surface transforms for player trends, pitcher trends, prediction outcomes, and daily model scorecards
- Baseline training and prediction entrypoints for full-game totals, first-five totals, player hits, and starter strikeouts

## Setup

1. Create a venv.
2. Install dependencies.
3. Copy `config/.env.example` to `config/.env` and adjust values.
4. **SQLite (default):** run migrations; a project database file is created at `db/mlb_predictor.sqlite3` when needed.
5. **Optional Postgres:** start Docker Compose and set `DATABASE_URL` in `.env`, then apply migrations.

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item config\.env.example config\.env
# Optional: docker compose up -d
make db-migrate
```

## Daily Flow

```text
ingest raw -> refresh daily aggregates -> build feature snapshots -> train or score -> persist predictions -> backfill outcomes
```

Recommended order:

1. `make prepare-slate-inputs`
2. Fill or confirm the generated lineup and market CSVs in `data/raw/` as needed. `manual_market_totals.csv` now seeds full-game totals, first-five totals via `first_five_total`, and player prop templates. For strikeouts, use the generated `pitcher_strikeouts` rows as optional overrides when you have better numbers. `make ingest-today` only fails when those blank strikeout templates still have no matching live or manual market line.
3. `make ingest-today`
4. `make refresh-aggregates`
5. `make features-today`
6. `make predict-today`
7. `make product-surfaces`

If you want to train the first-five model explicitly after backfilling labeled first-five results:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make train-first5-totals
```

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

`make start-app` runs uvicorn with **`--reload`**, so edits to Python under `src/` (API, routers, `app_logic`, etc.) are picked up automatically after you save; you do not need to restart the process by hand unless reload fails or you change the environment outside the watched tree.

If you start uvicorn yourself instead of Make, add **`--reload`** for the same behavior. For a one-off run without auto-reload (closer to production), omit `--reload` and restart the process after code changes.

**Desktop shell (`make run-desktop`):** the embedded server does not use reload. Restart the desktop app to load API code changes.

**Ingest logs** (for example Odds API lines about `totals_1st_1_innings` / NRFI–YRFI) appear in the terminal where **`market_totals`** or the pipeline runs (`make ingest-today`, dashboard jobs, etc.), not in the browser. Refresh the API or hit `/api/doctor` to see data that ingest has already written to the database.

`make start-app` binds **127.0.0.1** by default so the JSON API is not exposed to your LAN. Override with `MLB_PREDICTOR_HOST=0.0.0.0` only if you intend to reach the API from other machines; in that case set `MLB_PREDICTOR_API_KEY` in the environment and send `Authorization: Bearer <key>` or `X-MLB-Predictor-Api-Key` on requests (see `config/.env.example`).

Open `http://localhost:8000` for the dashboard.

**macOS and Linux:** There is no separate “Mac web build.” The dashboard is a normal browser UI served by FastAPI; on a Mac, use the same Python venv + `make start-app` (or `uvicorn src.api.app:app --host 127.0.0.1 --port 8000`) and open Safari or Chrome to `http://localhost:8000`. The Windows installer/PyInstaller path is optional; cross-platform use is “run the API locally, use the web UI.” For a shared team URL, deploy the same app behind HTTPS on a server (Docker, systemd, etc.)—still one web app for all clients.

Useful endpoints:

- `GET /health` for service and database status
- `GET /api/status?date=YYYY-MM-DD` for model and prediction availability
- `GET /api/predictions/totals?date=YYYY-MM-DD` for latest totals rows
- `GET /api/predictions/hits?date=YYYY-MM-DD` for latest 1+ hit rows
- `GET /api/model-scorecards?target_date=YYYY-MM-DD` for trailing scorecard rows by market
- `GET /api/trends/players/{player_id}?target_date=YYYY-MM-DD` for hitter trend history
- `GET /api/trends/pitchers/{pitcher_id}?target_date=YYYY-MM-DD` for pitcher trend history
- `POST /api/pipeline/run` to rebuild features and rescore a target date

## Desktop Packaging

There is now a first Windows desktop-shell path for the app. This wraps the existing FastAPI app, seeds the bundled `data/` snapshot into a user-writable runtime directory, starts the local API server in the background, and opens a desktop window when `pywebview` is installed.

### For Testers

The best way to test the packaged app is to download a release asset from the repository's GitHub Releases page, not the source-code ZIP from the main repo page.

Current Windows release flow:

1. Open the latest GitHub Release.
2. Download the versioned Windows asset for that release, usually `MLBPredictor-Windows-v<version>-PortableInstaller.zip` or `MLBPredictor-Windows-v<version>-Setup.exe` when an Inno Setup installer was published.
3. If you downloaded the portable ZIP, extract it and double-click `install_mlb_predictor.cmd`. `install_mlb_predictor.ps1` is still included, but some Windows setups open `.ps1` files in an editor instead of executing them.
4. Launch `MLBPredictor` from the shortcut or install folder.

What happens on first launch:

- the app creates a per-user runtime folder under `%LOCALAPPDATA%\MLBPredictor`
- the local SQLite database is created at `%LOCALAPPDATA%\MLBPredictor\db\mlb_predictor.sqlite3`
- bundled schema migrations are applied automatically
- bundled park-factor reference data is seeded automatically

This means a normal tester does not need to install Postgres or run bootstrap commands just to open the desktop app.

### Updating Data vs Updating the App

These are two separate workflows:

1. Daily data updates: use the app's Update Center or other in-app update actions to refresh schedules, features, and predictions for a target day. You do not need to download a new GitHub release every day just to refresh the slate data.
2. App updates: when the desktop code changes, download the next GitHub Release and install it over the existing app.

There is not an auto-updater for the desktop binary yet. App upgrades are currently manual release downloads.

Because the installed app files and the per-user SQLite database live in separate locations, installing a newer release should keep your existing local app data unless you manually delete `%LOCALAPPDATA%\MLBPredictor`.

Install the extra desktop dependencies:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
.venv\Scripts\pip install -r requirements-desktop.txt
```

Run the desktop shell locally:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make run-desktop
```

Build a Windows distributable with PyInstaller:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make build-desktop
```

The desktop packaging step now validates that `db\mlb_predictor.sqlite3` contains the bundled historical SQLite seed required for rebuild flows. If those history tables are empty, the build fails fast instead of silently shipping a stale desktop runtime. For intentional UI-only packaging during local debugging, you can override that guard with `python scripts\build_windows_app.py --allow-incomplete-sqlite-seed`.

Build an installer on top of the desktop bundle:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make build-installer
```

For a strict `Setup.exe` build that fails loudly when Inno Setup is not installed:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_installer.py --app-version 0.1.0-beta1 --require-inno
```

Run the full Windows release flow in one step:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
make build-release
```

For tagged beta artifacts with versioned names:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\build_windows_release.py --app-version 0.1.0-beta1 --require-inno
```

If Inno Setup is installed, this produces a versioned standard Windows installer such as `MLBPredictor-Windows-v0.1.0-beta1-Setup.exe`. If not, the build falls back to a versioned portable installer bundle such as `MLBPredictor-Windows-v0.1.0-beta1-PortableInstaller.zip` with `install_mlb_predictor.ps1` and `uninstall_mlb_predictor.ps1` plus the packaged app folder.

The full release build now also writes sidecar release metadata for the selected app version:

- `MLBPredictor-Windows-v0.1.0-beta1-manifest.json`
- `MLBPredictor-Windows-v0.1.0-beta1-checksums.txt`
- `MLBPredictor-Windows-v0.1.0-beta1-release-notes.md`

For a local runtime/operator check before or after packaging, you can inspect the aggregated health payload directly:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python -m src.utils.doctor --json
```

For a local desktop smoke run that checks health, doctor, board, scorecards, top-miss review, and CLV review against either a running app or a booted local server:

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\run_desktop_smoke.py --json
```

To exercise every main Update Center action against the running app as part of release validation, add `--exercise-update-actions`. You can narrow the run with repeated `--update-action` flags if you only want specific actions.

```powershell
cd S:\Projects\AI_Predictions\mlb-predictor
python scripts\run_desktop_smoke.py --json --exercise-update-actions
```

The portable bundle also includes `install_mlb_predictor.cmd` and `uninstall_mlb_predictor.cmd` so testers can launch install or uninstall by double-clicking on a typical Windows machine.

For a repeatable beta ship flow, see [docs/BETA_RELEASE_CHECKLIST.md](docs/BETA_RELEASE_CHECKLIST.md).

The desktop runtime now defaults to a per-user SQLite file under `%LOCALAPPDATA%\MLBPredictor\db\mlb_predictor.sqlite3` when no custom `DATABASE_URL` is configured. Explicit `DATABASE_URL` overrides are still respected for Postgres-based workflows.

On desktop SQLite startups, the launcher now applies the bundled migrations automatically before the API comes up, so a first-run local database file is initialized with the app schema.

On top of schema creation, the desktop runtime now seeds bundled park-factor reference data automatically on SQLite first-run startup, so local installs come up with baseline reference rows without manual bootstrap commands.

Desktop startup also now repairs an older default local SQLite runtime when its historical seed is incomplete but the bundled app database is healthier. That keeps upgraded local installs from getting stuck on stale desktop-history blockers after the packaged seed has improved.

Current limitation: this removes the external database requirement for startup and base schema creation, but it is not the full embedded-database cutover yet. The remaining migration work is porting the broader ingestion and training queries that still rely on Postgres-specific SQL semantics.

## Embedded DB Groundwork

The first migration step away from the external Postgres requirement is now in place:

- the shared DB utility layer supports SQLite-safe engine creation and SQLite upserts
- table existence checks now go through SQLAlchemy inspection instead of the Postgres-only `information_schema` path
- the API layer now uses dialect-aware SQL helpers for list binds, seasonal year filters, ratio math, JSON text extraction, and SQLite-safe boolean parsing in the main hit/totals surfaces

This is not the full embedded-database cutover yet. The remaining blockers are the larger set of Postgres-specific raw SQL queries and migrations, but the utility layer no longer hard-locks the app to Postgres-only upsert behavior.

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
- Odds API totals snapshots when `THE_ODDS_API_KEY` is set, with `data/raw/manual_market_totals.csv` as a manual fallback or override for full-game and first-five totals plus player props

## Near-Term Build Order

1. Load park factors and confirm venue coordinates.
2. Backfill 2025 priors and 2026 daily operational tables.
3. Train totals baseline and validate calibration.
4. Train hits baseline once lineup coverage is stable.
5. Dial in lineup and streak features after the baseline app loop is stable.
