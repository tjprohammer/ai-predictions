# MLB Predictor Detailed Architecture and Release Plan

## Goal

Turn the current MLB Predictor into a trustworthy desktop product that:

1. pulls and validates its own data,
2. stores everything it needs in a local app-owned database,
3. runs daily prediction workfl([github.com](https://github.com/tjprohammer/ai-predictions/blob/main/mlb-predictor/README.md))inside the app,
4. grades its own past predictions,
5. and ships as a downloadable Windows release for non-technical users.

---

## 1) Product outcomes

The product should support four user jobs:

### A. Prepare today’s slate

The user opens the app and immediately sees:

- today’s games,
- probable starters,
- weather,
- market totals,
- top hitters,
- strikeout context,
- model edges,
- data freshness,
- and whether lineups are projected or confirmed.

### B. Refresh today’s data

The user clicks one button to refresh today’s slate inputs and another button to rebuild predictions.

### C. Review yesterday’s outcomes

The user can review:

- predicted totals,
- market totals,
- final scores,
- which props hit or missed,
- and how the model performed by market.

### D. Install and run locally

A normal user should be able to download an installer, open the app, and use it without setting up Postgres, Python, or a CLI.

---

## 2) Product principles

1. **Time-safe only**: every prediction row must be reproducible from information available before first pitch.
2. **Market is context, not truth**: odds help calibrate and rank, but the model cannot become a copy of the sportsbook line.
3. **One source-of-truth database**: the app owns its local database and writes all operational data there.
4. **Every ingestion is auditable**: every refresh run records source, timestamp, status, and row counts.
5. **Every prediction is gradeable**: predictions must always be stored with version, feature cutoff, and eventual outcome.
6. **App updates and data updates are separate**: shipping a new installer is not the same as refreshing today’s slate.

---

## 3) Recommended technical architecture

## Runtime architecture

### Desktop shell

- Python desktop wrapper around the existing local FastAPI app
- pywebview or the current browser-shell approach for rendering the UI
- background API server started by the desktop launcher

### Local API

- FastAPI remains the main backend
- exposes read APIs for board/detail/history/trends
- exposes write APIs for ingestion jobs, prediction runs, and grading jobs

### Local database

Use **SQLite as the production runtime database** for shipped desktop installs.

Why:

- zero setup for users,
- good enough for single-user desktop workloads,
- simple backups,
- easy bundling,
- easy reset/reseed behavior.

### Analytics / training storage

Use:

- SQLite for operational app state and product surfaces
- Parquet files for model-ready historical snapshots
- model artifacts on disk under `data/models/`

Optional:

- keep DuckDB as a dev-only analysis helper if you want faster notebook-style experimentation
- keep Postgres only for development if there are still queries that are easier to iterate there

### Background job layer

Add one internal job runner with these properties:

- job_id
- job_type
- target_date
- status
- started_at
- finished_at
- rows_written
- warning_count
- error_message
- initiated_by

The UI should poll job status instead of blocking.

---

## 4) Data source strategy

Build around a **tiered source model**, not a single fragile feed.

## Tier 1: canonical game backbone

Use for:

- schedule
- game ids
- teams
- start times
- status
- probable starters
- final scores
- official boxscore outcomes

This should be the canonical game spine that all other data maps onto.

## Tier 2: advanced performance stats

Use for:

- rolling hitter quality
- rolling pitcher quality
- xwOBA
- hard-hit rate
- whiff indicators
- CSW-like derivatives
- rolling strikeout skill

This tier powers feature engineering.

## Tier 3: weather

Use for:

- forecast at prediction time
- observed weather for outcome review
- wind direction/speed
- temp
- humidity
- pressure if available

Important rule:

- pregame predictions should use the forecast that would have been available before first pitch,
- postgame evaluation can store observed or archived actual weather separately.

## Tier 4: markets

Use for:

- full game totals
- first five totals
- pitcher strikeout lines
- player hit props if you decide to support them broadly

Important rule:

- store each market snapshot with bookmaker, pull timestamp, event id, and normalized side/line metadata.
- never overwrite the “closest pregame” line used by a prediction.

## Tier 5: manual overrides

Use for:

- lineups not yet confirmed,
- missing props,
- corrected starter assignments,
- emergency market fixes.

Manual rows should always beat auto-ingested rows when marked as active.

---

## 5) Data quality and reliability plan

This is the biggest missing piece.

For every ingestor, implement the same contract:

### Required fields

- `source_name`
- `source_record_id`
- `ingested_at`
- `source_event_ts`
- `target_date`
- `game_id`
- `raw_payload_hash`
- `parse_status`
- `validation_status`
- `warning_flags`

### Validation rules

Examples:

- start time must exist and parse to UTC
- game_id must map to a known game
- away/home teams must map to canonical ids
- starter player ids must map to a canonical player record
- market totals must be positive and within plausible bounds
- lineup rows must have batting order positions in valid range
- odds snapshots after first pitch should not be used for pregame feature sets

### Freshness rules

Each product surface should expose freshness:

- schedule freshness
- weather freshness
- market freshness
- lineup freshness
- score freshness

### Confidence rules

Every game card should show a data confidence badge, for example:

- **green**: confirmed lineup + fresh market + good starter mapping
- **yellow**: projected lineup or stale market snapshot
- **red**: missing starter or missing market

---

## 6) Canonical database design

Use a **normalized core + denormalized product surfaces** pattern.

## Core tables

### Reference tables

- `teams`
- `players`
- `venues`
- `sportsbooks`
- `park_factors`
- `model_registry`

### Game backbone

- `games`
- `game_status_history`
- `game_participants`
- `probable_starters`
- `starting_lineups`
- `lineup_versions`

### Raw ingestion tables

- `raw_schedule_ingest`
- `raw_boxscore_ingest`
- `raw_weather_ingest`
- `raw_market_ingest`
- `raw_statcast_ingest`
- `raw_lineup_ingest`

### Derived daily state

- `team_offense_daily`
- `bullpens_daily`
- `player_batting_form_daily`
- `pitcher_form_daily`
- `venue_weather_daily`
- `market_closing_lines`

### Feature tables

- `game_features_totals`
- `game_features_first5_totals`
- `player_features_hits`
- `game_features_pitcher_strikeouts`

### Prediction tables

- `predictions_totals`
- `predictions_first5_totals`
- `predictions_hits`
- `predictions_pitcher_strikeouts`

### Outcome / grading tables

- `prediction_outcomes_daily`
- `model_scorecards_daily`
- `prediction_calibration_bins`
- `recommendation_history`

### Job / app control tables

- `pipeline_runs`
- `pipeline_run_steps`
- `app_settings`
- `data_source_health`
- `release_metadata`

---

## 7) Feature engineering expansion

## Totals model

Expand features into these buckets:

- team rolling offense,
- starter quality,
- bullpen state,
- weather,
- park,
- schedule fatigue,
- lineup strength,
- handedness splits,
- recent run environment,
- market context.

Recommended additional totals features:

- last 7 / 14 day offense splits
- team wRC+ proxy or equivalent rolling run-creation proxy
- top 5 hitters projected quality score
- projected lineup continuity score
- starter recent pitch count trend
- starter third-time-through-order risk proxy
- bullpen leverage usage last 3 days
- travel distance / getaway-day flag
- roof/open-air flag
- umpire later if coverage is dependable

## Hits model

Add:

- projected lineup slot
- PA expectation
- opposing starter handedness split
- recent hard-hit trend
- contact trend
- chase / whiff trend if stable
- opposing bullpen contact environment
- park singles factor / hit factor if you build it
- hit streak state captured as a feature, not a standalone decision rule

## Pitcher strikeout model

Add:

- projected innings distribution
- batters faced expectation
- opponent projected lineup K tendency
- opponent contact quality tendency
- recent swinging strike trend
- called strike proxy / CSW proxy
- pitch count leash trend
- rest days
- platoon mix
- weather / park effects on pitch count sustainability if measurable
- market strikeout line as an optional calibration feature only

---

## 8) Product surfaces and APIs

## Primary API surfaces

### Slate board

`GET /api/games/board?target_date=YYYY-MM-DD`

For each game return:

- game metadata
- market summary
- totals prediction summary
- first five summary
- starter summary
- top hitters per team
- strikeout summary
- freshness badges
- confidence badge
- lock status

### Game detail

`GET /api/games/{game_id}`

Return:

- full lineups
- team trend blocks
- starter trend blocks
- totals rationale block
- pitcher K rationale block
- player hit cards
- recent matchup context
- final result if complete
- links to historical prediction snapshots

### Historical slate

`GET /api/history/slate?target_date=YYYY-MM-DD`

Return:

- all predictions as they were at lock time
- final outcomes
- hit/miss grading
- top misses and top wins
- data health notes

### Model analytics

- `GET /api/model-scorecards?target_date=YYYY-MM-DD`
- `GET /api/trends/players/{player_id}`
- `GET /api/trends/pitchers/{pitcher_id}`
- `GET /api/recommendations/history`

## Job APIs

- `POST /api/jobs/prepare-slate`
- `POST /api/jobs/sync-markets`
- `POST /api/jobs/sync-results`
- `POST /api/jobs/build-features`
- `POST /api/jobs/run-predictions`
- `POST /api/jobs/grade-predictions`
- `POST /api/jobs/retrain-models`
- `GET /api/jobs/{job_id}`

---

## 9) App buttons / automation design

Your buttons should map to very clear steps.

## Update Center

### Button 1: Prepare Slate Inputs

Runs:

- schedule refresh
- probable starters refresh
- weather forecast refresh
- lineup template generation
- base market pull

### Button 2: Refresh Markets

Runs:

- odds sync only
- line reconciliation
- line freeze candidate creation

### Button 3: Build Predictions

Runs:

- aggregate refresh
- feature generation
- model scoring
- product surface refresh

### Button 4: Grade Recent Predictions

Runs:

- final result sync
- market close reconciliation
- prediction outcome grading
- scorecard rebuild

### Button 5: Retrain Models

Runs:

- historical backfill checks
- train totals
- train first five
- train hits
- train strikeouts
- write evaluation report

### Button 6: Open Logs / Run History

Shows:

- run status
- warnings
- rows ingested
- skipped games
- failed mappings

---

## 10) Release / packaging plan

Do **not** change direction away from your current Windows packaging path unless it becomes painful. You already have the right idea:

- local desktop shell,
- local API,
- user-writable runtime directory,
- bundled seed data,
- installer + portable option.

## Recommended release structure

### Installer output

Ship these assets on GitHub Releases:

- `MLBPredictor-Windows-v<version>-Setup.exe`
- `MLBPredictor-Windows-v<version>-PortableInstaller.zip`
- `checksums.txt`
- release notes

### Runtime layout

Per-user runtime directory:

- `db/app.sqlite3`
- `models/*.joblib` or equivalent
- `logs/*.log`
- `exports/*.csv`
- `cache/`
- `config/user_settings.json`

### First launch behavior

On first launch:

1. create runtime dirs
2. create SQLite database
3. apply migrations
4. seed park factors / reference data
5. validate bundled models exist
6. show setup health screen

### App update behavior

App updates should:

- replace binaries,
- preserve runtime DB,
- preserve exports/logs/settings,
- run migration checks on next launch.

### Data update behavior

Daily data updates should happen entirely in-app.
Do not require users to download a new release for normal daily use.

### Release channels

Create at least two channels:

- **stable**
- **beta**

Use beta for testing new ingestion logic or new model lanes.

### Auto-update later

Not required immediately.
When ready, add:

- update manifest
- current version endpoint or release metadata file
- in-app “Download latest release” prompt

---

## 11) What must be finished before the app is truly standalone

This is the critical checkpoint.

The app is not fully standalone until:

1. all runtime reads and writes work on SQLite,
2. all migrations run automatically on SQLite,
3. all default product surfaces can populate without Postgres,
4. prediction refresh jobs work without external DB setup,
5. grading jobs work without external DB setup,
6. the app can start clean on a fresh machine using only the installer.

If some heavy historical training queries still need Postgres, that is acceptable temporarily **only if**:

- the shipped app does not need them for normal user workflows,
- and training is clearly marked as a power-user or developer action.

---

## 12) Recommended next implementation order

## Phase 1: Make the shipped app operationally complete

1. finish SQLite runtime compatibility for read/write product paths
2. add pipeline run tables and job history
3. add data freshness badges
4. add explicit grading workflow for recent slates
5. make the board/detail/history flows stable

## Phase 2: Make ingestion trustworthy

1. add canonical IDs and source mapping tables
2. add raw ingest audit tables
3. add validation and confidence scoring
4. add market snapshot versioning
5. add forecast-vs-observed weather separation

## Phase 3: Make the model easier to trust

1. scorecards by market
2. calibration pages
3. bad-pick review
4. hit streak label auditing
5. explained prediction blocks in game detail

## Phase 4: Make the product deeper

1. first five polish
2. pitcher strikeout lane completion
3. recommendation history
4. filters and saved views
5. better exports

---

## 13) Acceptance criteria

You are done with the first strong version when:

- a new user can install the app from GitHub Releases,
- the app launches without Python/Postgres setup,
- the user can prepare today’s slate,
- the user can build predictions,
- the user can grade yesterday’s predictions,
- every game card shows freshness/confidence status,
- every prediction stores its model version and feature cutoff,
- and scorecards make it obvious whether the model is actually improving.

---

## 14) Strong recommendation

Your highest-value move is **not** adding more models first.

It is this sequence:

1. finish the standalone runtime database story,
2. lock down ingestion quality and auditability,
3. make grading and scorecards first-class,
4. then expand model depth.

I’ve got the repo map. The good news is the folders already line up with the work: src/ingestors, src/features, src/models, src/transforms, src/api, src/desktop, db/migrations, scripts, and the Windows installer files are all in place, so this is mostly a “tighten and extend” job rather than a restructure.

Thought for 1m 58s

Yes. Here’s the concrete repo checklist I’d use next.

The repo already has the right spine for this: a local FastAPI app, static dashboard pages, dedicated ingestors, feature builders, model train/predict entrypoints, transforms, a desktop launcher, Windows packaging scripts, installer assets, schema migrations, and testslow into discrete steps: prepare-slate-inputs, ingest-today, refresh-aggregates, features-today, predict-today, product-surfaces, run-desktop, build-desktop, build-installer, and build-release. ath is not a big rewrite. It is to harden three things in this order:

standalone runtime on SQLite for normal users,
reliable audited ingestion,
graded prediction history and release polish.
That fits the product direction already described in your current plan for compact slate views, detail views, and historical review. /`
config/.env.example

Add and normalize these groups:

app mode: APP_ENV, APP_DB_MODE=sqlite|postgres, DATABASE_URL
runtime paths: APP_RUNTIME_DIR, APP_LOG_DIR, APP_EXPORT_DIR
source toggles: ENABLE_ODDS_API, ENABLE_WEATHER_API, ENABLE_MANUAL_MARKETS, ENABLE_MANUAL_LINEUPS
data quality knobs: MARKET_STALE_MINUTES, WEATHER_STALE_MINUTES, ALLOW_PROJECTED_LINEUPS
model controls: MODEL_VERSION, TRAIN_START_DATE, TRAIN_END_DATE, TARGET_DATE, PRIOR_MODE
packaging/update metadata: APP_VERSION, RELEASE_CHANNEL=stable|beta

Definition of done:

one .env supports local dev,
one no-Postgres .env supports packaged desktop,
app starts cleanly with SQLite-only defaults.
The repo already expects env-driven setup and currently documents both local Postgres and SQLite desktop runtime paths, so this is the right place to make the split explicit. ### db/migrations/
You already have 001_schema.sql through 006_first5_totals.sql. Add the next migrations instead of editing old ones. db/migrations/007_ingestion_audit.sql
Add:
pipeline_runs
pipeline_run_steps
source_health
raw_ingest_events
raw_payload_registry

Purpose:

every ingestor run records source, target date, started/finished timestamps, rows inserted, warnings, and errors.
db/migrations/008_market_snapshot_versioning.sql

Add:

market_snapshot_versions
player_prop_snapshot_versions
market_selection_freezes

Purpose:

keep the actual pregame line used by the model,
stop overwriting market rows with newer pulls.
db/migrations/009_lineup_and_weather_versioning.sql

Add:

lineup_versions
weather_forecast_snapshots
weather_observed_snapshots

Purpose:

separate forecast-at-prediction-time from observed-postgame weather.
db/migrations/010_prediction_grading.sql

Add:

prediction_grades
recommendation_history
model_calibration_bins
daily_model_rollups

Purpose:

make “recent predictions and their outcomes” first-class in the app.
db/migrations/011_sqlite_compat_cleanup.sql

Move remaining Postgres-only SQL into dialect-safe versions where possible.
The README explicitly says the remaining blocker is broader ingestion/training queries that still rely on Postgres-specific SQL semantics, so this migration should be aimed directly at that cutoff. /README.md`
Expand it with:

exact required templates for packaged app first run
which CSVs are optional overrides versus required seeds
which files are safe for users to edit manually
db/seeds/

Add or formalize:

park_factors.csv
team_aliases.csv
player_aliases.csv
sportsbook_aliases.csv
venue_coordinates.csv

The repo already uses seeded park factors and manual lineup/market templates, so aliases and venue reference data are the next practical seeds to formalize. ils/`

Current files are db.py, db_migrate.py, settings.py, logging.py, pipeline_range.py, backfill_dim_players.py, and cli.py. s/settings.py`
Add typed settings for:

source toggles
runtime paths
stale thresholds
release channel
SQLite/Postgres mode
model artifact paths
src/utils/db.py

This is one of the most important files now.
Add:

dialect-safe helpers for upserts, JSON extraction, date math, list binds
reusable insert/update helpers for snapshot-versioned tables
helper for marking a “frozen” market snapshot per game/market
helper for job tracking rows
src/utils/db_migrate.py

Add:

migration table verification
automatic migration-on-startup for desktop
dry-run / validate mode for CI
src/utils/logging.py

Add structured logs for:

job_id
source_name
target_date
game_id
row counts
warnings
failed mappings
src/utils/cli.py

Add unified commands for:

prepare-slate
refresh-markets
build-predictions
grade-predictions
retrain-models
doctor
New: src/utils/doctor.py

Add a desktop health check:

DB write test
migrations current?
models exist?
API keys present?
latest job failures?
raw data folders writable? 4) src/ingestors/

Current ingestors are games.py, starters.py, boxscores.py, player_batting.py, pitcher_statcast.py, lineups.py, weather.py, market_totals.py, park_factors.py, prepare_slate_inputs.py, plus common.py. stors/common.py`
This should become the ingestion contract.
Add a shared pattern for every source:

source_name
source_record_id
ingested_at
target_date
game_id
payload hash
parse status
validation status
warnings list
stale/fresh classification
src/ingestors/games.py

Add:

canonical MLB game spine
game status history
first-pitch UTC normalization
team and venue alias mapping
src/ingestors/starters.py

Add:

starter confidence field
probable vs confirmed distinction
starter source timestamp
starter change tracking
src/ingestors/boxscores.py

Add:

final score grading hooks
actual totals
actual first-five totals when derivable
starter final line and bullpen contribution extraction
src/ingestors/player_batting.py

Add:

rolling hit-form surfaces
strict date cutoff protections
lineup-slot join support
src/ingestors/pitcher_statcast.py

Add:

rolling recent K indicators
workload trend
batters faced trend
better fallback behavior when Statcast coverage is sparse
src/ingestors/lineups.py

Add:

projected vs confirmed states
versioning by snapshot timestamp
source URL storage
slot validation
src/ingestors/weather.py

Add:

forecast snapshot saved at prediction time
observed weather saved separately for grading
stale detection by game start proximity
src/ingestors/market_totals.py

Add:

snapshot versioning
“closest pregame frozen line” logic
support for totals, first-five, and pitcher strikeouts through one normalized contract
bookmaker normalization layer
src/ingestors/prepare_slate_inputs.py

This should generate better operator-facing templates.
Add:

editable CSVs with confidence columns
missing-data flags
default starter strikeout template lines
clear “this game still needs manual review” markers
New: src/ingestors/validator.py

Centralize cross-source checks:

game exists?
market mapped?
starter mapped?
venue mapped?
lineup complete enough?
forecast present?
source too stale? 5) src/features/

Current files are common.py, contracts.py, priors.py, totals_builder.py, first5_totals_builder.py, hits_builder.py, and strikeouts_builder.py. ures/contracts.py`
Make this the hard feature schema.
For every feature row require:

game_id
target_date
feature_cutoff_ts
data_confidence_score
market_snapshot_id
weather_snapshot_id
lineup_version_id
starter_version_id
model_family
src/features/common.py

Add:

one join layer for game, team, player, weather, lineup, and market context
one safety guard that rejects rows where feature_cutoff_ts > first_pitch
src/features/priors.py

Add:

explicit prior blending modes
audit fields showing prior weight used
simpler fallback for low-sample hitters and pitchers
src/features/totals_builder.py

Add:

lineup strength score
bullpen fatigue
travel/rest flag
park/open-roof context
team offense last 7/14
starter recent pitch count trend
market snapshot provenance
src/features/first5_totals_builder.py

Add:

heavier starter weighting
reduced bullpen dependence
first-five-specific target/grade logic
src/features/hits_builder.py

Add:

projected lineup slot
PA expectation
handedness split
recent hits in last 3/5/7
rolling hard-hit proxy
confirmed/projected lineup flag
src/features/strikeouts_builder.py

Add:

opponent projected lineup K tendency
projected batters faced
recent swinging-strike / K-rate trend
leash/pitch-count trend
market freeze id for strikeout props 6) src/models/

Current files are train*\* and predict*\* for totals, first-five, hits, and strikeouts, plus common.py. ls/common.py`
Add:

model registry helper
artifact metadata writer
evaluation summary writer
calibration helper
prediction record serializer
src/models/train_totals.py

Add:

saved model metadata
calibration report
feature importance export
training window logging
src/models/train_first5_totals.py

Add:

first-five-specific calibration and scorecards
src/models/train_hits.py

Add:

probability calibration by confidence bucket
split by confirmed vs projected lineup
src/models/train_strikeouts.py

Add:

pitcher-K-specific evaluation
threshold probability outputs
src/models/predict_totals.py

Add:

frozen market comparison
explanation payload for UI
confidence tier
reasons for no-bet / low-confidence rows
src/models/predict_first5_totals.py

Same as totals, but separate market family.

src/models/predict_hits.py

Add:

store recommended players plus rationale fields used in UI
result-ready row shape for historical grading
src/models/predict_strikeouts.py

Add:

threshold probabilities
fair odds estimates
edge versus frozen market line 7) src/transforms/

Current files are offense_daily.py, bullpens_daily.py, and product_surfaces.py. sforms/offense_daily.py`
Add:

rolling windows used by totals and hits
team-vs-handedness surfaces
“current form” table materialization
src/transforms/bullpens_daily.py

Add:

bullpen usage last 1/3/5 days
leverage proxy
fatigue classification
src/transforms/product_surfaces.py

This should become the place where the UI gets its dense, ready-to-render views.
Add:

slate board rows
game detail rows
trend cards
recent outcomes
scorecards
recommendation history
freshness/confidence badges
New: src/transforms/grading.py

Add:

totals grade
first-five grade
hits grade
strikeouts grade
outcome ingestion fallback rules 8) src/api/

Current API folder contains app.py and static/ with index.html, totals.html, results.html, hot-hitters.html, pitchers.html, and favicon.svg. app.py`
Add or cleanly separate endpoints for:

GET /api/games/board
GET /api/games/{game_id}
GET /api/history/slate
GET /api/recommendations/history
GET /api/jobs/{job_id}
POST /api/jobs/prepare-slate
POST /api/jobs/refresh-markets
POST /api/jobs/build-predictions
POST /api/jobs/grade-predictions
POST /api/jobs/retrain-models

Right now the README documents status, predictions, trends, scorecards, and one pipeline endpoint. Expand that into proper job-oriented routes because your app buttons already imply that workflow. static/index.html`
Refactor into:

compact slate board
freshness/confidence badges
update center button group
active job status panel
src/api/static/results.html

Turn this into the real historical-review page:

predicted vs actual
market line vs model line
hit/miss markers
recent model scorecards
src/api/static/hot-hitters.html

Add:

rolling hit form
streak context
confirmed/projected lineup marker
link back to game detail
src/api/static/pitchers.html

Add:

probable starters
K model output
recent form
opponent K tendency
workload/leash context
src/api/static/totals.html

Keep:

totals lane
first-five lane
edge and confidence
line freeze timestamp
New static pages
game.html for detailed game view
jobs.html for run history and failures
settings.html for local app/source config 9) src/desktop/

Current desktop entrypoint is launcher.py. top/launcher.py`
Add:

first-run bootstrap screen
migration progress
DB doctor check
graceful fallback if API startup fails
runtime path display for logs/db/exports
app version + release channel display

Definition of done:

fresh Windows machine can install and launch without touching Python or Postgres,
launcher shows useful recovery info when data/bootstrap fails. 10) scripts/ and installer/windows/

Current scripts are build_windows_app.py, build_windows_installer.py, and build_windows_release.py. Installer assets include MLBPredictor.iss plus install/uninstall .cmd and .ps1 files. build_windows_app.py`
Add:

verify bundled models exist
verify migrations folder exists
verify seed files exist
write build manifest JSON
scripts/build_windows_installer.py

Add:

checksums file generation
stable/beta channel labeling
release-notes injection
runtime dir preservation guarantees
scripts/build_windows_release.py

Add:

one command to produce:
installer
portable zip
checksums
manifest
release notes draft
installer/windows/MLBPredictor.iss

Add:

version metadata from build manifest
nicer upgrade path text
preserve user runtime data
optional desktop shortcut checkbox
install_mlb_predictor.cmd / .ps1

Add:

friendlier logging
failure messaging
post-install launch option
uninstall_mlb_predictor.cmd / .ps1

Add:

explicit prompt to keep or remove user data under %LOCALAPPDATA%\MLBPredictor 11) tests/

You already have a strong start here: API SQL helpers, update jobs, desktop launcher, DB compat, migrations, feature contracts, lineups, market totals, priors, product surfaces, packaging, and park-factor bootstrap tests. # New tests

test_ingestion_audit.py
test_market_snapshot_freeze.py
test_weather_forecast_vs_observed.py
test_prediction_grading.py
test_sqlite_pipeline_end_to_end.py
test_release_manifest.py
Expand existing tests
test_api_update_jobs.py: cover all new job endpoints
test_db_compat.py: freeze remaining SQLite/Postgres parity issues
test_product_surfaces.py: cover board/detail/history payloads
test_build_windows_installer.py: validate checksums + manifest + runtime preservation rules 12) Exact implementation order
Slice 1 — make the app truly standalone
config/.env.example
src/utils/settings.py
src/utils/db.py
src/utils/db_migrate.py
src/desktop/launcher.py
tests/test_db_compat.py
tests/test_desktop_launcher.py
Slice 2 — make ingestion trustworthy
db/migrations/007_ingestion_audit.sql
db/migrations/008_market_snapshot_versioning.sql
src/ingestors/common.py
src/ingestors/market_totals.py
src/ingestors/weather.py
src/ingestors/lineups.py
src/ingestors/validator.py
tests/test_market_totals.py
tests/test_ingestion_audit.py
Slice 3 — make recent outcomes great
db/migrations/010_prediction_grading.sql
src/transforms/grading.py
src/transforms/product_surfaces.py
src/api/app.py
src/api/static/results.html
tests/test_product_surfaces.py
tests/test_prediction_grading.py
Slice 4 — polish the downloadable release
scripts/build_windows_app.py
scripts/build_windows_installer.py
scripts/build_windows_release.py
installer/windows/MLBPredictor.iss
install/uninstall scripts
tests/test_build_windows_installer.py
tests/test_release_manifest.py 13) What “v1 ready for other users” means

You’re ready to hand this to outside users when all of these are true:

install from release asset works,
app boots with SQLite only,
“Prepare Slate” works,
“Build Predictions” works,
“Grade Recent Predictions” works,
every game shows freshness/confidence,
recent outcomes page is trustworthy,
upgrading the app does not wipe user data.
That target matches the repo’s existing separation between in-app daily updates and manual GitHub Release app updates. is into a GitHub-issues style backlog next, with issue titles, acceptance criteria, and recommended order by milestone.
