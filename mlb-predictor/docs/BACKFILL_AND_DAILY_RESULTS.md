# Backfill notes and Daily Results

## What we backfill (CLI)

Two layers:

1. **Facts / ingest** — schedule, finals, batting, lineups, observed weather, team aggregates  
   Typical one-shot window (opening day → today):

   - `python -m src.ingestors.games --start-date YYYY-MM-DD --end-date YYYY-MM-DD`
   - `python -m src.ingestors.boxscores --start-date … --end-date …`
   - `python -m src.ingestors.player_batting --start-date … --end-date …`
   - `python -m src.ingestors.lineups_backfill --start-date … --end-date …`
   - `python -m src.ingestors.weather --start-date … --end-date … --mode observed`
   - `python -m src.transforms.offense_daily --start-date … --end-date …`
   - `python -m src.transforms.bullpens_daily --start-date … --end-date …`

   **Optional / slow:** `src.ingestors.matchup_splits` over the same range (BvP / platoon cache only; not required for grading surfaces).

2. **Product surfaces** — rebuilds derived tables from data already in the DB (does not re-run model training):

   - `python -m src.transforms.product_surfaces --start-date YYYY-MM-DD --end-date YYYY-MM-DD`

   Populates/rebuilds including `prediction_outcomes_daily`, recommendation history, scorecards, calibration bins, trend tables.

**Update Center** runs `product_surfaces` with `--target-date` only (single day). Season-long history needs the **date-range** CLI above.

## Where Daily Results come from (UI)

- **Page:** `src/api/static/results.html`
- **API:** `GET /api/results/daily?target_date=…` → `app_logic._fetch_daily_results`
- **Board-style buckets** (green / watchlist / experimental for *archived* recommendation rows) read from `prediction_outcomes_daily` via `_fetch_game_recommendation_results`, using `meta_payload` flags set when `product_surfaces` runs `_build_best_bet_outcomes` (green/watchlist ranks from `best_bets.snapshot_recommendation_tiers`).

**Watchlist can legitimately be empty** in archived recommendation results: if no row that day has `is_board_watchlist_pick` in meta *and* snapshot flags are missing, the watchlist list stays empty (watchlist does not use the green fallback). **NRFI/YRFI are excluded** from green/watchlist recommendation rows (`_fetch_game_recommendation_results`); they only appear under the experimental bucket or the experimental carousel.

**Live board** `best_bets` / `watchlist_markets` (from `GET /api/games/board`) come from `_fetch_game_board` + `flatten_*` — separate from archived `green_picks` in the same payload.

**Dashboard “Green picks” strip** in `index.html` uses **`best_bets`** (live positive-EV cards). **`green_picks`** remains in the API as **archived** graded rows for tooling; it is stored as `archivedGreenPicks` in the client if you wire it later.

**Experimental** rows are those tagged experimental in meta or markets in `EXPERIMENTAL_CAROUSEL_MARKETS` (`nrfi`, `yrfi`) when present in tracked outcomes.

**Main slate tables** on the same page (totals, hitters, strikeouts, first-five) come from live prediction tables (`predictions_totals`, etc.) joined to `games`, not only from `prediction_outcomes_daily`.

**“Missing box score rows” (hitters):** a row counts as missing when the game status looks **final** but there is **no** matching `(game_id, player_id)` in `player_game_batting` (so `hits` is unknown). List gaps and per-date counts:

`python scripts/audit_missing_hitter_boxscores.py --date YYYY-MM-DD`  
`python scripts/audit_missing_hitter_boxscores.py --start-date … --end-date …`

Then re-run **`src.ingestors.player_batting`** (and **`boxscores`** if needed) for those `game_date`s.

## NRFI / YRFI — one row per game

Experimental first-inning lines use **`game_markets`** rows typed `nrfi` and `yrfi`. The app and `product_surfaces` **keep at most one** of these **per `game_id`**: **NRFI wins** when both exist (see `dedupe_experimental_first_inning_by_game` in `src/utils/best_bets.py`). Re-run **`product_surfaces`** after changing this logic so `prediction_outcomes_daily` matches.

## Suggested order: retrain vs “refresh everything”

1. **Finish ingest** (from `mlb-predictor/`): `games` → `boxscores` → `player_batting` → … for the dates you care about.  
2. **Optional retrain** (better *future* predictions): `python -m src.models.train_totals` (and other `train_*` modules as you use them). Training uses whatever features/outcomes are in the DB — it does not replace step 1.  
3. **Optional re-predict past slates** (only if you want old dates rescored with new models): `predict_*` per date or range — heavy.  
4. **Publish surfaces**: `python -m src.transforms.product_surfaces --start-date … --end-date …` so outcomes, scorecards, and experimental rows align with current predictions + markets + finals.  
5. **App**: run the API, use **Update Center** for day-to-day (single date) or rely on CLI for **range** backfills. Open **Daily Results** / board for a given `target_date`.

**Rule of thumb:** retrain when improving **models**; run **ingest + `product_surfaces`** when improving **graded history and UI**.

## Retraining vs backfill

- **Retraining** changes model weights for **future** scoring. It does **not** by itself fill Daily Results for past dates.
- To refresh **historical** displays you need stored predictions + markets + finals, then **`product_surfaces`** (and ingest if scores/rows were missing).
- Re-running **`predict_*` for past dates** after a retrain *can* change stored predictions for those days; then run **`product_surfaces`** again for that range if you want outcomes and recommendation meta aligned to the new model scores.

**Practical recommendation:** run **training** when you want better forward predictions, not as a substitute for `product_surfaces`. Use **range `product_surfaces`** (and ingest) to backfill **display/grading** for dates that already have predictions in the DB.
