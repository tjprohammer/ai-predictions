# Main board, Daily Results, and how the pieces fit together

This document is the source of truth for **what is a separate model / surface** vs what is “team best bets,” and **where each payload comes from**. It exists so UI and API changes do not mix unrelated lanes.

## Separate products (do not conflate)

| Surface | Model / inputs | API / UI | Notes |
|--------|----------------|----------|--------|
| **Green strip (“best picks”)** | Team/game markets from `build_market_cards_for_game` + `flatten_best_bets` | `GET /api/games/board` → `best_bets` (flattened live). Dashboard: `state.greenPicks`. | **Only** `BEST_BET_MARKET_KEYS` (ML, spreads, totals, F5 team markets, etc.). **Not** hitter props. |
| **Watchlist (main board)** | Same board rows; `flatten_watchlist_markets(..., secondary_lines_only=False)` | `watchlist_markets` on the same board response. | Shows games **without** a green-strip pick for that slate (overflow / “next” games). **Not** the same list as Daily Results watchlist. |
| **Slugger / HR strip** | HR model + slugger selection (`_fetch_slugger_hr_bets`) | `slugger_hr_bets` on the board response. Own panel in `index.html`. | **Separate** from green/watchlist. Never use hitter `min_probability` for this strip. **Model quality:** see `MODEL_REWORK_HR_AND_EXPERIMENTAL.md`. |
| **NRFI / YRFI (experimental)** | `predictions_inning1_nrfi` + market rows; `_fetch_experimental_market_cards` | `experimental_markets` on the board. | **Not** on the green strip or team watchlist. Tracked under experimental in Daily Results. **Model quality:** see `MODEL_REWORK_HR_AND_EXPERIMENTAL.md`. |
| **1+ Hits / hitter cards** | Hits model, `hit_targets`, game detail | Per-game hitter payloads, hot-hitters page. | **Not** team “best bets.” |

## Daily Results page (`results.html`)

- **API:** `GET /api/results/daily?target_date=…` → `app_logic._fetch_daily_results`.
- **Green / watchlist rows** come from `prediction_outcomes_daily` via `_fetch_game_recommendation_results`, **plus** live mirrors from the board when archives are empty (`_live_green_board_rows_for_daily_results`, `_live_watchlist_board_rows_for_daily_results`).
- **Watchlist (Daily Results)** uses `flatten_watchlist_markets(..., secondary_lines_only=True)` so **secondary** team markets (not the same card as the green strip) still appear for grading when every game already has a green pick.

## `prediction_outcomes_daily`: two market-string families

The table is filled from more than one builder:

1. **`_build_prediction_outcomes`** (totals model lane): writes `market = 'totals'` or `'first5'` (legacy strings).
2. **`_build_best_bet_outcomes`** (board snapshot): writes `market` = **`BEST_BET_MARKET_KEYS`** style (`game_total`, `moneyline`, `first_five_total`, …).

**TRACKED_RECOMMENDATION_MARKETS** and the **best-bet history carousel** query must include **both** naming styles where we still want rows to appear. If you only filter on `BEST_BET_MARKET_KEYS`, **`totals` / `first5` rows disappear** from history and from Daily Results buckets even though the data exists.

## Best Bet History carousel (main dashboard)

- **API:** `GET /api/recommendations/best-bets-history` → `_fetch_best_bet_history_payload`.
- Reads `prediction_outcomes_daily` with `entity_type = 'game'` and **game-level markets**, including legacy `totals` and `first5` plus all `BEST_BET_MARKET_KEYS`.
- **UI filter:** `bestBetMarketGroup` maps legacy `totals` → `game_total` and `first5` → First 5 group so filters match.

## Why “Best Bets / green strip” can be empty while Watchlist is full

They are **different rules**:

- **Green strip** (`flatten_best_bets` → `best_bets` on the API): a market must be **strict positive** (`positive`) **or** pass **soft green** gates (`qualifies_board_green_strip`: weighted EV, prob edge, certainty, model prob — see `BOARD_GREEN_SOFT_*` in `best_bets.py`). On thin-trust slates, **nothing may qualify**, so `best_bets` is `[]`.
- **Watchlist** (`flatten_watchlist_markets` with `secondary_lines_only=False`): fills with **next-best team markets** for games that **do not** have a green-strip pick, using looser `_is_watchlist_candidate` rules. You can easily have **dozens of watchlist cards** and **zero** green strip cards.

The dashboard **always shows** the Best Bets panel; when the list is empty, the copy explains this (it is not a missing API field).

## Future slates (tomorrow or any date after today)

This is **normal operating mode** for pre-first-pitch work:

- **No final scores** and **no graded** Daily Results until those games finish.
- **Green strip** may stay empty: soft-green gates depend on weighted EV, certainty, and posted markets; **tomorrow’s** slate often has **thinner trust** or **incomplete lines** until ingest catches up near lock.
- **Watchlist** can still be large: it uses **looser** watchlist rules for team markets on games that don’t place a pick on the green strip.
- **HR / experimental** are separate models; rerun **`predict_hr`** / inning-1 predict for that **`game_date`** after features update.

The main dashboard detects a **future** `target_date` (local calendar) and adjusts the empty-state copy accordingly.

## HR strip: same faces or tiny P(HR)

- **Same players day to day:** slugger selection is **ranked** from `predictions_player_hr` + `iter_slugger_tracked_cards` (per-game cap, then global cap). Until the slate or model output changes a lot, the **top** names can repeat.
- **Absurdly small P(HR) (e.g. 0.0001%):** check **`predictions_player_hr.predicted_hr_probability`** — if values are near **1e‑6**, the **model or feature pipeline** for that date is wrong or stale. Re-run **`hr_builder`** and **`predict_hr`** for the target date after lineups/features update. The UI prefers **`hr_probability_display`** from the API and uses **`cache: 'no-store'`** on the board fetch to reduce stale browser caching.

## Empty slate checklist (e.g. “nothing for 4/16”)

1. **Games:** `games` has rows for that `game_date`?
2. **Predictions:** `predictions_totals` / markets run for that date?
3. **Board:** `GET /api/games/board?target_date=…` returns non-empty `games` and non-empty `best_bets` when edges exist?
4. **Outcomes:** After games finish, **`product_surfaces`** (date range including that day) populates `prediction_outcomes_daily` for archived green/history?
5. **Not a bug:** Past **future** dates have no boxscores; pre-first-pitch days may have predictions but no grades.

## Files (quick reference)

| Area | Primary code |
|------|----------------|
| Flatten green / watchlist | `src/utils/best_bets.py` |
| Board payload | `src/api/routers/games_routes.py`, `app_logic._fetch_game_board` |
| Daily Results payload | `app_logic._fetch_daily_results` |
| Outcomes writers | `src/transforms/product_surfaces.py` |
| Best bet history | `app_logic._fetch_best_bet_history_payload` |
| Dashboard board UI | `src/api/static/index.html` |
| Daily Results UI | `src/api/static/results.html` |
