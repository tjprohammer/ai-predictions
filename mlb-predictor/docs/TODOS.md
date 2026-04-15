# Engineering backlog (living list)

Prioritized follow-ups from product review and API work (April 2026).

## Near term

1. **HTTP routes modularization** — Done: `http_routes.py` aggregates domain routers (`meta_routes`, `html_routes`, `api_feed_routes`, `games_routes`, `jobs_routes`, `ops_routes`). Add a new sub-router file and `include_router` when a domain grows.
2. **NRFI/YRFI visibility** — Lines come from `game_markets` (`nrfi` / `yrfi`) when Odds returns `totals_1st_1_innings` on the player-prop/first-five pull. **`/api/doctor`** exposes `experimental_markets` + check. **Ingest** now **logs** counts of matched events that include `totals_1st_1_innings` vs missing, with sample matchups when some are missing (`market_totals._fetch_odds_api_player_prop_rows`). **Manual override:** same `MANUAL_MARKETS_CSV` schema as other game markets—rows with `market_type` `nrfi` or `yrfi` and valid prices upsert into `game_markets`.
3. **README architecture** — **Done:** “API application layout” subsection in the root README.

## Medium term

4. **Certainty vs projection (product/model)** — **Shipped (April 2026):** input trust grades, `promotion_tier` on best bets, history buckets + monotonicity on `GET /api/recommendations/best-bets-history`. See [CALIBRATION_AND_PRODUCT.md](CALIBRATION_AND_PRODUCT.md). Ongoing model work remains in [MODEL_REWORK_PLAN.md](MODEL_REWORK_PLAN.md) Phases 3–4.
5. **Totals narrative vs model** — **Shipped:** `totalsReasonList` prepends model lane / confidence / suppress + input trust; heuristic bullets documented as supporting only in [TOTALS_NARRATIVE.md](TOTALS_NARRATIVE.md).

## How to use this file

Update or archive rows as they ship. Link PRs in commit messages rather than duplicating long diffs here.
