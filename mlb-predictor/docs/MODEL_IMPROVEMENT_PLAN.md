# Model & data improvement plan

Track work to strengthen **NRFI**, **totals / first-five**, **evaluation**, and **board** behavior. Related narrative for totals-only strategy: [TOTALS_IMPROVEMENT_PLAN.md](./TOTALS_IMPROVEMENT_PLAN.md).

**How to use:** Check boxes as you complete items. Add a one-line note under **Progress log** with the date.

---

## Phase 0 — Baseline audit (before / after metrics)

Re-run anytime: `python scripts/audit_model_improvement_plan.py` (optional `--start-date` / `--end-date`). Makefile: `make audit-model-plan` (uses default 540-day window ending yesterday).

- [x] **Inning‑1 labels (baseline snapshot 2026-04-20):** In the last **540** days, **2 681** final games but only **146** with `total_runs_inning1` — **2 535** need boxscore backfill before NRFI training is viable.
- [ ] **Boxscore coverage:** In progress — `make backfill-inning1 START_DATE=... END_DATE=...` (or `python -m src.ingestors.boxscores ... --final-missing-inning1-only`). Re-run the audit script after; expect `with_inning1` near final game count (minus rare incomplete linescores).
- [x] **F5 `market_total` density (baseline):** `first5_totals` parquets ~**4.1%** non-null `market_total`; `game_markets` had only **139** distinct games with `first_five_total` lines in-window — historical odds backfill + Odds API usage is the bottleneck (not the cutoff logic).
- [x] **Totals sanity:** `audit_model_improvement_plan.py` prints the latest `data/reports/totals/*.json` fundamentals MAE, team-average baseline, post-calibration MAE, and `market_shrink` when present (retrain totals to refresh numbers).

---

## Phase 1 — Data bottleneck

### 1A. NRFI / `total_runs_inning1`

- [ ] Backfill **boxscores** for all final games in the training window (use `--final-missing-inning1-only` once DB has been audited).
- [ ] Rebuild **`first5_totals`** features (if F5 rows change), then **`inning1_nrfi`** builder (`src/features/inning1_nrfi_builder.py`).
- [ ] Re-run **`train_inning1_nrfi`** after labeled count is well above minimum.
- [ ] *(Optional)* Use CLI `--min-labeled-rows` when you intentionally experiment below production threshold.

### 1B. Totals / first‑five — market coverage & alignment

- [x] **Pregame cutoff for completed games:** `default_cutoff()` uses **first pitch** (`game_start_ts`) when `games.status == final`, so historical feature rebuilds do not take `now` as cutoff (avoids post-game / lookahead snapshots). See **Progress log**.
- [ ] **Historical odds coverage:** Backfill `game_markets` for `total` and `first_five_total` on train dates (where missing). Range helper: `make backfill-markets-range START_DATE=... END_DATE=...` (API keys / quota).
- [ ] **Single-book policy (optional):** Prefer board book or consensus rule consistently in feature builders and baselines.

---

## Phase 2 — Totals lane (market prior + objectives)

- [x] **Market-anchored layer:** Existing `fit_market_calibrator` Ridge blend on validation; `train_totals` mirrors full predict path (fallback market anchor) for `post_calibration_*` metrics.
- [x] **Secondary metrics:** Validation report includes median pinball loss (τ=0.5) and O/U log loss vs posted total (`validation_metrics` in totals report).
- [x] **Weak-model shrink:** When raw fundamentals MAE exceeds the best baseline on validation, `train_totals` grid-searches a convex `market_shrink` w toward `market_total` (after calibration); `predict_totals` applies it. Diagnostics in `market_shrink_diagnostics`.

---

## Phase 3 — First‑five as its own problem

- [ ] Add **F5-specific** pregame context (bullpen rest / workload, opener if detectable, etc.) in `first5_totals` builder.
- [x] Allow **pregame F5 market** as model input (not only post-hoc calibration): `market_total` is `FIELD_ROLE_CORE_PREDICTOR` for `first5_totals` in `contracts.py` — **retrain** `train_first5_totals` after rebuilding features.
- [x] Keep **`lane_status`** in artifact honest; use in product (Phase 5).

---

## Phase 4 — Evaluation discipline

- [ ] **Rolling / walk-forward CV** in train scripts (multiple time folds; mean ± std of metrics).
- [ ] **Stratified reports:** Calendar-month validation MAE/RMSE is in reports for **totals** and **first5_totals**; stratification by **slate size** is still open.
- [ ] **Calibration bins / reliability** per lane in reports (extend beyond hits isotonic story).

---

## Phase 5 — Product / board

- [x] Gate **first-five** when `lane_status == below_baseline`: `lane_status` stored on `predictions_first5_totals` (migration `031`), set at predict from artifact; board payload clears **recommended_side**; **green / watchlist / Top EV** skip F5 cards via `lane_research_only` on market cards (`best_bets.py`, `top_ev_pick.py`).
- [x] UI / copy: dashboard hero + Live Board copy emphasizes **hits, HR, Ks**; totals / F5 framed as supporting context.
- [x] **Board action score:** Time-ordered holdout ROC-AUC logged (`train_board_action_score.py --holdout-fraction`, default 0.2); final artifact still fit on all rows for sample size.

---

## Progress log

| Date       | What changed |
| ---------- | ------------ |
| 2026-04-20 | Added this doc. Fixed `default_cutoff(..., game_status=...)` so **final** games use `game_start_ts` (pregame) instead of `now`; wired `status` into feature builders that were missing it. Added `train_inning1_nrfi --min-labeled-rows`. |
| 2026-04-20 | Phase 0 audit script `scripts/audit_model_improvement_plan.py` + `make audit-model-plan`. Boxscores flag `--final-missing-inning1-only`. `pipeline_range` extended for inning1 + total_bases. Started inning‑1 boxscore backfill for 2024-10-25..2026-04-18 (long run — re-audit when complete). |
| 2026-04-20 | Phase 5: F5 `below_baseline` gating + migration `031` + predict copies `lane_status`; board action score time-holdout AUC logging. **Run `db-migrate`**, then **`predict_first5_totals`** so new column is populated. |
| 2026-04-16 | Phase 2 totals: validation pinball + O/U log loss, monthly/temporal-half MAE, `market_shrink` after calibration when fundamentals lose to baselines; audit script reads latest totals report. Phase 3: F5 `market_total` as core predictor in contracts (retrain F5). Makefile `backfill-inning1` / `backfill-markets-range`. UI copy emphasizes hits/HR/Ks. |

---

## Run order for “retrain then predict tomorrow”

1. **Audit:** `python scripts/audit_model_improvement_plan.py`
2. **Ingest:** `python -m src.ingestors.boxscores --start-date ... --end-date ... --final-missing-inning1-only` → re-audit.
3. **Markets:** `python -m src.ingestors.market_totals --start-date ... --end-date ...` (needs API keys / quota) to widen `total` + `first_five_total` in `game_markets`.
4. **Features:** `src.utils.pipeline_range` (or per-lane builders) for **totals**, **first5_totals**, **inning1_nrfi**, hits, HR, total_bases, strikeouts.
5. **Train:** `make retrain-models` (or per-lane).
6. **Predict + surfaces:** predict for slate date + `product_surfaces`.

`pipeline_range` now includes **inning1_nrfi_builder**, **total_bases_builder**, **predict_inning1_nrfi**, and **predict_total_bases** so range backfills match `Makefile` `features-today` / predict coverage.
