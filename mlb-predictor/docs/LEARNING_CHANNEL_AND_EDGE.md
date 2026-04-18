# Learning channel, tracked picks, and edge vs the market

This document aligns **product intent** with **what the codebase actually implements**: you are not wrong to want **best picks** and **Top EV** to feed a **feedback loop**. The system is designed for that—but **learning** is not “the API retrains every night automatically.” It is a **pipeline**: observe → store → measure → (human or job) **decide** → retrain / recalibrate / tune thresholds → deploy.

---

## 1. What “learning channel” means here

| Layer | Role |
|--------|------|
| **Live picks** | Today’s board: model + features + lines → **best bets** (`flatten_best_bets`) and **Top EV** (`top_ev_pick`). New data every day changes **inputs**; weights change only when you **train/deploy** a new model. |
| **Observation** | After games, we **grade** what we recommended vs results and vs the market (where modeled). |
| **Feedback** | Stored rows drive **scorecards**, **calibration bins**, **recommendation history**, and any **offline** analysis you run. |
| **Edge** | Sustainable edge requires **calibrated** probabilities, **stable** labels for what we “said,” and **periodic** model/threshold updates—not one hidden step inside `GET /api/games/board`. |

**Snapshots** (`run` / `lock` on green and Top EV) exist so the **observation layer** grades the **same** headline pick users saw, not a recomputed pick after lineups or lines moved. That is **essential** for a trustworthy learning signal.

---

## 2. What is already implemented (this repo)

### 2.1 `prediction_outcomes_daily` — central outcome table

Built by `src/transforms/product_surfaces.py` in `_build_prediction_outcomes` (and helpers it calls). For each date range it **deletes and rebuilds** rows for that window, then upserts. Rows include (where applicable):

- **Model identity:** `model_name`, `model_version`, `prediction_ts`
- **Grading:** `graded`, `success`, `actual_side`, `actual_value`, Brier / squared error
- **vs market:** `beat_market`, CLV fields, closing line comparisons for supported markets

**Board product picks wired into outcomes:**

- **`_build_best_bet_outcomes`** — team **best-bet** cards (green/watchlist-style markets from `BEST_BET_MARKET_KEYS`), graded vs boxscore / first-five / props as implemented.
- **`_build_top_ev_outcomes`** — **Top EV** pick per game (uses frozen snapshots when present: lock → run → live), same grading primitives.

Also in the same build: totals lane, experimental markets, hits, HR, pitcher strikeouts, etc. (full list in `product_surfaces.py`).

### 2.2 Downstream of outcomes

From `product_surfaces.main()`:

- **`model_scorecards_daily`** — aggregates from `prediction_outcomes_daily` (win rates, graded counts, CLV summaries by market/model/version).
- **`_build_calibration_bins`** — calibration diagnostics from outcomes.
- **`recommendation_history`** — recommendation-style history table for the product.
- **Player / pitcher trends** — feature-style trend tables for surfaces.

### 2.3 When this runs

- **CLI / pipeline:** `python -m src.transforms.product_surfaces --start-date … --end-date …`
- **Update jobs:** `src/api/update_job_sequences.py` and `src/utils/pipeline_range.py` include `product_surfaces` so outcomes can be refreshed after scores exist.
- **Daily Results UI (`src/api/static/results.html`):**
  - **Update Scores** (`refresh_results`) pulls boxscores and related ingest, then runs `product_surfaces` **for the full results backfill window** (same multi-day range as ingest—not only the single selected date), so `prediction_outcomes_daily` and scorecards stay aligned with recent finals.
  - **Rebuild learning tables** (`rebuild_learning_tables`) runs only `product_surfaces` over that same window—use after scores are current when you want to refresh outcomes/scorecards without re-ingesting (fast nightly option).
  - **Retrain models** (`retrain_models`) runs the full training pipeline plus prediction refresh for the selected date; long-running—confirm in the browser.

### 2.4 Model training vs outcomes (important distinction)

- **Training scripts** (`train_*`, feature builders, backtests) use **historical seasons**, rolling windows, and **held-out** evaluation—not a live “yesterday’s row updates weights tonight” loop inside the API.
- **`src/models/common.py`** includes **market calibration** helpers (`fit_market_calibrator`, `calibrate_with_market`) used where the training/predict stack plugs them in—**fitting** a calibrator is an **offline** step with enough labeled rows.

So: **tracked picks in `prediction_outcomes_daily` are the right fuel for learning**; **automatic weight updates** are **not** implied by the dashboard alone—they come from **jobs you schedule** (retrain, recalibrate, tune gates).

---

## 3. Why this matches “especially for best picks and EV picks”

- **Best bets** and **Top EV** are the **product-facing** choices users act on. Their outcomes **are** written into `prediction_outcomes_daily` (via `_build_best_bet_outcomes` and `_build_top_ev_outcomes`).
- **Snapshots** ensure those rows correspond to **stable** recommendations when you run `product_surfaces` after the slate—fixing the “daily update scrambled my label” failure mode that **poisons** the learning signal.

So the **intent you described** is correct: **track → measure → improve**. The implementation is **observation + analytics tables + pipelines**; **closing the loop** is operational (retrain cadence, calibration refresh, threshold reviews using scorecards and bins).

---

## 4. Roadmap: tightening the loop (suggested, not all implemented)

Concrete ways to **use** this channel for edge (incremental):

1. **Weekly or monthly** review: `model_scorecards_daily` and calibration bins by `market` and `model_version`.
2. **Retrain** totals / props / strikeouts models on schedule; bump `model_version` when deploy changes.
3. **Recalibrate** using recent `prediction_outcomes_daily` slices where `graded` is true (respect chronology; avoid leakage).
4. **Tune** `BOARD_GREEN_*`, `BEST_BET_THRESHOLD_MAP`, and Top EV candidate rules using **out-of-time** backtests plus live outcome stats.
5. **Backfill** bad historical days after snapshot fixes so training/eval sets align with what was actually shown.

---

## 5. North star: calibrated “action” score (product ML)

This is the machine-learning direction that matches **“best picks for a reason”**: a **dedicated model or calibrator** whose job is **whether to act** on a candidate card—not replacing every underlying totals/hits model, but **ranking and gating** promotions using **outcomes** as supervision.

**Idea**

- **Target:** Labels from **`prediction_outcomes_daily`** (and stable snapshots): graded **win/loss/push**, Brier-friendly probabilities, optional **CLV** or closing-line features as auxiliary signal—not leakage from future lines you didn’t have at entry.
- **Inputs (examples):** Per-card **model probability**, **no-vig / market** probability, **weighted EV**, **input trust**, **game certainty**, market family, line type—everything you already compute in `build_market_cards_for_game` / `flatten_best_bets`.
- **Output:** A **calibrated action score**—e.g. \(\mathbb{P}(\text{win} \mid \text{bet})\) after isotonic/Platt, or a **single “expected value of acting”** score trained with a proper scoring rule on historical rows.
- **Training:** **Chronological splits** (train on past seasons / past months, validate forward); avoid random shuffles across time. Retrain on a schedule when enough new graded rows exist.
- **Deployment:** New field on cards (e.g. `action_score` / `action_tier`) consumed by **green / watchlist** rules—initially **alongside** existing gates, then gradually **replacing** hand-tuned thresholds where the score proves better in backtest and live scorecards.

**Why it’s different from today’s gates**

- Today, **best picks** = **rules on top of** base model outputs. An **action score** = **learned mapping from (features → historical outcomes)** for the **decision** itself, explicitly fit so **calibration and ranking** improve on your tracked results.

**Prerequisites (already in motion)**

- Honest labels: **snapshots** + **`prediction_outcomes_daily`** for board picks.
- Ops: **Rebuild learning tables** / `product_surfaces` after finals so rows exist for training.

### 5.1 Implemented (v1): board action score

- **Train:** `python -m src.models.train_board_action_score` (or it runs automatically at the **end** of **`Retrain models`** / `src.models.retrain_models`). Needs enough graded rows in **`prediction_outcomes_daily`** for `BEST_BET_MARKET_KEYS` (default minimum **80**; configurable `--min-rows`). Writes `MODEL_DIR/board_action_score/action_classifier.joblib` + `action_classifier_meta.json`.
- **Model:** `StandardScaler` + **logistic regression** (`class_weight=balanced`) on 9 features aligned between historical outcomes and live cards (model prob, no-vig, loss prob, weighted EV, prob edge, trust score, certainty, strict-positive flag, normalized market id).
- **Serve:** `annotate_market_card_for_display` attaches **`action_score`** (0–1, estimated P(win)) and **`action_score_model`** when the artifact loads. **Top EV** is unchanged (separate selector).
- **UI:** Main board green-strip chips show **Act %** when present; watchlist cards show **Action %** in the meta line.

Improve later: chronological CV, isotonic calibration, richer meta (e.g. certainty in outcomes), and using `action_score` inside promotion gates—not only display.

---

## 6. File reference

| Piece | Location |
|--------|----------|
| Action score train / infer | `src/models/train_board_action_score.py`, `src/models/board_action_score.py` |
| Outcome build (best bets + Top EV + …) | `src/transforms/product_surfaces.py` |
| Snapshots (stable labels) | `docs/BOARD_BEST_BETS_AND_SNAPSHOTS.md`, `src/api/app_logic.py` |
| Pipeline wiring | `src/utils/pipeline_range.py`, `src/api/update_job_sequences.py` |
| Market calibration primitives | `src/models/common.py` |
| Board / Daily Results product map | `docs/BOARD_AND_DAILY_RESULTS_PRODUCT.md` |

---

## 7. One-sentence summary

**Yes:** we implemented tracking **precisely** so **best picks** and **Top EV** can feed measurement and, downstream, **learning**—with snapshots keeping labels honest; **no:** the live API does not silently retrain every night; **edge** comes from **using** `prediction_outcomes_daily` and scorecards in disciplined **offline** updates and deployments. A **first action-score model** (§5.1) trains on those outcomes and surfaces **`action_score`** on team best-bet cards.
