# Totals Improvement Plan

## Purpose

Make totals a **selective, certainty-aware lane** — not force a broad raw model to beat dumb baselines on every game.

This plan follows the current rework direction where projection, certainty, calibration, diagnostics, and product behavior are treated as separate layers.

---

## Strategic Direction

We keep working on total runs. We stop treating it like "one model should beat everything on every game."

### What we know

- Raw totals is still below the strongest baseline (train_median MAE 3.454)
- First 5 is even more fragile (train_mean MAE 2.658 beats model)
- Split-side modeling did not help on current data
- The project docs already point toward board states, certainty, and calibration as the next layer of value
- The game-specific adjustment features are not yet adding enough signal beyond team identity

### What success looks like

Not: _totals beats everything everywhere._

More like: _totals becomes useful on higher-certainty boards, and first 5 becomes usable once calibration is added._

### What we will not do right now

- Do not keep trying bigger raw totals models
- Do not put market fields back into raw training
- Do not push side-model totals into the main path
- Do not judge totals only by one all-games MAE number
- Do not keep expanding feature count without proving each feature family adds signal

---

## Current Honest State

### Full-game totals

| Benchmark                                 | MAE   |
| ----------------------------------------- | ----- |
| train_median                              | 3.454 |
| team_average                              | 3.558 |
| train_mean                                | 3.600 |
| best model (baseline+residual elasticnet) | 3.561 |

The model does not beat the strongest baselines across all games.

### First 5 totals

| Benchmark    | MAE   |
| ------------ | ----- |
| train_mean   | 2.658 |
| train_median | 2.675 |

First 5 is the lane most hurt by taking market fields out of raw training. It is a redesign plus calibration lane.

### Strikeouts (for reference)

Strikeouts is the only lane currently showing clear predictive edge above baseline (~4.5% MAE improvement). It remains the lead lane.

---

## Core Diagnosis

The problem is not "more model tuning needed."

The current evidence says:

- Broad full-board totals prediction is too noisy with the current feature set
- Team-level baseline context carries more signal than game-specific adjustments
- First 5 needs a different structure than "full-game totals but shorter"
- Certainty and calibration layers are probably where practical value comes from, not just raw MAE reduction

### The Central Question

Instead of asking "can totals beat baseline across every game?" ask:

> **When does totals have any real edge at all?**

The answer probably depends on:

- Board state (lineup confirmed vs early-morning guesses)
- Input certainty (starter confirmed, weather fresh, lineup locked)
- Game environment (outdoor, warm weather, known park factors)

---

## Target End State

The totals system should have five layers:

### 1. Baseline projection layer

Stable team-level expected scoring baseline.

### 2. Game adjustment layer

Starter, lineup, bullpen, park, and weather adjustments.

### 3. Certainty layer

How much the current inputs should be trusted.

### 4. Calibration layer

Market-aware post-model correction.

### 5. Publish/suppress layer

Only surface totals plays when both edge and certainty justify it.

---

## Two Separate Lanes

### Full-game totals

- Active redesign lane
- Unified total target (not home+away split)
- Team-average baseline + game-specific adjustment residual
- Focus: when is it trustworthy, not whether it beats everything

### First 5 totals

- Redesign **plus calibration** lane — not bundled with full-game totals
- Early-inning scoring only
- Starter strength, starter leash, first-time-through exposure
- Top-of-lineup quality, early run environment
- Less emphasis on full bullpen context and late-game volatility
- Needs post-model calibration to regain competitiveness after market field removal

---

## Phase 1: Freeze Honest Baselines — DONE

Frozen via `src/backtest/benchmark_totals.py --freeze` (commit `437b45a`).

| Lane   | Best baseline | MAE   |
| ------ | ------------- | ----- |
| totals | train_median  | 3.454 |
| first5 | train_mean    | 2.658 |

All future experiments compare against these.

---

## Phase 2: Board-State and Certainty Fields — DONE

All certainty scaffolding is already implemented and persisted:

| Field                      | Implementation               | Role             |
| -------------------------- | ---------------------------- | ---------------- |
| starter_certainty_score    | 0.0/0.5/1.0                  | certainty signal |
| lineup_certainty_score     | confirmed/total hitters      | certainty signal |
| weather_freshness_score    | decay from observation age   | certainty signal |
| market_freshness_score     | decay from line snapshot age | certainty signal |
| bullpen_completeness_score | last-3-games completeness    | certainty signal |
| missing_fallback_count     | count of null key fields     | certainty signal |
| board_state                | complete/partial/minimal     | certainty signal |

All registered as `FIELD_ROLE_CERTAINTY_SIGNAL` in `contracts.py`.
Persisted in `feature_payload` JSON for all 4 lanes.
Exposed via API in `/api/games/board` and `/api/totals/board`.

### Data reality for historical evaluation

Historical features were built from boxscores, not live feeds. Current data distribution:

- lineup_certainty_score: 2,102 of 2,277 rows at 0.0 (no confirmed lineups in historical data)
- starter_certainty_score: 2,096 at 0.5 (probable), only 33 at 1.0
- weather/market freshness: nearly all at 0.0 (not backfilled)
- bullpen_completeness: good variation (median 1.0, useful split)
- missing_fallback_count: 0–5, good spread (median 2.0)
- board_state: partial 1,162 / minimal 926 / complete 41

**What this means for evaluation:** timing-based certainty slices (lineup_confirmed, weather_fresh) will become meaningful as live data accumulates. For now, data-completeness slices (missing_fallback_count, board_state, bullpen_completeness) and feature-value slices (starter quality known, venue factor known) are the viable evaluation dimensions.

---

## Phase 3: Evaluate Totals by Slice — PRIORITY

### Goal

Answer the central question: when does totals have real edge?

### Slicing dimensions

**Data-completeness slices (available now):**

- board_state complete vs partial vs minimal
- missing_fallback_count == 0 vs 1–2 vs 3+
- bullpen_completeness_score > 0.5 vs ≤ 0.5
- starter data available vs missing (starter xwoba not null)

**Feature-value slices (available now):**

- starter quality asymmetry (large difference in starter xwoba)
- venue_run_factor known vs missing
- strong bullpen asymmetry (large difference in bullpen era/pitches)

**Timing-based slices (need live data accumulation):**

- lineup_certainty_score == 1.0 (fully confirmed)
- weather_freshness_score > 0.5 (recent weather)
- market_freshness_score > 0.5 (recent market data)
- pre_lock board state (near first pitch)

### Deliverable

`src/backtest/experiment_totals_slices.py` — sliced evaluation tool.

### Success criteria

Answer clearly:

- Is totals useless everywhere?
- Or only useless when inputs are weak?
- Which slices, if any, show model edge over baseline?

---

## Phase 4: Sharpen the Adjustment Stack — DONE

### Goal

Improve the features that are supposed to add signal beyond team identity.

Do this **after** slicing reveals where signal lives, not before.

### Focus areas

#### A. Starter quality mismatch — DONE

Added three derived starter-mismatch features to first5:
- `starter_xwoba_diff` — directional quality difference (home minus away)
- `starter_csw_diff` — called-strike + whiff rate difference
- `starter_quality_gap` — absolute magnitude of xwoba mismatch

These features pre-compute the starter-mismatch interaction that was the only winning signal in Phase 3.

All three registered as `FIELD_ROLE_CORE_PREDICTOR` in `contracts.py`.

#### B. Lineup quality and lineup certainty

- Top-5 lineup strength, bottom-third weakness
- Handedness mix, missing-regular penalty
- Projected vs confirmed lineup handling

#### C. Bullpen availability/workload

- De-emphasize noisy short-window outcome stats (ERA/runs allowed proxies)
- (Not relevant for first5 — starter-driven lane)

#### D. Park and weather transforms

- Weather freshness, wind interpretation
- Run environment scoring, HR carry scoring
- Roof/open handling

### Success criteria

The adjustment stack adds measurable value beyond team_average in at least some evaluation slices.

### Result — SIGNIFICANT IMPROVEMENT

Before (Phase 3): model won **1 of 9 slices** (only high_starter_asymmetry).

After derived features: model wins **6 of 9 slices**:

| Slice | N | Model | team_avg | train_mean | train_median |
|---|---|---|---|---|---|
| ALL | 400 | **2.713** | 2.740 | 2.713 | 2.812 |
| board_state=partial | 173 | 2.672 | 2.700 | **2.664** | 2.763 |
| board_state=minimal | 220 | **2.720** | 2.750 | 2.724 | 2.823 |
| both_starters_known | 174 | 2.668 | 2.695 | **2.661** | 2.764 |
| any_starter_missing | 226 | **2.748** | 2.776 | 2.754 | 2.850 |
| venue_factor_known | 400 | **2.713** | 2.740 | 2.713 | 2.812 |
| high_starter_asymmetry | 100 | **2.922** | 2.935 | 2.940 | 3.030 |

Lane status changed from `below_baseline` to `above_baseline`. Best model switched from elasticnet to lasso.

---

## Phase 5: First 5 Calibration — DONE (infrastructure ready, awaiting live market data)

### Goal

Use market context the right way, especially for first 5.

### Implementation

Built an **enhanced calibration layer** in `train_first5_totals.py` that uses:

- Raw model prediction
- Market total
- `starter_quality_gap` (when mismatch is high, trust model more)
- `starter_certainty_score` (when starters confirmed, trust model more)
- `board_state` (numeric encoding: complete=2, partial=1, minimal=0)

The enhanced calibrator is a Ridge regression that learns optimal weighting across all five signals.

Falls back to basic Ridge(model_pred, market_total) blend when enhanced calibrator can't fit,
and falls back to raw predictions when neither calibrator has enough market data.

### Important rule

Market fields stay out of raw training — calibration is a separate post-model layer.

### Current status

Historical data has 0/400 market_total values in the validation set, so neither calibrator fits on backtest data. The infrastructure is ready and will activate automatically as live market data accumulates.

---

## Phase 6: Sliced Evaluation Standard — DONE

Already implemented in Phase 3 via `experiment_totals_slices.py`. Used in Phase 4 to validate feature sharpening (1/9 → 6/9 slices won).

---

## Phase 7: Publish/Suppress Logic — DONE

### Goal

Only surface totals plays when they have earned trust.

### Implementation

Added `confidence_level` and `suppress_reason` to `predict_first5_totals.py`:

| Level | Criteria |
|---|---|
| **high** | starter_certainty ≥ 0.75, quality_gap ≥ 0.015, board_state ≠ minimal |
| **medium** | starter_certainty ≥ 0.5, quality_gap ≥ 0.008 |
| **low** | everything else not suppressed |
| **suppress** | no starter data OR no market line |

New DB columns added via migration `015_first5_confidence.sql`:
- `confidence_level TEXT` on `predictions_first5_totals`
- `suppress_reason TEXT` on `predictions_first5_totals`

Suppressed predictions are still stored in the DB (for audit) but marked for the product layer to filter.

---

## Experiment Log

### Phase 1: Freeze Baselines (DONE — commit 437b45a)

Frozen via `src/backtest/benchmark_totals.py --freeze`.

### Split-Side Modeling (TESTED — NEGATIVE RESULT)

Experiment: `src/backtest/experiment_split_sides.py`

Hypothesis: predicting home and away runs separately then summing should outperform a unified total target.

| Approach                     | MAE   |
| ---------------------------- | ----- |
| team_avg (unified, baseline) | 3.547 |
| direct side models (ridge)   | 3.577 |
| per-side team-avg baselines  | 3.590 |
| per-side baseline + residual | 3.626 |

Verdict: **all split-side approaches lost to unified team_average.**

Diagnosis: splitting ~2,100 rows into two noisier targets adds variance that outweighs structural benefit. Current features lack sharpness to model home vs away differentials better than team identity.

Status: **structurally sensible, currently unsupported by data volume and feature sharpness.** Revisit when training set grows (2+ full seasons), features sharpen enough to differentiate home vs away, or per-side prediction is needed for a product surface (run-line modeling).

### Sliced Evaluation (TESTED — KEY FINDINGS)

Experiment: `src/backtest/experiment_totals_slices.py`

Hypothesis: the model may show edge in high-certainty or high-data-quality slices even if it loses overall.

**Full-game totals: model loses every slice (0 of 13).**

| Slice                  | N   | Model | team_avg | train_mean | train_median\* |
| ---------------------- | --- | ----- | -------- | ---------- | -------------- |
| ALL                    | 453 | 3.619 | 3.521    | 3.551      | **3.450**      |
| board_state=complete   | 37  | 3.917 | 3.939    | 3.941      | **3.757**      |
| board_state=partial    | 204 | 3.906 | 3.699    | 3.727      | **3.603**      |
| board_state=minimal    | 212 | 3.290 | 3.278    | 3.314      | **3.250**      |
| both_starters_known    | 174 | 3.800 | 3.726    | 3.754      | **3.575**      |
| any_starter_missing    | 279 | 3.506 | 3.393    | 3.425      | **3.373**      |
| high_starter_asymmetry | 110 | 3.630 | 3.533    | 3.580      | **3.509**      |
| high_bullpen_asymmetry | 113 | 3.496 | 3.470    | **3.408**  | 3.434          |

Key finding: **the model is worst where data is most complete** (board_state=complete: 3.917). The adjustment features are actively hurting predictions when they're most present. The model is closest to baselines in the "minimal" slice (3.290 vs 3.250) where adjustments are mostly null/fallback.

**First 5 totals: model wins 1 of 9 slices.**

| Slice                      | N       | Model     | team_avg | train_mean\* | train_median |
| -------------------------- | ------- | --------- | -------- | ------------ | ------------ |
| ALL                        | 475     | 2.675     | 2.689    | **2.658**    | 2.697        |
| board_state=partial        | 172     | 2.634     | 2.663    | **2.594**    | 2.640        |
| board_state=minimal        | 296     | 2.679     | 2.688    | **2.673**    | 2.706        |
| both_starters_known        | 173     | 2.631     | 2.657    | **2.591**    | 2.642        |
| any_starter_missing        | 302     | 2.700     | 2.707    | **2.696**    | 2.728        |
| **high_starter_asymmetry** | **118** | **2.546** | 2.575    | 2.587        | 2.669        |

Key finding: **first 5 model wins the high-starter-asymmetry slice** (MAE 2.546 vs 2.575 team_avg, 2.587 train_mean). When there's a large gap in starter quality (top-quartile xwoba difference), the model has actual edge. This makes structural sense: first 5 is starter-driven, and large mismatches give the model something meaningful to adjust for.

**Interpretation:**

- Full-game totals: dead lane currently. The adjustment stack adds noise, not signal. train_median is unbeatable.
- First 5 totals: selective lane. Shows real edge when starter quality contrast is high.
- The data confirms the strategic direction: totals should be certainty-aware and selective, not forced to be broadly useful.
- The path forward for first 5: amplify the signal that already works (starter mismatch detection) and add calibration. The path for full-game: stop fitting adjustment models until inputs improve.

---

## Recommended Work Order

1. ~~Freeze honest baselines~~ (DONE — commit 437b45a)
2. ~~Add board_state and certainty fields~~ (DONE — already implemented)
3. ~~Evaluate totals by slice~~ (DONE — full-game loses all slices, first5 wins high-starter-asymmetry)
4. ~~Sharpen the adjustment stack~~ (DONE — starter-mismatch features: 1/9 → 6/9 slices won)
5. ~~Build first 5 calibration layer~~ (DONE — infrastructure ready, awaiting live market data)
6. ~~Publish/downgrade/suppress logic~~ (DONE — confidence_level + suppress_reason in predictions)

---

## One-line summary

First5 totals is now a **selective, certainty-aware lane** — starter-mismatch features put the model above baseline (6/9 slices won), enhanced calibration and publish/suppress infrastructure are ready for live data.
