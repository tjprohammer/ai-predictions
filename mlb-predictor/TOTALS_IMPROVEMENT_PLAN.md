# Totals Improvement Plan

## Purpose

This document defines the next improvement path for the full-game total runs and first 5 total runs lanes in mlb-predictor.

The goal is not to force a raw model to predict every MLB game better than every baseline. The goal is to make totals useful by improving:

- raw run projection quality
- certainty-awareness
- board-state handling
- calibration against market context
- publish/suppress behavior

This plan follows the current rework direction where projection, certainty, calibration, diagnostics, and product behavior are treated as separate layers.

---

## Current Honest State

Recent diagnostics show:

- full-game totals improved after feature cleanup, but still do not beat the strongest simple baselines
- first 5 totals also do not beat baseline and appear even more sensitive to loss of market information
- the current totals adjustment features are not adding enough signal beyond basic baseline context
- strikeouts is the only lane currently showing clear predictive edge above baseline
- Phase 1 already separated core predictors from calibration inputs, certainty signals, diagnostic flags, and product-only fields, which gives totals a cleaner foundation for the next redesign

**What that means:**

Totals is not ready for "tune forever and hope."
Totals needs a more disciplined redesign.

---

## Core Diagnosis

The problem is not simply "more model tuning needed."

The current evidence suggests:

- broad full-board totals prediction is too noisy with the current feature set
- team-level baseline context is carrying more signal than the current game-specific adjustments
- first 5 totals likely needs a different structure than just "full-game totals but shorter"
- certainty and calibration layers are probably where a lot of practical value will come from, not just raw MAE reduction

---

## Target End State

The totals system should eventually have five layers:

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

## Phase 1: Freeze the Current Honest Baseline

### Goal
Lock in the current state so future improvements can be measured honestly.

### Tasks
- preserve the current best totals and first 5 baseline comparisons
- store benchmark outputs for:
  - train_mean
  - train_median
  - team_average
  - current best raw model
  - current best baseline + residual model
- make these comparisons part of the standard totals evaluation flow

### Deliverable
A reproducible benchmark table for both totals lanes.

### Success criteria
Every future totals experiment must answer:
- does it beat team_average?
- does it beat train_median or train_mean?
- does it improve only MAE, or also usefulness in high-certainty slices?

This matches the recent diagnostics showing totals and first 5 are still below their strongest baselines.

---

## Phase 2: Split Full-Game Totals and First 5 Into Separate Strategies

### Goal
Stop treating these as near-identical lanes.

Your feature dictionary currently shows first 5 largely mirroring full-game totals, with Phase 1 already noting that this needs rethinking.

### Full-game totals strategy

Model:
- home team expected runs
- away team expected runs
- combine into full-game total

### First 5 strategy

Model:
- early-inning scoring only
- starter strength
- starter leash / first-time-through exposure
- top-of-lineup quality
- early run environment

Less emphasis on:
- full bullpen context
- late-game run volatility

### Deliverable
Two distinct modeling specs:
- `FULL_GAME_TOTALS_SPEC`
- `FIRST5_TOTALS_SPEC`

### Success criteria
The repo no longer treats first 5 as a near-clone of full-game totals.

This follows the model rework doc's direction that full-game totals and first 5 are separate lanes and that first 5 should remain starter- and top-of-lineup-first.

---

## Phase 3: Rebuild the Game Adjustment Features

### Goal
Improve the parts of totals that are supposed to add signal beyond team identity.

### Focus areas

#### A. Starter quality and starter certainty

Improve:
- starter identity reliability
- recent pitch count
- recent leash behavior
- recent command deterioration
- velocity trend
- handedness-adjusted opponent threat

Why: The rework plan already centers starter quality and starter certainty as major totals drivers.

#### B. Lineup quality and lineup certainty

Improve:
- top-5 lineup strength
- bottom-third weakness
- handedness mix
- missing-regular penalty
- projected vs confirmed lineup handling

Why: The feature dictionary already shows lineup-driven fields are important across totals lanes, but they need stronger timing rules and probably sharper transforms.

#### C. Bullpen availability/workload

Shift emphasis toward:
- pitches last 3 days
- innings last 3 days
- leverage-arm availability
- back-to-back usage
- bullpen completeness

De-emphasize or test retirement of:
- short-window bullpen outcome noise like ERA/runs allowed proxies unless they prove helpful

Why: The docs already flagged bullpen workload vs outcome noise as an open audit question.

#### D. Park and weather transforms

Improve:
- weather freshness
- wind interpretation
- run environment scoring
- HR carry scoring
- roof/open handling where available

Why: The model rework explicitly says weather and park belong in the numeric prediction layer, not just explanations.

### Deliverable
A reduced, sharper totals feature stack built around:
- baseline team scoring
- starter adjustment
- lineup adjustment
- bullpen availability/workload adjustment
- park/weather adjustment

### Success criteria
The adjustment stack must add measurable value beyond team_average, at least in some evaluation slices.

---

## Phase 4: Add Board-State Modeling

### Goal
Stop training/evaluating totals as one blended problem.

The docs already define board states:
- early
- morning
- lineup_confirmed
- pre_lock

### Tasks
- make board_state explicit in totals features and outputs
- evaluate totals separately by board state
- test whether totals only becomes useful in later states

### Core experiments
- early board only
- morning board only
- lineup-confirmed only
- pre-lock only

### Deliverable
Board-state-sliced totals evaluation.

### Success criteria
Answer this clearly:
- is totals useless everywhere?
- or only useless when inputs are weak?
- or actually useful once lineup/weather/starter certainty improves?

This is one of the highest-value questions in the whole project.

---

## Phase 5: Add Certainty Scaffolding

### Goal
Separate "prediction quality" from "input trust."

The feature dictionary already proposes:
- starter_certainty_score
- lineup_certainty_score
- weather_freshness_score
- market_freshness_score
- bullpen_completeness_score
- missing_fallback_count
- board_state

### Tasks
Implement totals-specific certainty signals:
- starter certainty
- lineup certainty
- weather freshness
- market freshness
- bullpen completeness
- fallback count

### Deliverable
A totals certainty payload for both full-game and first 5.

### Success criteria
Every totals prediction can say:
- what the raw number is
- how trustworthy the input state is
- whether it should be shown, downgraded, or suppressed

This is one of the main principles of the model rework plan.

---

## Phase 6: Add Post-Model Calibration

### Goal
Use market context the right way.

Phase 1 already concluded that:
- market-derived fields should not live in raw model training
- first 5 especially suffered when market_total was removed from the raw predictor set
- calibration is the priority follow-up for first 5 totals

### Tasks
Build a post-model calibration layer using:
- raw model projection
- market total
- line movement
- market freshness
- certainty bucket
- board state

### Important rule
Do not push pricing fields back into raw training.

### Deliverable
Calibrated totals output:
- raw total projection
- calibrated total projection
- market gap
- confidence/certainty bucket

### Success criteria

**For first 5:**
- recover toward prior baseline competitiveness without reintroducing market leakage into raw training

**For full-game totals:**
- test whether calibration improves decision quality even if raw MAE gains remain modest

---

## Phase 7: Change the Evaluation Standard

### Goal
Stop judging totals only by broad MAE.

### Required evaluation views
- overall MAE
- lineup-confirmed-only MAE
- outdoor-only MAE
- warm-weather MAE
- high-certainty-only MAE
- strongest disagreement-vs-market slice
- A/B certainty bucket performance vs C/D

### Why
The rework plan is already pointing toward certainty-aware product behavior, not universal all-games prediction.

### Deliverable
Totals scorecard by slice, not just one aggregate number.

### Success criteria
Find out whether totals is:
- a dead lane
- a selective lane
- or a broad lane that just needs better modeling

My guess is it becomes a selective lane if improved correctly.

---

## Phase 8: Publish/Suppress Logic

### Goal
Only surface totals plays when they have earned trust.

### Tasks
Define:
- publish thresholds
- downgrade thresholds
- suppress thresholds

Use:
- raw edge
- calibrated edge
- certainty score
- board state
- fallback count

### Deliverable
Totals surface behavior for:
- full-game totals
- first 5 totals

### Success criteria
The app stops presenting weak-input totals predictions as equally actionable.

This is explicitly aligned with the rework plan's "edge and certainty must be separate" rule.

---

## Success Targets

These are practical targets, not promises.

### Full-game totals

**Near-term:**
- beat team_average reliably
- close the gap to train_median

**Mid-term:**
- beat both in at least lineup-confirmed / high-certainty slices

**Long-term:**
- broad-lane usefulness if raw projections, certainty, and calibration all improve together

### First 5 totals

**Near-term:**
- beat train_mean honestly
- recover competitiveness through post-model calibration

**Mid-term:**
- become useful in starter-confirmed / lineup-confirmed slices

**Long-term:**
- operate as a distinct early-innings lane, not a clone of full-game totals

---

## What Not To Do

- do not reinsert market pricing fields into raw training
- do not reinsert certainty signals into raw training just to improve MAE
- do not judge totals only by one aggregated metric
- do not assume first 5 and full-game totals should share identical modeling logic
- do not keep expanding feature count without proving each feature family adds signal

---

## Recommended Work Order

1. freeze honest baselines
2. split full-game vs first 5 strategy
3. rebuild starter / lineup / bullpen / weather adjustments
4. add explicit board state
5. add certainty scaffolding
6. add post-model calibration
7. evaluate by slice
8. wire publish/suppress behavior

---

## One-line summary

Totals gets better by becoming cleaner, later, and more selective — not by forcing a bigger raw model to predict every game equally well.
