# Totals “lean” copy (`totalsReasonList`)

## Purpose

`totalsReasonList` in `src/api/static/index.html` produces short bullet reasons for the full-game total **direction** (over vs under vs flat). It helps users see **which features** align with the lean: team run context, park, weather, starter shape, bullpen, line movement.

## What it is not

- **Not** a calibrated probability statement by itself.
- **Not** guaranteed to list every driver the model uses internally.
- **Not** a substitute for `lane_status`, `confidence_level`, `suppress_reason`, or **input trust** — those describe **model lane** and **input quality** explicitly.

## What was added (April 2026)

The same function now **prepends** concise lines from:

- **Model lane:** `totals.lane_status`, `totals.confidence_level`, `totals.suppress_reason`
- **Input trust:** `data_quality.input_trust.grade` and a truncated `summary`

So the UI separates **“what the lane says”** and **“how good the inputs are”** from the **heuristic story** bullets.

## Heuristic thresholds

Internal thresholds (e.g. `4.6` expected runs, `0.335` top-5 xwOBA, park factor cutoffs) are **display helpers**. Changing them only affects copy, not scoring. To change the actual model, edit training/scoring code and artifacts, not this list.
