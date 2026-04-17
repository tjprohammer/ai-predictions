# Rework roadmap: HR model and NRFI/YRFI (experimental)

These lanes are **separate** from the team **green strip** (see `BOARD_AND_DAILY_RESULTS_PRODUCT.md`). Improving them means **features + training + calibration + evaluation**, not changing board composition.

## Code map

| Lane | Features | Train | Predict | Notes |
|------|----------|-------|-----------|--------|
| **HR (P(HR))** | `src/features/` (HR builder used by `hr_builder` CLI — see update job sequence) | `src/models/train_hr.py` | `src/models/predict_hr.py` | Uses `HR_TARGET_COLUMN`, field roles from `src/features/contracts.py`; calibration logic (logit / isotonic) in `train_hr`. |
| **Inning-1 NRFI** | `src/features/inning1_nrfi_builder.py` | `src/models/train_inning1_nrfi.py` | `src/models/predict_inning1_nrfi.py` | Target: `INNING1_NRFI_TARGET_COLUMN`; needs enough labeled rows (`_MIN_LABELED_ROWS` in trainer). |

**UI / product surfaces:** slugger strip and experimental carousel consume **`predictions_player_hr`** and **`predictions_inning1_nrfi`** (plus market rows), shaped in `app_logic` and `product_surfaces` — fix quality at **prediction** time, not in the API layer.

## What “better calculations” usually means

1. **Labels** — HR and 1st-inning runs must match **final** boxscores; stale or partial ingest poisons training.
2. **Features** — Add or fix pitcher/batter/park/context columns in the **builder**, then extend `feature_columns_for_roles` / contracts if new fields are part of the snapshot.
3. **Class balance & rare events** — HR is extremely imbalanced; `train_hr` already uses weighting and calibration — revisit **calibrator choice**, clipping, and whether **isotonic** is collapsing (see `_isotonic_eligible` in `train_hr.py`).
4. **NRFI sample size** — Trainer exits early if labeled rows &lt; 120; expand historical backfill (`inning1_nrfi` snapshots) before expecting stable AUC/Brier.
5. **Backtesting** — Compare Brier / log-loss / calibration bins **out of time** (chronological split is already in `train_*`; extend with rolling backtests or `product_surfaces` outcome tables if you add them).

## Suggested workflow

1. Snapshot current metrics from training reports (`save_report` outputs) and, if available, **Daily Results** / `prediction_outcomes_daily` for HR and experimental rows.
2. Fix **data** (boxscores, `total_runs_inning1`, HR outcomes) for the train window.
3. Iterate **features → retrain → predict** for one date range; compare to baseline on holdout weeks.
4. Only then adjust **slugger ranking** (`src/utils/slugger_hr_selection.py`) or **carousel copy** — ranking should follow better probabilities.

## Tests to keep green while refactoring

- `tests/test_experimental_nrfi_yrfi.py`, `tests/test_nrfi_experimental_context.py`
- Any HR training smoke tests under `tests/` matching `train_hr` / `predict_hr`

This file is a **roadmap**; delete or rewrite sections as experiments land.
