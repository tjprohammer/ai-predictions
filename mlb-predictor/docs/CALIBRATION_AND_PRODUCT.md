# Calibration, trust, and product verification

This doc closes the loop between [MODEL_REWORK_PLAN.md](MODEL_REWORK_PLAN.md) Phase 5–6 and what the app ships today. It is the operational counterpart to model training notes in `MODEL_NOTES.md`.

## What is implemented (product)

| Theme | Where |
|--------|--------|
| Input trust A–D (pregame context quality) | `src/utils/input_trust.py`, certainty payload on the board (`data_quality.input_trust`) |
| Best-bet stratification | `promotion_tier`: `actionable` (EV + trust A/B) vs `edge_only` (EV with C/D) — `src/utils/best_bets.py` |
| Graded history by trust | `GET /api/recommendations/best-bets-history` returns `by_input_trust`, `by_input_trust_market`, `monotonicity` |
| Market freeze before features | `src.transforms.freeze_markets` → `market_selection_freezes` (see `FEATURE_DICTIONARY.md`) |

## Phase 6 “backtest” without a separate harness

Use **stored outcomes** and **APIs** instead of a one-off script until a dedicated harness exists:

1. **Trust vs win rate (monotonicity)** — Call best-bets history with a trailing `window_days`, `graded_only=true`. Read `monotonicity.status` (`monotonic`, `non_monotonic`, `insufficient_sample`) and `by_input_trust`. From the repo root, **`python scripts/report_best_bet_trust.py --graded-only`** prints the same summaries (uses `app_logic` against your configured DB).
2. **Model scorecards** — `GET /api/model-scorecards` (existing) for lane-level calibration summaries.
3. **CLV and picks** — Best Bet history rows include CLV fields; CLV review panels use the same date window.

**Definition:** “Actionable” spots should show **better** realized results (win rate, CLV, or Brier where applicable) than “edge_only” over enough samples; if not, tighten thresholds or trust gates (Phase 5 publish rules).

## Totals narrative vs model

The board’s **“why the model leans”** strings (`totalsReasonList` in `index.html`) are **supporting rationales** built from the same **projection inputs** (expected runs, park, weather, starters, pens, line move). They are **not** a second independent model.

**Authoritative** totals signals for the lane are:

- `totals.predicted_total_runs`, `totals.edge`, `totals.market_total`
- `totals.lane_status`, `totals.confidence_level`, `totals.suppress_reason`
- `data_quality.input_trust` (projection vs input quality)

See [TOTALS_NARRATIVE.md](TOTALS_NARRATIVE.md) for the full contract.

## Operational cadence

- **Pregame:** Refresh Everything (or Prepare Slate + markets) once inputs are stable; freeze step pins market selection for features.
- **Live slate:** Prefer **Update Scores** / **Refresh Daily Results** (Update Center shows “Scores & boxes · safe for live slates”) over Refresh Everything unless you intentionally want new odds and a full feature rebuild.
- **Retrain:** On your schedule when labeled history in snapshots grows; not nightly.
