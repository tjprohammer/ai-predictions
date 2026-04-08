# Prediction Model Experiments

Tracking branch: `feat/prediction-model-experiments`

---

## Experiment 1: Dual Predictions (Fundamentals vs Market-Calibrated)

**Hypothesis:** Market calibration collapses predictions toward consensus.
Tracking both allows day-over-day comparison to measure whether the fundamentals-only
prediction or the market-calibrated prediction is more accurate.

**Implementation:**
- Totals: `predicted_total_fundamentals` = baseline + residual model output (before Ridge
  market calibrator or 50% market-anchor fallback)
- Strikeouts: `predicted_strikeouts_fundamentals` = model output after isotonic correction
  but before market calibration Ridge

**Columns added:** `predictions_totals.predicted_total_fundamentals`,
`predictions_pitcher_strikeouts.predicted_strikeouts_fundamentals`

**UI:** Both values displayed on totals board, pitchers page, and results page.
Fundamentals shown with "(F)" label.

**Evaluation plan:** After accumulating 5+ days of dual predictions, compare MAE of
fundamentals-only vs calibrated vs market line against actual outcomes.

| Date | Games | Totals Fund MAE | Totals Cal MAE | Mkt MAE | K Fund MAE | K Cal MAE | K Mkt MAE | Notes |
|------|-------|-----------------|----------------|---------|------------|-----------|-----------|-------|
| | | | | | | | | |

---

## Experiment 2: Strikeout Differentiation Features

**Hypothesis:** K predictions cluster because features are slow-moving 3–5 start rolling
averages with only one team-level opponent feature. Adding pitcher-vs-team history,
lineup-level K%, and venue K factor should spread predictions apart.

**New features:**
- `pitcher_vs_team_k_rate` — historical K/start for this pitcher vs the specific opponent
- `opponent_lineup_k_pct_recent` — average K% of opponent lineup batters over last 7 games
- `venue_k_factor` — venue K-per-game ratio vs league average (>1 = K-friendly park)

**Evaluation plan:** Compare prediction variance (std dev) before and after feature
addition. Check whether spread correlates with actual outcome spread.

---

## Experiment 3: Prior Compression Tuning

**Hypothesis:** Early-season Bayesian prior blending pulls predictions ~70% toward league
average with only 3 starts. Reducing compression may improve early-season accuracy at the
cost of higher variance.

**Current settings:**
- `PRIOR_BLEND_MODE = standard` (default)
- `PRIOR_WEIGHT_MULTIPLIER = 1.0` (default)
- `TEAM_FULL_WEIGHT_GAMES = 30`
- `PITCHER_FULL_WEIGHT_STARTS = 10`

**Planned trials:**
| Trial | PRIOR_BLEND_MODE | PRIOR_WEIGHT_MULTIPLIER | TEAM_FULL_WEIGHT_GAMES | PITCHER_FULL_WEIGHT_STARTS | Result |
|-------|------------------|-------------------------|------------------------|-----------------------------|--------|
| A (baseline) | standard | 1.0 | 30 | 10 | |
| B (reduced) | reduced | 1.0 | 30 | 10 | |
| C (aggressive) | standard | 0.5 | 20 | 6 | |
| D (current only) | current_only | 1.0 | 30 | 10 | |

**Evaluation plan:** Run each trial for 3+ days, compare day-over-day prediction accuracy.
Use `src/backtest/compare_prior_modes.py` for historical comparison.
