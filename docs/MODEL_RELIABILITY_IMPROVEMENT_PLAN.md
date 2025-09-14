# Model Reliability Improvement Plan

Status: DRAFT (Phase 0 – Alignment)
Last Updated: 2025-09-08
Owner: (assign)

## Objectives

Improve trustworthiness and betting utility of MLB totals predictions by:

- Reducing error variance (stable MAE slice-to-slice)
- Restoring calibration (predicted vs realized totals slope ~1.0)
- Ensuring edges correlate with realized value (monotonic lift)
- Eliminating hidden data leakage / stale feature issues
- Introducing guardrails so unreliable models never auto-promote

## High-Level Phases

| Phase | Name                    | Core Output                                     | Promotion Gate                                    |
| ----- | ----------------------- | ----------------------------------------------- | ------------------------------------------------- |
| 1     | Diagnostics Baseline    | Slice metrics + residual dataset                | Completed 90d rolling-origin report               |
| 2     | Feature Audit & Hygiene | feature_audit.csv + reduced feature whitelist   | Coverage ≥90%, removed unstable features          |
| 3     | Protocol Overhaul       | Rolling-origin training harness + time-aware CV | Slice MAE stdev / mean ≤0.15                      |
| 4     | Bias & Calibration      | Serve-time bias adjust + calibration curve      | Calibration slope 0.90–1.10                       |
| 5     | Edge Quality            | Edge decile lift table + edge confidence metric | Top decile win% > median decile                   |
| 6     | Distribution & Variance | Mean + variance modeling + prediction intervals | 68% PI hit rate within ±5% tolerance              |
| 7     | Ensemble (Optional)     | Weighted blend with constraints                 | ≥1% MAE gain vs best single                       |
| 8     | Monitoring & Guardrails | Automated daily reliability report + alerts     | All guardrail thresholds green 7 consecutive days |
| 9     | Deployment Gate         | Versioned promotion policy                      | Policy documented & enforced                      |

## Phase 1 – Diagnostics Baseline

Deliverables:

- Script: `mlb/core/model_diagnostics.py`
- Outputs:
  - `outputs/diagnostics/slice_metrics.csv` (train_start, train_end, test_window, n_games, MAE, RMSE, DirAcc, calibration_slope, bias, residual_std)
  - `outputs/diagnostics/residuals.csv` (game_id, date, predicted, market_total, actual, residual, feature_count)
  - Charts (optional later) saved as PNG: residual_vs_pred, residual_distribution, calibration_curve
    Method:
- Rolling-origin windows (example): train 90 days → test next 7; slide by 7 until current.
- Compute linear regression slope (actual ~ predicted) on each test slice.
  Success Criteria to advance:
- Baseline metrics generated for ≥8 slices (≈ 2 months coverage) and archived.

## Phase 2 – Feature Audit & Hygiene

Deliverables:

- Script: `mlb/core/feature_audit.py`
- `feature_audit.csv` columns: feature, coverage_pct, null_count, mean, std, roll14_std, stability_ratio (roll14_std/std), abs_corr_target, abs_corr_residual (after baseline), variance_inflation_group, drop_reason(optional)
  Actions:
- Drop features with coverage <70% unless mission-critical.
- Merge collinear groups (|corr| > 0.90) keep highest stability_ratio inverse.
- Rebuild composite features only if they reduce residual variance >0.5% on validation slice.
  Success Gate:
- Selected feature set achieves equal or better MAE (±0.05) and improves calibration slope variance vs original.

## Phase 3 – Training Protocol Overhaul

Deliverables:

- Refactor training pipeline to support rolling-origin CV & time-aware hyperparam search.
- Persist `model_version.json` storing: hash(features+params), train_window_start, train_window_end, created_at UTC, metrics summary.
  Metrics:
- Slice MAE mean + stdev reported.
  Gate:
- Slice MAE stdev / mean ≤0.15 for chosen configuration.

## Phase 4 – Bias & Calibration Layer

Deliverables:

- Bias model: residual ~ market_total + temperature + lineup_strength_delta (or auto-selected covariates).
- Calibration curve data: bins of predicted_total (quantiles) vs mean actual & bin residual.
  Serve-Time Behavior:
- Apply bias correction only if last 14d residual mean statistically ≠ 0 (|t|>2).
  Gate:
- Calibration slope (actual vs corrected prediction) within 0.90–1.10 on latest test slice.

## Phase 5 – Edge Quality & Confidence

Deliverables:

- Edge decile table: decile(edge_abs), count, avg_edge, realized_win%, ROI (if graded bets), avg_residual.
- Edge confidence metric: `edge_conf = |edge| / predicted_std` (requires variance model in Phase 6 or bootstrap residual std proxy).
  Gate:
- Monotonic or near-monotonic lift: top decile win% ≥ (median decile win% + 5pp).

## Phase 6 – Distribution & Variance Modeling

Deliverables:

- Variance model: predict squared residuals or directly model Negative Binomial dispersion.
- Prediction intervals stored (p10, p50, p90) in predictions table.
  Metrics:
- Interval coverage: expected 80% interval contains ≈ 80% ±5% of outcomes over rolling 30d.
  Gate:
- 68% interval hit rate within ±5% of target & no extreme under-dispersion.

## Phase 7 – Ensemble (Optional)

Deliverables:

- Weighted blend (linear) of calibrated linear model + tree/GBR.
- Optimization: minimize rolling MAE subject to weights ≥0 & sum=1.
  Gate:
- ≥1% MAE improvement AND no degradation of calibration slope or edge monotonicity.

## Phase 8 – Monitoring & Guardrails

Deliverables:

- Daily script: `mlb/core/reliability_report.py` → `outputs/reliability/reliability_daily_<date>.json` & aggregated last_30d summary.
  Metrics & Thresholds:
- 7d MAE ≤ (30d MAE + 0.30)
- Calibration slope ∈ (0.85, 1.15)
- Edge bets last 50: win% ≥ 48%
- Feature coverage ≥ 90% of required set
- Drift test (mean residual last 7d vs prior 30d) |Δ| < 0.4
  Alerting:
- Log warning + (future) webhook/email; mark `gating_status` = FAIL if any breached.

## Phase 9 – Deployment Gate Policy

Mechanism:

- Predictions write path checks guardrail summary; if FAIL → write only to shadow columns (e.g., `predicted_total_candidate`).
- Manual promote command updates primary after human review.
  Artifacts:
- `PROMOTION_POLICY.md` documenting logic.

## Metrics Definitions

| Metric             | Definition                                                |
| ------------------ | --------------------------------------------------------- | ------------------- | --- |
| MAE                | mean(                                                     | prediction - actual | )   |
| DirAcc             | % where sign(pred - market) matches sign(actual - market) |
| Calibration Slope  | slope of OLS: actual ~ predicted                          |
| Bias               | mean(pred - actual) (negative = under-predicting)         |
| Edge               | predicted_total - market_total                            |
| Edge Win%          | % of bets (edge≥threshold) that win vs closing line       |
| Residual Stability | stdev(residual) rolling 14d                               |

## Quick Command Stubs (to be implemented)

```powershell
# Phase 1 diagnostics (placeholder)
python mlb/core/model_diagnostics.py --train-window 90 --test-window 7 --slices 10

# Phase 2 feature audit
python mlb/core/feature_audit.py --out outputs/diagnostics

# Generate reliability report
python mlb/core/reliability_report.py --days 30
```

## Data Tables Affected

- `enhanced_games` (read only for diagnostics)
- New (planned):
  - `model_slice_metrics`
  - `model_versions`
  - `prediction_intervals` (or added columns)
  - `reliability_guardrails`

## Rollback Plan

- Keep current production model artifacts intact (no overwrite) until Phase 4 passes gates.
- Each phase writes new artifacts under versioned directories (`models/experimental_vX`).
- Promotion requires copying to `models/active` (or updating pointer file) not deleting prior.

## Risks & Mitigations

| Risk                                 | Mitigation                                                 |
| ------------------------------------ | ---------------------------------------------------------- |
| Hidden leakage persists              | Add timestamp ordering & future value audit in diagnostics |
| Overfitting via complex ensemble     | Enforce slice variance threshold & calibration check       |
| Feature drift breaks model           | Daily guardrail drift test triggers full retrain           |
| Edge quality illusion (small sample) | Require minimum counts per decile before using table       |
| Under-dispersion of intervals        | Back-test coverage & inflate variance if consistently low  |

## Phase Advancement Checklist (Summary)

- Phase 1 → 2: slice_metrics.csv validated
- Phase 2 → 3: feature_audit.csv approved & features frozen
- Phase 3 → 4: rolling CV metrics stable
- Phase 4 → 5: calibration slope & bias corrected, edge recomputed
- Phase 5 → 6: edge lift monotonic
- Phase 6 → 7: interval coverage validated
- Phase 7 → 8: ensemble improves MAE ≥1%
- Phase 8 → 9: guardrail report green 7 consecutive days

## Immediate Next Actions (Week 1)

1. Implement Phase 1 diagnostics script skeleton.
2. Run on last 120 days; archive outputs.
3. Review residual patterns & identify candidate leakage indicators.
4. Begin feature audit script scaffolding.

---

This document will be updated at the end of each phase with actual metrics and links to artifacts.
