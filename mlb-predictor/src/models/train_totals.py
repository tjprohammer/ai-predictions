from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features.contracts import (
    FIELD_ROLE_CORE_PREDICTOR,
    TOTALS_META_COLUMNS,
    TOTALS_TARGET_COLUMN,
    feature_columns_for_roles,
    feature_field_roles,
)
from src.models.common import (
    calibrate_with_market,
    chronological_split,
    compute_sample_weights,
    encode_frame,
    fit_market_calibrator,
    load_feature_snapshots,
    log_loss_ou_vs_market_line,
    mean_pinball_loss,
    regression_metrics_by_month,
    regression_val_temporal_halves_mae,
    save_artifact,
    save_report,
)
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)

_FALLBACK_RESIDUAL_SCALE = 0.50
# Weak-model shrink caps: avoid w=1.0 so published totals are not identical to the posted line
# when the model only loses to train_median / team_average (market line is a separate baseline).
_MARKET_SHRINK_CAP = 0.85
_MIN_MARKET_BASELINE_ROWS = 20


def _post_calibrated_totals_val(
    fundamentals_pred: np.ndarray,
    val_frame: pd.DataFrame,
    market_calibrator: dict | None,
    residual_output: np.ndarray,
) -> np.ndarray:
    """Match ``predict_totals`` calibration + market-anchor fallback on validation rows."""
    calibrated, cal_mask = calibrate_with_market(
        fundamentals_pred, val_frame["market_total"], market_calibrator
    )
    out = calibrated.copy()
    mask = cal_mask.copy()
    market_vals = val_frame["market_total"]
    for i in range(len(out)):
        if mask[i]:
            continue
        mv = market_vals.iloc[i]
        if mv is not None and not pd.isna(mv):
            out[i] = float(mv) + _FALLBACK_RESIDUAL_SCALE * float(residual_output[i])
    return out


# ---------------------------------------------------------------------------
# Baseline-first architecture
# ---------------------------------------------------------------------------
# The baseline captures team run-environment context: the historical average
# total runs when this home team hosts and this away team visits.  The residual
# model learns the game-specific adjustment from starters, lineups, bullpen.
#
# Why a lookup baseline instead of a formula?  We tested
#   baseline = home_runs_rate_blended + away_runs_rate_blended (* park_factor)
# and it gave MAE 4.05-4.09 — worse than train_mean (3.60) because rolling
# run rates are too noisy at the game level.  The team-average lookup (MAE 3.56)
# is smoother and already beats the formula.
# ---------------------------------------------------------------------------

# Features the residual model trains on (game-specific adjustments only)
ADJUSTMENT_COLUMNS = [
    # Starting pitching matchup
    "home_starter_xwoba_blended", "away_starter_xwoba_blended",
    "home_starter_csw_blended", "away_starter_csw_blended",
    "home_starter_rest_days", "away_starter_rest_days",
    "home_starter_fb_velo_blended", "away_starter_fb_velo_blended",
    "home_starter_whiff_pct_blended", "away_starter_whiff_pct_blended",
    "home_starter_hard_hit_pct_blended", "away_starter_hard_hit_pct_blended",
    "home_starter_avg_ip_blended", "away_starter_avg_ip_blended",
    "home_starter_k_per_9_blended", "away_starter_k_per_9_blended",
    # Today's lineups
    "home_lineup_top5_xwoba", "away_lineup_top5_xwoba",
    "home_lineup_k_pct", "away_lineup_k_pct",
    # Bullpen state
    "home_bullpen_era_last3", "away_bullpen_era_last3",
    "home_bullpen_pitches_last3", "away_bullpen_pitches_last3",
    "home_bullpen_b2b", "away_bullpen_b2b",
    # Park effects
    "venue_run_factor", "venue_hr_factor",
]


def compute_baseline(
    frame: "pd.DataFrame",
    home_avgs: "pd.Series",
    away_avgs: "pd.Series",
    fallback: float,
) -> np.ndarray:
    """Team run-environment baseline from training-set lookup tables.

    Returns (home_team_avg + away_team_avg) / 2 per row, falling back
    to *fallback* (global training mean) for unseen teams.
    """
    h = frame["home_team"].map(home_avgs).fillna(fallback).astype(float)
    a = frame["away_team"].map(away_avgs).fillna(fallback).astype(float)
    return ((h + a) / 2).values


def main() -> int:
    settings = get_settings()
    frame = load_feature_snapshots("totals")
    if frame.empty:
        log.info("No totals feature snapshots found")
        return 0

    trainable = frame[frame[TOTALS_TARGET_COLUMN].notna()].copy()
    if trainable.empty:
        log.info("No labeled totals feature rows found")
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough totals rows for validation split")
        return 0

    # --- Compute baselines ---
    y_train = train_frame[TOTALS_TARGET_COLUMN].astype(float)
    y_val = val_frame[TOTALS_TARGET_COLUMN].astype(float)
    train_mean = float(y_train.mean())

    # Team-average lookup baseline
    home_avgs = train_frame.groupby("home_team")[TOTALS_TARGET_COLUMN].mean()
    away_avgs = train_frame.groupby("away_team")[TOTALS_TARGET_COLUMN].mean()
    baseline_train = compute_baseline(train_frame, home_avgs, away_avgs, train_mean)
    baseline_val = compute_baseline(val_frame, home_avgs, away_avgs, train_mean)
    residual_train = y_train.values - baseline_train
    residual_val = y_val.values - baseline_val

    log.info(
        "Baseline — train MAE=%.3f val MAE=%.3f (baseline alone, before residual model)",
        float(mean_absolute_error(y_train, baseline_train)),
        float(mean_absolute_error(y_val, baseline_val)),
    )

    # --- Adjustment features for residual model ---
    all_core = feature_columns_for_roles(
        "totals",
        [FIELD_ROLE_CORE_PREDICTOR],
        available_columns=list(trainable.columns),
    )
    adjustment_features = [c for c in ADJUSTMENT_COLUMNS if c in all_core]
    log.info("Adjustment features (%d): %s", len(adjustment_features), adjustment_features)

    category_columns: list[str] = []
    X_train = encode_frame(train_frame[adjustment_features], category_columns)
    X_val = encode_frame(val_frame[adjustment_features], category_columns, training_columns=list(X_train.columns))

    # --- Train residual model candidates ---
    candidates = {
        "ridge": Ridge(alpha=1.0),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
    }
    best_name = None
    best_model = None
    best_rmse = math.inf
    best_final_predictions = None
    metrics = {}

    train_weights = compute_sample_weights(train_frame["game_date"])
    log.info(
        "Sample weights — min=%.3f median=%.3f max=%.3f (half_life=180d)",
        train_weights.min(), float(np.median(train_weights)), train_weights.max(),
    )

    for name, model in candidates.items():
        if hasattr(model, "steps"):
            last_step_name = model.steps[-1][0]
            model.fit(X_train, residual_train, **{f"{last_step_name}__sample_weight": train_weights})
        else:
            model.fit(X_train, residual_train, sample_weight=train_weights)
        residual_pred = model.predict(X_val)
        final_predictions = baseline_val + residual_pred
        mae = mean_absolute_error(y_val, final_predictions)
        rmse = math.sqrt(mean_squared_error(y_val, final_predictions))
        residual_mae = mean_absolute_error(residual_val, residual_pred)
        pinball = mean_pinball_loss(y_val.values, final_predictions, tau=0.5)
        metrics[name] = {"mae": mae, "rmse": rmse, "residual_mae": residual_mae, "pinball_median": pinball}
        log.info(
            "  %s: MAE=%.3f RMSE=%.3f residual_MAE=%.3f pinball=%.3f",
            name, mae, rmse, residual_mae, pinball,
        )
        if rmse < best_rmse:
            best_name = name
            best_model = model
            best_rmse = rmse
            best_final_predictions = final_predictions

    # --- Baseline benchmarks ---
    baselines = {}
    baselines["train_mean"] = {
        "mae": float(mean_absolute_error(y_val, [train_mean] * len(y_val))),
        "rmse": float(math.sqrt(mean_squared_error(y_val, [train_mean] * len(y_val)))),
    }
    train_median = float(y_train.median())
    baselines["train_median"] = {
        "mae": float(mean_absolute_error(y_val, [train_median] * len(y_val))),
        "rmse": float(math.sqrt(mean_squared_error(y_val, [train_median] * len(y_val)))),
    }
    baselines["team_average"] = {
        "mae": float(mean_absolute_error(y_val, baseline_val)),
        "rmse": float(math.sqrt(mean_squared_error(y_val, baseline_val))),
    }
    if "market_total" in val_frame.columns:
        market_mask = val_frame["market_total"].notna()
        if market_mask.sum() > 0:
            market_vals = val_frame.loc[market_mask.index[market_mask], "market_total"].astype(float)
            baselines["market_total"] = {
                "mae": float(mean_absolute_error(y_val[market_mask.values], market_vals)),
                "rmse": float(math.sqrt(mean_squared_error(y_val[market_mask.values], market_vals))),
                "rows": int(market_mask.sum()),
            }
    log.info("Baselines: %s", baselines)
    log.info(
        "Best model '%s' MAE=%.3f vs best baseline MAE=%.3f (%s)",
        best_name,
        metrics[best_name]["mae"],
        min(b["mae"] for b in baselines.values() if "mae" in b),
        min(baselines, key=lambda k: baselines[k].get("mae", float("inf"))),
    )

    artifact_name = f"totals_{settings.model_version_prefix}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    residual_std = float((y_val.values - best_final_predictions).std()) if len(y_val) else 1.0

    # --- Market calibration layer ---
    market_calibrator = fit_market_calibrator(
        best_final_predictions,
        val_frame["market_total"],
        y_val,
    )
    calibration_metrics = None
    if market_calibrator is not None:
        from sklearn.metrics import mean_absolute_error as _mae, mean_squared_error as _mse

        cal_mask = val_frame["market_total"].notna()
        cal_preds = market_calibrator["calibrator"].predict(
            np.column_stack(
                [best_final_predictions[cal_mask.values], val_frame.loc[cal_mask.index[cal_mask], "market_total"].astype(float).values]
            )
        )
        calibration_metrics = {
            "calibrated_mae": float(_mae(y_val[cal_mask.values], cal_preds)),
            "calibrated_rmse": float(_mse(y_val[cal_mask.values], cal_preds) ** 0.5),
            "raw_mae_same_rows": float(_mae(y_val[cal_mask.values], best_final_predictions[cal_mask.values])),
            "raw_rmse_same_rows": float(_mse(y_val[cal_mask.values], best_final_predictions[cal_mask.values]) ** 0.5),
            "calibration_rows": market_calibrator["calibration_rows"],
            "model_weight": market_calibrator["model_weight"],
            "market_weight": market_calibrator["market_weight"],
            "intercept": market_calibrator["intercept"],
        }

    residual_out = best_final_predictions - baseline_val
    post_cal = _post_calibrated_totals_val(
        best_final_predictions, val_frame, market_calibrator, residual_out
    )
    cal_res_std = (
        float(market_calibrator["calibration_residual_std"])
        if market_calibrator is not None
        else residual_std
    )
    cal_res_std = max(cal_res_std, 0.5)

    validation_metrics: dict[str, object] = {
        "fundamental_val_mae": float(mean_absolute_error(y_val, best_final_predictions)),
        "post_calibration_val_mae": float(mean_absolute_error(y_val, post_cal)),
        "pinball_median_fundamental": float(mean_pinball_loss(y_val.values, best_final_predictions, tau=0.5)),
        "pinball_median_post_calibration": float(mean_pinball_loss(y_val.values, post_cal, tau=0.5)),
        "validation_by_month_fundamental": regression_metrics_by_month(
            val_frame["game_date"], y_val.values, best_final_predictions
        ),
        "temporal_halves_mae_fundamental": regression_val_temporal_halves_mae(y_val.values, best_final_predictions),
    }
    ou_raw = log_loss_ou_vs_market_line(
        y_val.values, best_final_predictions, val_frame["market_total"], std=residual_std
    )
    if ou_raw:
        validation_metrics["ou_log_loss_vs_line_fundamental"] = ou_raw
    ou_post = log_loss_ou_vs_market_line(y_val.values, post_cal, val_frame["market_total"], std=cal_res_std)
    if ou_post:
        validation_metrics["ou_log_loss_vs_line_post_calibration"] = ou_post

    best_bl_mae = min(b["mae"] for b in baselines.values() if "mae" in b)
    market_baseline = baselines.get("market_total") or {}
    market_bl_mae = market_baseline.get("mae")
    market_bl_rows = int(market_baseline.get("rows") or 0)
    model_mae_fund = float(metrics[best_name]["mae"])
    loses_to_market_baseline = (
        market_bl_mae is not None
        and market_bl_rows >= _MIN_MARKET_BASELINE_ROWS
        and model_mae_fund > float(market_bl_mae)
    )
    if not loses_to_market_baseline:
        if market_bl_mae is None or market_bl_rows < _MIN_MARKET_BASELINE_ROWS:
            validation_metrics["market_shrink_skipped"] = (
                f"market_total baseline unavailable or rows<{_MIN_MARKET_BASELINE_ROWS} "
                f"(rows={market_bl_rows}); no convex shrink toward line"
            )
        else:
            validation_metrics["market_shrink_skipped"] = (
                f"fundamentals MAE {model_mae_fund:.3f} does not lose to market_total baseline "
                f"MAE {float(market_bl_mae):.3f}; shrink toward line disabled"
            )

    market_shrink = 0.0
    market_shrink_diagnostics: dict[str, float] | None = None
    if loses_to_market_baseline:
        mkt_series = val_frame["market_total"]
        has_m = mkt_series.notna().values
        if int(has_m.sum()) >= _MIN_MARKET_BASELINE_ROWS:
            best_w = 0.0
            best_mae_shrink = float(mean_absolute_error(y_val, post_cal))
            mkt_arr = mkt_series.astype(float).values
            for w in np.linspace(0.0, _MARKET_SHRINK_CAP, 18):
                blended = post_cal.copy()
                blended[has_m] = (1.0 - w) * post_cal[has_m] + w * mkt_arr[has_m]
                mae_w = float(mean_absolute_error(y_val, blended))
                if mae_w < best_mae_shrink - 1e-9:
                    best_mae_shrink = mae_w
                    best_w = float(w)
            market_shrink = best_w
            market_shrink_diagnostics = {
                "raw_model_mae": model_mae_fund,
                "best_overall_baseline_mae": float(best_bl_mae),
                "market_baseline_mae": float(market_bl_mae),
                "post_cal_mae": float(mean_absolute_error(y_val, post_cal)),
                "blended_mae": float(best_mae_shrink),
                "market_shrink_cap": float(_MARKET_SHRINK_CAP),
            }
            log.info(
                "Market shrink — fundamentals lose to market_total baseline; w=%.2f (cap=%.2f; post_cal MAE %.3f → blended %.3f)",
                market_shrink,
                _MARKET_SHRINK_CAP,
                market_shrink_diagnostics["post_cal_mae"],
                market_shrink_diagnostics["blended_mae"],
            )

    artifact = {
        "lane": "totals",
        "architecture": "baseline_plus_residual",
        "lane_status": "research_only",
        "model_name": best_name,
        "model_version": artifact_name,
        "trained_at": datetime.now(timezone.utc),
        "field_roles": feature_field_roles("totals"),
        "selected_feature_roles": [FIELD_ROLE_CORE_PREDICTOR],
        "baseline_home_avgs": home_avgs.to_dict(),
        "baseline_away_avgs": away_avgs.to_dict(),
        "baseline_fallback": train_mean,
        "adjustment_columns": adjustment_features,
        "feature_columns": adjustment_features,
        "training_columns": list(X_train.columns),
        "category_columns": category_columns,
        "metrics": metrics,
        "baselines": baselines,
        "calibration_metrics": calibration_metrics,
        "validation_metrics": validation_metrics,
        "residual_std": residual_std if residual_std > 0 else 1.0,
        "market_shrink": market_shrink,
        "market_shrink_diagnostics": market_shrink_diagnostics,
        "market_calibrator": market_calibrator,
        "model": best_model,
    }
    artifact_path = save_artifact("totals", artifact_name, artifact)
    report_payload = {k: v for k, v in artifact.items() if k not in ("model", "market_calibrator")}
    if market_calibrator is not None:
        report_payload["market_calibrator"] = {k: v for k, v in market_calibrator.items() if k != "calibrator"}
    report_path = save_report("totals", artifact_name, report_payload)
    log.info("Saved totals artifact %s and report %s", artifact_path, report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())