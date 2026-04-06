from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
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
from src.models.common import chronological_split, compute_sample_weights, encode_frame, fit_market_calibrator, load_feature_snapshots, save_artifact, save_report
from src.utils.logging import get_logger
from src.utils.settings import get_settings


log = get_logger(__name__)


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
        metrics[name] = {"mae": mae, "rmse": rmse, "residual_mae": residual_mae}
        log.info("  %s: MAE=%.3f RMSE=%.3f residual_MAE=%.3f", name, mae, rmse, residual_mae)
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

    artifact = {
        "lane": "totals",
        "architecture": "baseline_plus_residual",
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
        "residual_std": residual_std if residual_std > 0 else 1.0,
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