"""Experiment: predict home runs and away runs separately, then combine.

Phase 2 of the Totals Improvement Plan.  Tests whether predicting each side's
scoring independently yields a better total than predicting the combined
total directly.

Usage:
    python -m src.backtest.experiment_split_sides
"""
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
    TOTALS_HOME_RUNS_COLUMN,
    TOTALS_AWAY_RUNS_COLUMN,
    TOTALS_TARGET_COLUMN,
    feature_columns_for_roles,
)
from src.models.common import (
    chronological_split,
    compute_sample_weights,
    encode_frame,
    load_feature_snapshots,
)
from src.models.train_totals import compute_baseline
from src.utils.logging import get_logger


log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Side-specific feature columns
# ---------------------------------------------------------------------------
# Home side: features about the home team's offense, the away starter they
# face, and the away bullpen they'll exploit.
HOME_FEATURES = [
    "home_runs_rate_blended",
    "home_xwoba_blended",
    "home_iso_blended",
    "home_lineup_top5_xwoba",
    "home_lineup_k_pct",
    "away_starter_xwoba_blended",    # pitcher they face
    "away_starter_csw_blended",
    "away_starter_rest_days",
    "away_bullpen_era_last3",
    "away_bullpen_pitches_last3",
    "away_bullpen_b2b",
    "venue_run_factor",
    "venue_hr_factor",
]

AWAY_FEATURES = [
    "away_runs_rate_blended",
    "away_xwoba_blended",
    "away_iso_blended",
    "away_lineup_top5_xwoba",
    "away_lineup_k_pct",
    "home_starter_xwoba_blended",    # pitcher they face
    "home_starter_csw_blended",
    "home_starter_rest_days",
    "home_bullpen_era_last3",
    "home_bullpen_pitches_last3",
    "home_bullpen_b2b",
    "venue_run_factor",
    "venue_hr_factor",
]


def _train_side(
    name: str,
    features: list[str],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    target_col: str,
) -> dict:
    """Train candidates for one side (home or away), return best."""
    y_train = train_frame[target_col].astype(float)
    y_val = val_frame[target_col].astype(float)

    available = [c for c in features if c in train_frame.columns]
    cat_cols: list[str] = []
    X_train = encode_frame(train_frame[available], cat_cols)
    X_val = encode_frame(val_frame[available], cat_cols, training_columns=list(X_train.columns))

    weights = compute_sample_weights(train_frame["game_date"])

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
    }

    best_name, best_model, best_mae = None, None, math.inf
    best_preds = None
    for cname, model in candidates.items():
        if hasattr(model, "steps"):
            last = model.steps[-1][0]
            model.fit(X_train, y_train, **{f"{last}__sample_weight": weights})
        else:
            model.fit(X_train, y_train, sample_weight=weights)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = math.sqrt(mean_squared_error(y_val, preds))
        log.info("  %s %s: MAE=%.3f RMSE=%.3f", name, cname, mae, rmse)
        if mae < best_mae:
            best_name, best_model, best_mae = cname, model, mae
            best_preds = preds

    return {
        "name": best_name,
        "model": best_model,
        "mae": best_mae,
        "predictions": best_preds,
        "features": available,
    }


def _train_side_with_baseline(
    name: str,
    features: list[str],
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    target_col: str,
    team_col: str,
) -> dict:
    """Train candidates using team-average baseline + residual for one side."""
    y_train = train_frame[target_col].astype(float)
    y_val = val_frame[target_col].astype(float)
    train_mean = float(y_train.mean())

    # Per-team baseline
    team_avgs = train_frame.groupby(team_col)[target_col].mean()
    baseline_train = train_frame[team_col].map(team_avgs).fillna(train_mean).astype(float).values
    baseline_val = val_frame[team_col].map(team_avgs).fillna(train_mean).astype(float).values
    residual_train = y_train.values - baseline_train
    residual_val = y_val.values - baseline_val

    baseline_mae = mean_absolute_error(y_val, baseline_val)
    log.info("  %s team-avg baseline MAE=%.3f", name, baseline_mae)

    available = [c for c in features if c in train_frame.columns]
    cat_cols: list[str] = []
    X_train = encode_frame(train_frame[available], cat_cols)
    X_val = encode_frame(val_frame[available], cat_cols, training_columns=list(X_train.columns))

    weights = compute_sample_weights(train_frame["game_date"])

    candidates = {
        "ridge": Ridge(alpha=1.0),
        "elasticnet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000)),
        "gbr": GradientBoostingRegressor(random_state=42, learning_rate=0.05, n_estimators=300, max_depth=3),
    }

    best_name, best_mae = None, math.inf
    best_preds = None
    for cname, model in candidates.items():
        if hasattr(model, "steps"):
            last = model.steps[-1][0]
            model.fit(X_train, residual_train, **{f"{last}__sample_weight": weights})
        else:
            model.fit(X_train, residual_train, sample_weight=weights)
        residual_pred = model.predict(X_val)
        final = baseline_val + residual_pred
        mae = mean_absolute_error(y_val, final)
        rmse = math.sqrt(mean_squared_error(y_val, final))
        log.info("  %s baseline+%s: MAE=%.3f RMSE=%.3f", name, cname, mae, rmse)
        if mae < best_mae:
            best_name, best_mae = cname, mae
            best_preds = final

    return {
        "name": best_name,
        "mae": best_mae,
        "predictions": best_preds,
        "baseline_mae": baseline_mae,
        "features": available,
    }


def main() -> int:
    frame = load_feature_snapshots("totals")
    if frame.empty:
        log.info("No totals feature snapshots")
        return 0

    # Need both side targets
    mask = (
        frame[TOTALS_TARGET_COLUMN].notna()
        & frame[TOTALS_HOME_RUNS_COLUMN].notna()
        & frame[TOTALS_AWAY_RUNS_COLUMN].notna()
    )
    trainable = frame[mask].copy()
    log.info("Trainable rows with home/away splits: %d", len(trainable))
    if trainable.empty:
        return 0

    train_frame, val_frame = chronological_split(trainable)
    if val_frame.empty:
        log.info("Not enough rows for val split")
        return 0

    y_val_total = val_frame[TOTALS_TARGET_COLUMN].astype(float)
    y_val_home = val_frame[TOTALS_HOME_RUNS_COLUMN].astype(float)
    y_val_away = val_frame[TOTALS_AWAY_RUNS_COLUMN].astype(float)

    train_mean_total = float(train_frame[TOTALS_TARGET_COLUMN].astype(float).mean())
    train_median_total = float(train_frame[TOTALS_TARGET_COLUMN].astype(float).median())

    log.info("Split: %d train, %d val", len(train_frame), len(val_frame))
    log.info("Val date range: %s → %s", val_frame["game_date"].min(), val_frame["game_date"].max())

    # --- Reference baselines (on total) ---
    ref_mean_mae = mean_absolute_error(y_val_total, [train_mean_total] * len(y_val_total))
    ref_median_mae = mean_absolute_error(y_val_total, [train_median_total] * len(y_val_total))

    # team-average total baseline
    home_avgs_total = train_frame.groupby("home_team")[TOTALS_TARGET_COLUMN].mean()
    away_avgs_total = train_frame.groupby("away_team")[TOTALS_TARGET_COLUMN].mean()
    team_avg_total_preds = compute_baseline(val_frame, home_avgs_total, away_avgs_total, train_mean_total)
    ref_team_avg_mae = mean_absolute_error(y_val_total, team_avg_total_preds)

    log.info("=" * 60)
    log.info("REFERENCE BASELINES (total runs)")
    log.info("  train_mean:    MAE=%.3f", ref_mean_mae)
    log.info("  train_median:  MAE=%.3f", ref_median_mae)
    log.info("  team_average:  MAE=%.3f", ref_team_avg_mae)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # APPROACH A: Direct model (separate home + away, combine)
    # ------------------------------------------------------------------
    log.info("\n--- APPROACH A: Direct side models ---")
    home_direct = _train_side("HOME", HOME_FEATURES, train_frame, val_frame, TOTALS_HOME_RUNS_COLUMN)
    away_direct = _train_side("AWAY", AWAY_FEATURES, train_frame, val_frame, TOTALS_AWAY_RUNS_COLUMN)
    combined_direct = home_direct["predictions"] + away_direct["predictions"]
    direct_mae = mean_absolute_error(y_val_total, combined_direct)
    direct_rmse = math.sqrt(mean_squared_error(y_val_total, combined_direct))
    log.info("  HOME best: %s MAE=%.3f | AWAY best: %s MAE=%.3f", home_direct["name"], home_direct["mae"], away_direct["name"], away_direct["mae"])
    log.info("  COMBINED (home+away) MAE=%.3f RMSE=%.3f", direct_mae, direct_rmse)

    # ------------------------------------------------------------------
    # APPROACH B: Baseline+residual per side, combine
    # ------------------------------------------------------------------
    log.info("\n--- APPROACH B: Team-avg baseline + residual per side ---")
    home_bl = _train_side_with_baseline(
        "HOME", HOME_FEATURES, train_frame, val_frame,
        TOTALS_HOME_RUNS_COLUMN, "home_team",
    )
    away_bl = _train_side_with_baseline(
        "AWAY", AWAY_FEATURES, train_frame, val_frame,
        TOTALS_AWAY_RUNS_COLUMN, "away_team",
    )
    combined_bl = home_bl["predictions"] + away_bl["predictions"]
    bl_mae = mean_absolute_error(y_val_total, combined_bl)
    bl_rmse = math.sqrt(mean_squared_error(y_val_total, combined_bl))
    log.info("  HOME baseline+best MAE=%.3f | AWAY baseline+best MAE=%.3f", home_bl["mae"], away_bl["mae"])
    log.info("  COMBINED (baseline+residual) MAE=%.3f RMSE=%.3f", bl_mae, bl_rmse)

    # ------------------------------------------------------------------
    # APPROACH C: Team-avg-per-side baseline only (no residual model)
    # ------------------------------------------------------------------
    log.info("\n--- APPROACH C: Team-avg-per-side baseline only ---")
    train_mean_home = float(train_frame[TOTALS_HOME_RUNS_COLUMN].astype(float).mean())
    train_mean_away = float(train_frame[TOTALS_AWAY_RUNS_COLUMN].astype(float).mean())
    home_team_avgs = train_frame.groupby("home_team")[TOTALS_HOME_RUNS_COLUMN].mean()
    away_team_avgs = train_frame.groupby("away_team")[TOTALS_AWAY_RUNS_COLUMN].mean()
    home_baseline_preds = val_frame["home_team"].map(home_team_avgs).fillna(train_mean_home).astype(float).values
    away_baseline_preds = val_frame["away_team"].map(away_team_avgs).fillna(train_mean_away).astype(float).values
    combined_side_baseline = home_baseline_preds + away_baseline_preds
    side_baseline_mae = mean_absolute_error(y_val_total, combined_side_baseline)
    side_baseline_rmse = math.sqrt(mean_squared_error(y_val_total, combined_side_baseline))
    log.info("  HOME-only baseline MAE=%.3f", mean_absolute_error(y_val_home, home_baseline_preds))
    log.info("  AWAY-only baseline MAE=%.3f", mean_absolute_error(y_val_away, away_baseline_preds))
    log.info("  COMBINED (side baselines) MAE=%.3f RMSE=%.3f", side_baseline_mae, side_baseline_rmse)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("EXPERIMENT SUMMARY — Split-Side vs Unified Total")
    log.info("=" * 60)
    results = [
        ("train_mean (ref)", ref_mean_mae),
        ("train_median (ref)", ref_median_mae),
        ("team_average total (ref)", ref_team_avg_mae),
        ("side team-avg baselines", side_baseline_mae),
        ("direct side models", direct_mae),
        ("baseline+residual sides", bl_mae),
    ]
    results.sort(key=lambda x: x[1])
    for name, mae in results:
        marker = " <-- TARGET TO BEAT" if name == "train_median (ref)" else ""
        marker = " <-- CURRENT BEST MODEL" if name == "team_average total (ref)" else marker
        log.info("  %-30s MAE=%.3f%s", name, mae, marker)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
