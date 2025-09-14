#!/usr/bin/env python3
"""Retrain a totals prediction model from historical games.

Goals (covers user request for 'all 3'): 
 1. Retraining: Build a fresh regression model predicting total runs.
 2. Calibration: Learn a linear calibration mapping actual = a + b * raw_pred.
 3. Persistence: Save model + calibration metadata + validation metrics.

Lightweight design (no assumptions about existing feature engineering code):
 - Pulls historical games from enhanced_games table.
 - Minimal features: market_total, (home|away)_team categorical one-hot, month, dayofweek.
 - Target: total_runs.
 - Train/validation split: chronological (last N days for validation, default 30).
 - Model: GradientBoostingRegressor (robust to moderate feature scale, deterministic-ish).
 - Calibration: Ordinary Least Squares (numpy) on validation set predictions.
 - Outputs:
     models/retrained_totals_model_<YYYYMMDD_HHMMSS>.joblib
     models/retrained_totals_model_<timestamp>_metadata.json
     exports/retrained_totals_validation_predictions_<timestamp>.csv
 - Optional: --preview to print top feature importances.

Usage:
  python mlb/training/retrain_totals_model.py --days 180 --val-days 30 --min-games 500

Assumptions:
  - Database table enhanced_games has columns: date (DATE), home_team, away_team, market_total, total_runs.
  - Missing values are dropped.
  - If fewer than --min-games rows after filtering, abort.

Future extensions:
  * Add advanced feature sets (weather, pitcher stats) via JOINs.
  * Hyperparameter search.
  * Probabilistic modeling (distributional outputs).
"""

from __future__ import annotations

import os
import json
import argparse
import math
from datetime import date, datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Optional unified feature builder (brings retraining closer to serving parity)
try:  # pragma: no cover - optional import
    from mlb.features.shared_feature_builder import fetch_training_frame, build_feature_matrix as shared_build
    _SHARED_AVAILABLE = True
except Exception:
    _SHARED_AVAILABLE = False

try:
    from mlb.features.park_factors import get_park_factor
except Exception:  # pragma: no cover - fallback if module path differs
    def get_park_factor(team: str) -> float:  # type: ignore
        return 1.0

DEFAULT_DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")


def fetch_history(engine, days: int) -> pd.DataFrame:
    start_date = date.today() - timedelta(days=days)
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, home_team, away_team, market_total, total_runs
                FROM enhanced_games
                WHERE date >= :start
                  AND total_runs IS NOT NULL
                  AND market_total IS NOT NULL
                ORDER BY date
                """
            ),
            conn,
            params={"start": start_date},
        )
    return df


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    work = df.copy()
    work["month"] = pd.to_datetime(work["date"]).dt.month
    work["dow"] = pd.to_datetime(work["date"]).dt.dayofweek
    # Park factor (based on home team park)
    work["park_factor"] = work["home_team"].map(lambda t: get_park_factor(str(t)))
    # One-hot encode teams (combined to keep columns manageable)
    teams = pd.concat([work["home_team"], work["away_team"]]).unique()
    # Encode home & away separately
    for t in teams:
        work[f"home_{t}"] = (work["home_team"] == t).astype(int)
        work[f"away_{t}"] = (work["away_team"] == t).astype(int)
    feature_cols = [c for c in work.columns if c not in {"date", "total_runs", "home_team", "away_team"}]
    X = work[feature_cols].astype(float)
    y = work["total_runs"].astype(float)
    return X, y


def chronological_split(X: pd.DataFrame, y: pd.Series, dates: pd.Series, val_days: int):
    cutoff = dates.max() - timedelta(days=val_days - 1)
    train_mask = dates < cutoff
    val_mask = dates >= cutoff
    return X[train_mask], X[val_mask], y[train_mask], y[val_mask]


def linear_calibration(raw_pred: np.ndarray, actual: np.ndarray) -> Tuple[float, float]:
    # Fit a + b*x using least squares.
    X = np.vstack([np.ones_like(raw_pred), raw_pred]).T
    coef, *_ = np.linalg.lstsq(X, actual, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a, b


def main():  # pragma: no cover
    ap = argparse.ArgumentParser(description="Retrain totals model with calibration")
    ap.add_argument("--db", default=DEFAULT_DB, help="Database URL")
    ap.add_argument("--days", type=int, default=365, help="Number of past days of data to use (default 365)")
    ap.add_argument("--val-days", type=int, default=30, help="Validation window size in days (chronological tail, default 30)")
    ap.add_argument("--min-games", type=int, default=400, help="Minimum games required to proceed")
    ap.add_argument("--learning-rate", type=float, default=0.05, help="GBR learning rate")
    ap.add_argument("--n-estimators", type=int, default=500, help="GBR number of estimators")
    ap.add_argument("--max-depth", type=int, default=3, help="GBR max depth (via max_depth of individual trees)")
    ap.add_argument("--preview", action="store_true", help="Print top feature importances")
    ap.add_argument("--use-shared", action="store_true", help="Use shared feature builder (if available) for richer feature set")
    args = ap.parse_args()

    engine = create_engine(args.db, pool_pre_ping=True)
    # If shared builder requested and available, pull via shared path (superset of columns)
    if args.use_shared and _SHARED_AVAILABLE:
        df = fetch_training_frame(engine, args.days)
    else:
        df = fetch_history(engine, args.days)
    if df.empty:
        print("No historical data fetched; aborting.")
        return 1
    if len(df) < args.min_games:
        print(f"Only {len(df)} games (< min {args.min_games}); aborting.")
        return 1

    if args.use_shared and _SHARED_AVAILABLE:
        X, y = shared_build(df)
    else:
        X, y = build_features(df)
        if args.use_shared and not _SHARED_AVAILABLE:
            print("[WARN] --use-shared specified but shared_feature_builder not available; fell back to minimal features.")
    dates = pd.to_datetime(df["date"]).dt.date
    X_train, X_val, y_train, y_val = chronological_split(X, y, dates, args.val_days)
    if X_val.empty or X_train.empty:
        print("Invalid split (empty train or val). Reduce --val-days or increase --days.")
        return 1

    model = GradientBoostingRegressor(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(X_train, y_train)

    val_raw = model.predict(X_val)
    train_raw = model.predict(X_train)
    # Calibration on validation set
    a, b = linear_calibration(val_raw, y_val.values)
    val_cal = a + b * val_raw
    train_cal = a + b * train_raw

    def metrics(prefix: str, actual, pred):
        mae = mean_absolute_error(actual, pred)
        # Support older sklearn that lacks squared parameter
        mse = mean_squared_error(actual, pred)
        rmse = math.sqrt(mse)
        bias = float((pred - actual).mean())
        return {f"{prefix}_mae": mae, f"{prefix}_rmse": rmse, f"{prefix}_bias": bias}

    m = {}
    m.update(metrics("train_raw", y_train, train_raw))
    m.update(metrics("val_raw", y_val, val_raw))
    m.update(metrics("train_cal", y_train, train_cal))
    m.update(metrics("val_cal", y_val, val_cal))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", f"retrained_totals_model_{timestamp}.joblib")
    joblib.dump(model, model_path)

    metadata = {
        "timestamp": timestamp,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "days": args.days,
        "val_days": args.val_days,
        "model_type": "GradientBoostingRegressor",
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "calibration": {"a": a, "b": b},
        "metrics": m,
        "feature_columns": list(X.columns),
        "feature_builder": "shared" if (args.use_shared and _SHARED_AVAILABLE) else "minimal",
    }
    meta_path = model_path.replace(".joblib", "_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Export validation predictions for inspection
    os.makedirs("exports", exist_ok=True)
    val_export = pd.DataFrame({
        "date": df.iloc[X_val.index]["date"].values,
        "actual": y_val.values,
        "raw_pred": val_raw,
        "cal_pred": val_cal,
    })
    val_path = os.path.join("exports", f"retrained_totals_validation_predictions_{timestamp}.csv")
    val_export.to_csv(val_path, index=False)

    print("Retraining complete.")
    print(f"Model saved -> {model_path}")
    print(f"Metadata saved -> {meta_path}")
    print(f"Validation preds -> {val_path}")
    print("Validation raw vs calibrated MAE / Bias:")
    print(f"  Raw  MAE {m['val_raw_mae']:.3f}  Bias {m['val_raw_bias']:+.3f}")
    print(f"  Cal  MAE {m['val_cal_mae']:.3f}  Bias {m['val_cal_bias']:+.3f}")
    if args.preview and hasattr(model, "feature_importances_"):
        fi = sorted(zip(X.columns, model.feature_importances_), key=lambda z: z[1], reverse=True)[:15]
        print("Top feature importances:")
        for name, val in fi:
            print(f"  {name:25s} {val:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
