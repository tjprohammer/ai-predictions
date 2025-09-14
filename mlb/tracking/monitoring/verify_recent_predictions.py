#!/usr/bin/env python3
"""
Verify Recent MLB Totals Predictions
====================================

Purpose:
  Quickly inspect how well stored model predictions matched actual game totals
  over the last N days (default 7) directly from the `enhanced_games` table.

What it does:
  * Pulls recent games with actual results (total_runs not null)
  * Collects available prediction columns:
        - predicted_total (Learning Adaptive / Enhanced Bullpen)
        - predicted_total_learning (Ultra 80 Incremental)
        - predicted_total_whitelist (Whitelist pruned model, if present)
  * Computes error metrics per model (MAE, RMSE, Bias, Corr)
  * Compares directional OVER/UNDER signal vs market_total (if present)
  * Evaluates “recommendations” when |prediction - market| >= edge threshold
  * Saves a CSV snapshot in exports/ for audit & trust building

Usage (examples):
  python -m mlb.tracking.monitoring.verify_recent_predictions --days 7 --all
  python mlb/tracking/monitoring/verify_recent_predictions.py --days 10 --model predicted_total
  python mlb/tracking/monitoring/verify_recent_predictions.py --days 14 --edge-threshold 0.5 --all

Return codes:
  0 = success, 1 = no data / error.
"""

import os
import math
import argparse
from datetime import date, timedelta
from typing import List, Dict

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


DEFAULT_DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
PREDICTION_COLUMNS = [
    "predicted_total",              # Learning Adaptive
    "predicted_total_learning",     # Ultra 80 Incremental
    "predicted_total_whitelist"     # Whitelist pruned model
]


def fetch_recent(engine, days: int) -> pd.DataFrame:
    end_d = date.today() - timedelta(days=0)  # include today if results already populated
    start_d = end_d - timedelta(days=days - 1)
    with engine.begin() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT date, game_id, home_team, away_team,
                       market_total, total_runs,
                       predicted_total, predicted_total_learning, predicted_total_whitelist,
                       over_odds, under_odds
                FROM enhanced_games
                WHERE date BETWEEN :s AND :e
                  AND total_runs IS NOT NULL
                  AND (predicted_total IS NOT NULL
                       OR predicted_total_learning IS NOT NULL
                       OR predicted_total_whitelist IS NOT NULL)
                ORDER BY date DESC, game_id
                """
            ),
            conn,
            params={"s": start_d, "e": end_d},
        )
    return df


def _nanaware_rmse(err: pd.Series) -> float:
    v = err.dropna()
    return float(math.sqrt((v**2).mean())) if len(v) else float("nan")


def evaluate_model(df: pd.DataFrame, col: str, edge_threshold: float) -> Dict[str, float]:
    pred = df[col]
    actual = df["total_runs"]
    err = pred - actual
    abs_err = err.abs()
    mae = abs_err.mean()
    rmse = _nanaware_rmse(err)
    bias = err.mean()  # positive => over-predicting
    corr = pred.corr(actual)

    # Directional vs actual relative to market line (if market available)
    directional_accuracy = None
    rec_accuracy = None
    rec_n = 0
    coverage_n = len(df)

    if "market_total" in df.columns and df["market_total"].notna().any():
        mt = df["market_total"]
        # What actually happened vs line (OVER(>), UNDER(<), PUSH(=))
        actual_sign = np.sign(actual - mt)
        model_sign = np.sign(pred - mt)
        # Exclude pushes (0) from directional comparison
        mask_dir = (actual_sign != 0) & (model_sign != 0)
        if mask_dir.any():
            directional_accuracy = float((actual_sign[mask_dir] == model_sign[mask_dir]).mean())

        # Recommendation set based on edge threshold
        rec_mask = (pred - mt).abs() >= edge_threshold
        rec_n = int(rec_mask.sum())
        if rec_n > 0:
            rec_actual_sign = actual_sign[rec_mask]
            rec_model_sign = model_sign[rec_mask]
            valid = (rec_actual_sign != 0) & (rec_model_sign != 0)
            if valid.any():
                rec_accuracy = float((rec_actual_sign[valid] == rec_model_sign[valid]).mean())

    return {
        "model": col,
        "n": coverage_n,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "corr": corr,
        "dir_acc": directional_accuracy,
        "rec_threshold": edge_threshold,
        "rec_n": rec_n,
        "rec_acc": rec_accuracy,
    }


def summarize_day(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["error"] = (tmp[col] - tmp["total_runs"]).abs()
    return (
        tmp.groupby("date").agg(
            games=("game_id", "count"),
            avg_pred=(col, "mean"),
            avg_actual=("total_runs", "mean"),
            mae=("error", "mean"),
        ).round(2)
    )


def print_metrics(metrics: Dict[str, float]):
    def fmt(v, p=2):
        return "NA" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{p}f}"
    print(f"\nModel: {metrics['model']}")
    print("-" * (8 + len(metrics['model'])))
    print(f"Games: {metrics['n']}")
    print(f"MAE:   {fmt(metrics['mae'])}  | RMSE: {fmt(metrics['rmse'])}  | Bias (pred-actual): {fmt(metrics['bias'])}")
    print(f"Corr (Pred vs Actual): {fmt(metrics['corr'])}")
    if metrics['dir_acc'] is not None:
        print(f"Directional accuracy vs market (non-push): {metrics['dir_acc']*100:.1f}%")
    if metrics['rec_n'] > 0:
        ra = metrics['rec_acc']
        ra_str = "NA" if ra is None else f"{ra*100:.1f}%"
        print(f"Recommendation set (|Δ|≥{metrics['rec_threshold']}): {metrics['rec_n']} picks | Accuracy: {ra_str}")


def save_snapshot(df: pd.DataFrame, models: List[str]):
    from datetime import datetime as dt
    snap_cols = ["date", "game_id", "away_team", "home_team", "market_total", "total_runs"]
    for m in models:
        if m in df.columns:
            snap_cols.append(m)
    snap_cols = [c for c in snap_cols if c in df.columns]
    out = df[snap_cols].sort_values(["date", "game_id"]).copy()
    out_path = os.path.join("exports", f"recent_prediction_eval_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
    os.makedirs("exports", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nSaved detailed snapshot: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Verify recent totals predictions vs actual outcomes")
    ap.add_argument("--days", type=int, default=7, help="Number of recent days to evaluate (default 7)")
    ap.add_argument("--model", choices=PREDICTION_COLUMNS, default="predicted_total", help="Single model column to evaluate")
    ap.add_argument("--all", action="store_true", help="Evaluate all available model columns")
    ap.add_argument("--edge-threshold", type=float, default=0.75, help="Edge threshold (runs) for recommendation evaluation (default 0.75)")
    ap.add_argument("--db", default=DEFAULT_DB, help="Database URL override")
    args = ap.parse_args()

    print("\n🔍 Verifying recent model predictions vs actual results")
    print(f"Window: last {args.days} days (including today if results exist)")
    print(f"Edge threshold: ±{args.edge_threshold} runs")

    engine = create_engine(args.db, pool_pre_ping=True)
    df = fetch_recent(engine, args.days)
    if df.empty:
        print("No recent finished games with predictions found. Exiting.")
        return 1

    print(f"Loaded {len(df)} games spanning {df['date'].min()} to {df['date'].max()}")
    available_models = [c for c in PREDICTION_COLUMNS if c in df.columns and df[c].notna().any()]
    if not available_models:
        print("No prediction columns populated in this window.")
        return 1

    models_to_eval = available_models if args.all else [m for m in [args.model] if m in available_models]
    if not models_to_eval:
        print(f"Requested model {args.model} not present. Available: {available_models}")
        return 1

    for m in models_to_eval:
        sub = df[df[m].notna()].copy()
        metrics = evaluate_model(sub, m, args.edge_threshold)
        print_metrics(metrics)
        day_summary = summarize_day(sub, m)
        print("Daily breakdown (date | games | MAE | avg_pred | avg_actual):")
        for d, row in day_summary.iterrows():
            print(f"  {d} | {int(row.games):2d} | MAE={row.mae:.2f} | Pred={row.avg_pred:.2f} | Actual={row.avg_actual:.2f}")

    save_snapshot(df, models_to_eval)
    print("\nDone.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
