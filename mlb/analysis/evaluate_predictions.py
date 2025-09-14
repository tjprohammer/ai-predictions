#!/usr/bin/env python3
"""Evaluate prediction accuracy over a historical date range.

Usage examples:
  python mlb/analysis/evaluate_predictions.py --start 2025-08-01 --end 2025-09-02 \
      --cols predicted_total,market_total,predicted_total_original

Adds common metrics:
  - MAE, RMSE
  - Mean Bias (pred - actual)
  - Std of Error
  - % within 0.5 / 1.0 runs
  - Directional over/under accuracy vs market_total (if both available)
  - Error distribution percentiles
  - Spearman correlation between prediction & actual total_runs

Optional: write a CSV of per-game errors.
"""
import os
import argparse
from typing import List
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from scipy.stats import spearmanr

DEFAULT_DB = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")

METRIC_COLS = [
    "MAE","RMSE","MeanBias","ErrorStd","Within0.5","Within1.0","DirectionalAcc","SpearmanR","N"
]

def fetch_games(engine, start, end, cols: List[str]):
    cols_set = set(cols)
    needed = {"game_id","date","total_runs"} | cols_set | {"market_total"}
    sel = ",".join(sorted(needed))
    q = text(f"""
        SELECT {sel}
        FROM enhanced_games
        WHERE date BETWEEN :start AND :end
          AND total_runs IS NOT NULL -- completed games only
        ORDER BY date, game_time_utc NULLS LAST, game_id
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"start": start, "end": end})
    return df


def compute_metrics(df: pd.DataFrame, pred_col: str):
    actual = pd.to_numeric(df["total_runs"], errors="coerce")
    preds = pd.to_numeric(df[pred_col], errors="coerce")
    mask = actual.notna() & preds.notna()
    actual = actual[mask]
    preds = preds[mask]
    if len(actual) == 0:
        return {c: np.nan for c in METRIC_COLS}
    err = preds - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    bias = float(np.mean(err))
    estd = float(np.std(err, ddof=0))
    within05 = float(np.mean(np.abs(err) <= 0.5))
    within10 = float(np.mean(np.abs(err) <= 1.0))
    # Directional accuracy relative to market (pred - market vs actual - market)
    dir_acc = np.nan
    if "market_total" in df.columns and df["market_total"].notna().any():
        mkt = pd.to_numeric(df.loc[mask, "market_total"], errors="coerce")
        market_mask = mkt.notna()
        if market_mask.any():
            pred_dir = np.sign(preds[market_mask] - mkt[market_mask])
            actual_dir = np.sign(actual[market_mask] - mkt[market_mask])
            # Ignore ties where difference == 0
            valid = (pred_dir != 0) & (actual_dir != 0)
            if valid.any():
                dir_acc = float(np.mean(pred_dir[valid] == actual_dir[valid]))
    spear = np.nan
    try:
        spear, _ = spearmanr(preds, actual)
    except Exception:
        pass
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MeanBias": bias,
        "ErrorStd": estd,
        "Within0.5": within05,
        "Within1.0": within10,
        "DirectionalAcc": dir_acc,
        "SpearmanR": spear,
        "N": int(len(actual))
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    ap.add_argument("--cols", default="predicted_total,market_total", help="Comma list of prediction columns to evaluate")
    ap.add_argument("--db", default=DEFAULT_DB, help="Database URL")
    ap.add_argument("--per_game_csv", help="Optional path to write per-game error CSV")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(',') if c.strip()]

    print(f"Evaluating columns: {cols}")
    engine = create_engine(args.db)
    df = fetch_games(engine, args.start, args.end, cols)
    print(f"Loaded {len(df)} completed games")

    metrics_rows = []
    for col in cols:
        if col not in df.columns:
            print(f"WARNING: Column {col} not in dataset, skipping")
            continue
        m = compute_metrics(df, col)
        metrics_rows.append((col, m))

    # Output metrics
    if not metrics_rows:
        print("No metrics computed.")
        return

    # Determine best (lowest MAE) among evaluated
    best = min(metrics_rows, key=lambda x: x[1]['MAE'] if not np.isnan(x[1]['MAE']) else 1e9)

    print("\n=== Prediction Performance ===")
    header = ["Column"] + METRIC_COLS
    print("\t".join(header))
    for col, m in metrics_rows:
        mark = "*" if col == best[0] else ""
        print("\t".join([
            f"{col}{mark}",
            *(f"{m[k]:.3f}" if isinstance(m[k], (float,int)) and not np.isnan(m[k]) else "nan" for k in METRIC_COLS)
        ]))

    # Optional per-game CSV
    if args.per_game_csv:
        rows = []
        for col in cols:
            if col in df.columns:
                err = pd.to_numeric(df[col], errors='coerce') - pd.to_numeric(df['total_runs'], errors='coerce')
                rows.append(df[['game_id','date','total_runs']].assign(pred_col=col, prediction=df[col], error=err))
        if rows:
            out = pd.concat(rows)
            out.to_csv(args.per_game_csv, index=False)
            print(f"Wrote per-game errors to {args.per_game_csv}")

    print(f"\nBest MAE: {best[0]} = {best[1]['MAE']:.3f}")

if __name__ == "__main__":
    main()
