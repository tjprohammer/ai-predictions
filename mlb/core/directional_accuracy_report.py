#!/usr/bin/env python3
"""
Directional Accuracy & Error Decomposition
=========================================
Computes:
  - MAE / RMSE vs actual for whitelist baseline predictions (if existing)
  - Market MAE / RMSE baseline
  - Directional accuracy (pred vs market vs actual outcome)
  - Error buckets (abs error bins)
Pulls most recent N completed days.
"""
import os, json
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_absolute_error, mean_squared_error

DB = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
ENGINE = create_engine(DB)
DAYS = int(os.getenv('DIRACC_DAYS','14'))
TODAY = datetime.utcnow().date()
START = (TODAY - timedelta(days=DAYS)).strftime('%Y-%m-%d')

q = text("""
    SELECT date, game_id, home_team, away_team, total_runs, market_total,
           predicted_total, predicted_total_learning, predicted_total_original
    FROM enhanced_games
    WHERE date BETWEEN :start AND :today
      AND total_runs IS NOT NULL
""")

df = pd.read_sql(q, ENGINE, params={'start': START, 'today': TODAY})
if df.empty:
    raise SystemExit('No completed games in window')

# Pick a primary model column (use predicted_total if present else predicted_total_learning)
if df['predicted_total'].notna().any():
    primary_col = 'predicted_total'
elif 'predicted_total_learning' in df.columns and df['predicted_total_learning'].notna().any():
    primary_col = 'predicted_total_learning'
else:
    primary_col = None

market = df['market_total']
actual = df['total_runs']

summary = {}
if primary_col:
    pred = df[primary_col]
    mask = pred.notna()
    summary['model_MAE'] = float(mean_absolute_error(actual[mask], pred[mask]))
    summary['model_RMSE'] = float(mean_squared_error(actual[mask], pred[mask], squared=False))
    summary['model_bias'] = float((pred[mask] - actual[mask]).mean())

if market.notna().any():
    m_mask = market.notna()
    summary['market_MAE'] = float(mean_absolute_error(actual[m_mask], market[m_mask]))
    summary['market_RMSE'] = float(mean_squared_error(actual[m_mask], market[m_mask], squared=False))
    summary['market_bias'] = float((market[m_mask] - actual[m_mask]).mean())

# Directional accuracy: sign(pred - market) vs sign(actual - market)
if primary_col and market.notna().any():
    pred = df[primary_col]
    dir_df = df.loc[pred.notna() & market.notna()].copy()
    dir_df['pred_side'] = np.sign(dir_df[primary_col] - dir_df['market_total'])
    dir_df['actual_side'] = np.sign(dir_df['total_runs'] - dir_df['market_total'])
    # Treat small edges as 0 (hold)
    edge_thresh = float(os.getenv('DIRACC_EDGE_THRESH','0.15'))
    dir_df.loc[(dir_df[primary_col] - dir_df['market_total']).abs() < edge_thresh, 'pred_side'] = 0
    dir_df.loc[(dir_df['total_runs'] - dir_df['market_total']).abs() < edge_thresh, 'actual_side'] = 0
    summary['directional_accuracy'] = float((dir_df['pred_side'] == dir_df['actual_side']).mean())

# Error buckets
if primary_col:
    pred = df[primary_col]
    err = (pred - actual).abs()
    bins = [0,0.5,1.0,1.5,2.0,3.0,5.0,10.0]
    cats = pd.cut(err, bins=bins, include_lowest=True)
    bucket = cats.value_counts(normalize=True).sort_index()
    summary['error_buckets'] = {str(k): round(v,3) for k,v in bucket.items()}

summary['games_evaluated'] = int(len(df))
summary['window_days'] = DAYS

OUT = Path(__file__).parent / 'outputs' / 'directional_accuracy_summary.json'
OUT.parent.mkdir(exist_ok=True, parents=True)
OUT.write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
