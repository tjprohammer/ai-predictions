#!/usr/bin/env python3
"""
Prediction Tracking Report
- Exports a ledger of served predictions for a recent window (no scores to avoid leakage).
- Also exports a separate results file for completed games with outcomes and basic metrics.

Outputs:
- outputs/ledger/predictions_ledger_<ts>.csv           (no scores)
- outputs/ledger/predictions_results_<ts>.csv          (only completed games; with total_runs)
"""
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text


def build_report(days: int, outdir: str):
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    q = text('''
        SELECT 
            game_id,
            "date",
            home_team,
            away_team,
            market_total,
            predicted_total,
            predicted_total_whitelist,
            predicted_total_learning,
            prediction_timestamp,
            total_runs
        FROM enhanced_games
        WHERE "date" BETWEEN :s AND :e
          AND (predicted_total IS NOT NULL OR predicted_total_whitelist IS NOT NULL)
        ORDER BY "date", game_id
    ''')

    df = pd.read_sql(q, engine, params={'s': start_date, 'e': end_date})
    if df.empty:
        print('No served predictions found in window')
        return None

    # Choose published prediction (what UI shows): prefer predicted_total (primary), then whitelist
    pub = pd.to_numeric(df['predicted_total'], errors='coerce')
    wl = pd.to_numeric(df['predicted_total_whitelist'], errors='coerce')
    published = pub.where(pub.notna(), wl)

    # Source tag
    source = np.where(pub.notna(), 'primary', np.where(wl.notna(), 'whitelist', 'unknown'))

    # Predictions-only ledger (no scores)
    ledger = df[['game_id', 'date', 'home_team', 'away_team', 'market_total', 'prediction_timestamp']].copy()
    ledger['predicted_total'] = published
    ledger['predicted_total_primary'] = pub
    ledger['predicted_total_whitelist'] = wl
    ledger['source'] = source

    # Results for completed games (with outcomes, safe for post-game analysis)
    results = ledger.copy()
    results = results.join(pd.to_numeric(df['total_runs'], errors='coerce').rename('total_runs'))
    results = results[results['total_runs'].notna()].copy()

    # Metrics fields
    if not results.empty:
        y = results['total_runs'].astype(float).to_numpy()
        p = results['predicted_total'].astype(float).to_numpy()
        m = pd.to_numeric(df.loc[results.index, 'market_total'], errors='coerce').fillna(np.nan).to_numpy()
        results['abs_error'] = np.abs(p - y)
        results['squared_error'] = (p - y) ** 2
        # Directional accuracy vs market when market is available
        with np.errstate(invalid='ignore'):
            sign_ok = (np.sign(y - m) == np.sign(p - m))
        results['diracc_vs_market'] = np.where(np.isfinite(m), sign_ok, np.nan)

    outdir_path = Path(outdir) / 'ledger'
    outdir_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')

    ledger_path = outdir_path / f'predictions_ledger_{ts}.csv'
    results_path = outdir_path / f'predictions_results_{ts}.csv'
    ledger.to_csv(ledger_path, index=False)
    if not results.empty:
        results.to_csv(results_path, index=False)

    # Print concise summary
    print(f"Window: {start_date} → {end_date}")
    print(f"Ledger rows: {len(ledger)} | Completed with results: {len(results)}")
    if not results.empty:
        mae = results['abs_error'].mean()
        rmse = np.sqrt(results['squared_error'].mean())
        bias = float((results['predicted_total'] - results['total_runs']).mean())
        diracc = results['diracc_vs_market'].mean(skipna=True)
        print(f"MAE={mae:.3f} RMSE={rmse:.3f} Bias={bias:.3f} DirAcc={diracc if diracc==diracc else 'NA'}")
    print('Wrote:')
    print('  ', ledger_path)
    if results_path.exists():
        print('  ', results_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Export last N days of served predictions (no scores) and results for completed games')
    ap.add_argument('--days', type=int, default=10)
    ap.add_argument('--outdir', default='outputs')
    args = ap.parse_args()
    build_report(args.days, args.outdir)
