#!/usr/bin/env python3
"""Daily metrics logger.

Computes prediction performance for completed games for a trailing window
and stores aggregated metrics in prediction_metrics_daily table.

Usage:
  python mlb/analysis/log_daily_metrics.py --date 2025-09-02 \
      --lookback 30 --cols predicted_total,predicted_total_original,market_total

If the table does not exist it will be created.

Columns stored:
  date (the end date evaluated)
  metric_date_start / metric_date_end (window)
  pred_column
  mae, rmse, bias, error_std, within_0_5, within_1_0,
  directional_acc, spearman_r, n, best_mae_flag
  created_at
"""
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from scipy.stats import spearmanr

DB_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

METRICS = [
    'mae','rmse','bias','error_std','within_0_5','within_1_0','directional_acc','spearman_r','n'
]

def ensure_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS prediction_metrics_daily (
        id SERIAL PRIMARY KEY,
        date date NOT NULL,
        metric_date_start date NOT NULL,
        metric_date_end date NOT NULL,
        pred_column text NOT NULL,
        mae double precision,
        rmse double precision,
        bias double precision,
        error_std double precision,
        within_0_5 double precision,
        within_1_0 double precision,
        directional_acc double precision,
        spearman_r double precision,
        n integer,
        best_mae_flag boolean DEFAULT false,
        created_at timestamp DEFAULT now(),
        UNIQUE(date, pred_column, metric_date_start, metric_date_end)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))

def fetch_games(engine, start, end, columns):
    sel_cols = { 'game_id','date','total_runs','market_total' } | set(columns)
    sel = ",".join(sorted(sel_cols))
    q = text(f"""SELECT {sel} FROM enhanced_games
                 WHERE date BETWEEN :s AND :e AND total_runs IS NOT NULL
                 ORDER BY date, game_time_utc NULLS LAST""")
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={'s': start, 'e': end})

def compute(df, pred_col):
    if pred_col not in df.columns:
        return None
    actual = pd.to_numeric(df['total_runs'], errors='coerce')
    pred = pd.to_numeric(df[pred_col], errors='coerce')
    mask = actual.notna() & pred.notna()
    actual, pred = actual[mask], pred[mask]
    if len(actual) == 0:
        return None
    err = pred - actual
    out = {
        'mae': float(np.mean(np.abs(err))),
        'rmse': float(np.sqrt(np.mean(err**2))),
        'bias': float(np.mean(err)),
        'error_std': float(np.std(err, ddof=0)),
        'within_0_5': float(np.mean(np.abs(err) <= 0.5)),
        'within_1_0': float(np.mean(np.abs(err) <= 1.0)),
        'directional_acc': np.nan,
        'spearman_r': np.nan,
        'n': int(len(actual))
    }
    # Directional versus market, if available
    if 'market_total' in df.columns:
        mkt = pd.to_numeric(df.loc[mask, 'market_total'], errors='coerce')
        m_mask = mkt.notna()
        if m_mask.any():
            pred_dir = np.sign(pred[m_mask] - mkt[m_mask])
            act_dir = np.sign(actual[m_mask] - mkt[m_mask])
            valid = (pred_dir != 0) & (act_dir != 0)
            if valid.any():
                out['directional_acc'] = float(np.mean(pred_dir[valid] == act_dir[valid]))
    try:
        r, _ = spearmanr(pred, actual)
        out['spearman_r'] = float(r)
    except Exception:
        pass
    return out

def upsert_batch(engine, date, start, end, rows):
    # Determine best MAE
    best_mae = min([r['metrics']['mae'] for r in rows if r['metrics'] and not np.isnan(r['metrics']['mae'])], default=None)
    with engine.begin() as conn:
        for r in rows:
            m = r['metrics']
            if not m: continue
            params = {
                'date': date,
                'mstart': start,
                'mend': end,
                'pred_column': r['col'],
                'best': (m['mae'] == best_mae),
                **m
            }
            conn.execute(text("""
                INSERT INTO prediction_metrics_daily (
                    date, metric_date_start, metric_date_end, pred_column,
                    mae, rmse, bias, error_std, within_0_5, within_1_0,
                    directional_acc, spearman_r, n, best_mae_flag
                ) VALUES (
                    :date, :mstart, :mend, :pred_column,
                    :mae, :rmse, :bias, :error_std, :within_0_5, :within_1_0,
                    :directional_acc, :spearman_r, :n, :best
                ) ON CONFLICT (date, pred_column, metric_date_start, metric_date_end)
                DO UPDATE SET
                  mae = EXCLUDED.mae,
                  rmse = EXCLUDED.rmse,
                  bias = EXCLUDED.bias,
                  error_std = EXCLUDED.error_std,
                  within_0_5 = EXCLUDED.within_0_5,
                  within_1_0 = EXCLUDED.within_1_0,
                  directional_acc = EXCLUDED.directional_acc,
                  spearman_r = EXCLUDED.spearman_r,
                  n = EXCLUDED.n,
                  best_mae_flag = EXCLUDED.best_mae_flag,
                  created_at = now()
            """), params)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', default=datetime.utcnow().strftime('%Y-%m-%d'), help='Evaluation end date (completed games)')
    ap.add_argument('--lookback', type=int, default=30, help='Days lookback window length')
    ap.add_argument('--cols', default='predicted_total,predicted_total_original,market_total', help='Comma list of prediction columns')
    ap.add_argument('--db', default=DB_URL)
    args = ap.parse_args()

    end = args.date
    start = (datetime.strptime(end, '%Y-%m-%d') - timedelta(days=args.lookback-1)).strftime('%Y-%m-%d')
    cols = [c.strip() for c in args.cols.split(',') if c.strip()]

    engine = create_engine(args.db)
    ensure_table(engine)
    df = fetch_games(engine, start, end, cols)
    print(f"Loaded {len(df)} games in window {start}..{end}")

    rows = []
    for col in cols:
        metrics = compute(df, col)
        rows.append({'col': col, 'metrics': metrics})
        if metrics:
            print(f"{col}: MAE={metrics['mae']:.3f} Bias={metrics['bias']:.3f} N={metrics['n']}")
        else:
            print(f"{col}: no data")

    # Persist metrics (date=end, window start..end)
    upsert_batch(engine, end, start, end, rows)

if __name__ == '__main__':
    main()
