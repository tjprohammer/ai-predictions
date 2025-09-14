#!/usr/bin/env python3
"""
Model Diagnostics (Phase 1)

Generates rolling-origin slice metrics and residual dataset for totals predictions.

Outputs (under outputs/diagnostics/):
  - slice_metrics.csv : one row per evaluation slice
  - residuals.csv     : per-game residuals for all evaluated test slices

Slice protocol:
  Train window (days) -> following test window (days). Cursor advances by test window length.
  Only completed games (total_runs NOT NULL) in test window are scored.
  Published prediction: predicted_total -> fallback predicted_total_whitelist.

Metrics per slice:
  - n_test
  - MAE, RMSE
  - DirAcc (directional accuracy vs market line)
  - calibration_slope (OLS slope of actual ~ predicted)
  - bias (mean(pred - actual))
  - residual_std

CLI:
  python mlb/core/model_diagnostics.py --train-window 90 --test-window 7 --max-slices 20
  Optional: --start-date YYYY-MM-DD --end-date YYYY-MM-DD

Environment:
  DATABASE_URL (default: postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb)

Notes:
  - Phase 2 will extend feature coverage & leakage checks.
  - Feature count placeholder left (future: count of non-null whitelisted feature columns if available).
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import math
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text


def parse_args():
    ap = argparse.ArgumentParser(description="Rolling-origin diagnostics for totals predictions")
    ap.add_argument("--train-window", type=int, default=90, help="Nominal training window length in days (upper bound)")
    ap.add_argument("--test-window", type=int, default=7, help="Test window length in days")
    ap.add_argument("--step-days", type=int, default=None, help="Stride between slice starts (defaults to test-window). Use smaller than test-window for overlap.")
    ap.add_argument("--min-train-days", type=int, default=45, help="Allow shorter training windows down to this many days if full window unavailable early.")
    ap.add_argument("--max-slices", type=int, default=30, help="Maximum number of slices to evaluate (earliest first)")
    ap.add_argument("--start-date", type=str, default=None, help="Optional start date YYYY-MM-DD (earliest if omitted)")
    ap.add_argument("--end-date", type=str, default=None, help="Optional end date YYYY-MM-DD (latest if omitted)")
    ap.add_argument("--outdir", type=str, default="outputs/diagnostics", help="Output directory root")
    ap.add_argument("--verbose", action='store_true', help="Enable verbose slice logging")
    ap.add_argument("--min-test-size", type=int, default=30, help="Minimum test games for a slice to count in aggregates")
    return ap.parse_args()


def load_data(engine, start: date | None, end: date | None) -> pd.DataFrame:
    q = '''
        SELECT game_id, "date", market_total, predicted_total, predicted_total_whitelist,
               total_runs, prediction_timestamp
        FROM enhanced_games
        WHERE (predicted_total IS NOT NULL OR predicted_total_whitelist IS NOT NULL)
          {start_clause}
          {end_clause}
        ORDER BY "date", game_id
    '''
    clauses = {
        'start_clause': 'AND "date" >= :s' if start else '',
        'end_clause': 'AND "date" <= :e' if end else ''
    }
    sql = q.format(**clauses)
    params = {}
    if start: params['s'] = start
    if end: params['e'] = end
    df = pd.read_sql(text(sql), engine, params=params)
    # Coerce numeric
    for c in ["market_total", "predicted_total", "predicted_total_whitelist", "total_runs"]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def compute_slice_metrics(df: pd.DataFrame, train_start: date, train_end: date, test_start: date, test_end: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = (df['date'] >= test_start) & (df['date'] <= test_end)
    test = df[test_mask].copy()
    # Only completed games
    test = test[test['total_runs'].notna()].copy()
    if test.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Published prediction
    test['published_pred'] = test['predicted_total'].where(test['predicted_total'].notna(), test['predicted_total_whitelist'])
    test = test[test['published_pred'].notna()].copy()
    if test.empty:
        return pd.DataFrame(), pd.DataFrame()

    test['residual'] = test['published_pred'] - test['total_runs']
    test['edge'] = test['published_pred'] - test['market_total']
    sign_pred = np.sign(test['published_pred'] - test['market_total'])
    sign_actual = np.sign(test['total_runs'] - test['market_total'])
    dir_acc = float((sign_pred == sign_actual).mean()) if len(test) else float('nan')
    mae = float(test['residual'].abs().mean())
    rmse = float(math.sqrt((test['residual']**2).mean()))
    bias = float(test['residual'].mean())
    resid_std = float(test['residual'].std(ddof=0)) if len(test) > 1 else 0.0

    pred = test['published_pred'].values
    actual = test['total_runs'].values
    slope = float(np.cov(pred, actual, ddof=0)[0,1] / np.var(pred)) if np.var(pred) > 0 else float('nan')

    slice_row = pd.DataFrame([
        {
            'train_start': train_start,
            'train_end': train_end,
            'train_days': (train_end - train_start).days + 1,
            'test_start': test_start,
            'test_end': test_end,
            'test_days': (test_end - test_start).days + 1,
            'n_test': len(test),
            'mae': mae,
            'rmse': rmse,
            'dir_acc': dir_acc,
            'calibration_slope': slope,
            'bias': bias,
            'residual_std': resid_std,
        }
    ])

    residuals = test[['game_id', 'date', 'market_total', 'published_pred', 'total_runs', 'residual', 'edge']].copy()
    residuals.rename(columns={'published_pred': 'predicted'}, inplace=True)
    return slice_row, residuals


def main():
    args = parse_args()
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

    start = datetime.strptime(args.start_date, '%Y-%m-%d').date() if args.start_date else None
    end = datetime.strptime(args.end_date, '%Y-%m-%d').date() if args.end_date else None

    df = load_data(engine, start, end)
    if df.empty:
        print('No prediction data found in selected window.')
        return
    # Ensure date column is date type
    df['date'] = pd.to_datetime(df['date']).dt.date

    global_start = min(df['date']) if not start else start
    global_end = max(df['date']) if not end else end

    train_w = timedelta(days=args.train_window)
    test_w = timedelta(days=args.test_window)
    step_days = args.step_days if args.step_days is not None else args.test_window
    step_delta = timedelta(days=step_days)

    cursor = global_start
    slice_metrics = []
    residual_rows = []
    slices = 0
    verbose = args.verbose
    while True:
        nominal_train_start = cursor
        nominal_train_end = nominal_train_start + train_w - timedelta(days=1)
        test_start = nominal_train_end + timedelta(days=1)
        test_end = test_start + test_w - timedelta(days=1)
        if test_end > global_end:
            if verbose:
                print(f"STOP: test_end {test_end} exceeds global_end {global_end}")
            break

        # Adjust training window if not enough historical days yet
        actual_train_start = nominal_train_start
        if (nominal_train_end - actual_train_start).days + 1 < args.min_train_days:
            if verbose:
                print(f"SKIP: insufficient training span (<{args.min_train_days}d) for window starting {nominal_train_start}")
            cursor = cursor + step_delta
            continue
        # compute slice
        s_metrics, s_resid = compute_slice_metrics(df, actual_train_start, nominal_train_end, test_start, test_end)
        if s_metrics.empty:
            if verbose:
                print(f"SKIP: no completed predicted games in test window {test_start}→{test_end}")
            cursor = cursor + step_delta
            continue
        slice_id = slices
        s_resid['slice_id'] = slice_id
        slice_metrics.append(s_metrics)
        residual_rows.append(s_resid)
        if verbose:
            print(f"OK slice {slice_id}: train {actual_train_start}→{nominal_train_end} ({(nominal_train_end-actual_train_start).days+1}d) | test {test_start}→{test_end} n={int(s_metrics['n_test'].iloc[0])}")
        slices += 1
        if slices >= args.max_slices:
            if verbose:
                print("Reached max slices")
            break
        cursor = cursor + step_delta

    if not slice_metrics:
        print('No slices produced (insufficient data).')
        return

    slice_df = pd.concat(slice_metrics, ignore_index=True)
    resid_df = pd.concat(residual_rows, ignore_index=True)

    # Derive additional metrics per slice
    mdae_list = []
    cmae_list = []
    for idx, row in slice_df.iterrows():
        sid = idx
        resid_slice = resid_df[resid_df['slice_id'] == sid]['residual']
        if resid_slice.empty:
            mdae = float('nan'); cmae = float('nan')
        else:
            bias = row['bias']
            mdae = float(resid_slice.abs().median())
            cmae = float((resid_slice - bias).abs().mean())  # bias-corrected MAE
        mdae_list.append(mdae)
        cmae_list.append(cmae)
    slice_df['mdae'] = mdae_list
    slice_df['bias_corrected_mae'] = cmae_list
    # Flags
    slice_df['small_slice'] = slice_df['n_test'] < args.min_test_size
    slice_df['unstable_calibration'] = (~slice_df['calibration_slope'].between(0.6, 1.4)) | slice_df['calibration_slope'].isna()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    slice_path = outdir / 'slice_metrics.csv'
    resid_path = outdir / 'residuals.csv'
    slice_df.to_csv(slice_path, index=False)
    resid_df.to_csv(resid_path, index=False)

    print(f'Wrote {len(slice_df)} slice rows -> {slice_path}')
    print(f'Wrote {len(resid_df)} residual rows -> {resid_path}')
    if len(slice_df) > 0:
        filtered = slice_df[~slice_df['small_slice']].copy()
        filt_path = outdir / 'slice_metrics_filtered.csv'
        filtered.to_csv(filt_path, index=False)
        print(f'Filtered (n_test >= {args.min_test_size}) slices: {len(filtered)} -> {filt_path}')
        agg_cols = ['mae','bias_corrected_mae','mdae','rmse','dir_acc','calibration_slope','bias','residual_std']
        if not filtered.empty:
            agg = filtered[agg_cols].agg(['mean','std'])
            print('\nAggregate metrics (filtered):')
            print(agg.to_string())
        else:
            print('\nAggregate metrics (filtered): No slices meet min-test-size threshold')
        print('\nSlice flags summary:')
        flag_summary = slice_df[['small_slice','unstable_calibration']].mean().to_frame('fraction').T
        print(flag_summary.to_string())
        print('\nLast slice:')
        print(slice_df.tail(1).to_string(index=False))


if __name__ == '__main__':
    main()
