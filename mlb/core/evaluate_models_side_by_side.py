#!/usr/bin/env python3
"""
Evaluate Multiple Whitelist Models Side-by-Side
=============================================
Loads each provided model directory (expects model.joblib + bundle.json) and runs
consistent metrics on its held-out eval_recent.csv plus optional bias correction.

Bias Correction:
  If --bias-window > 0 and a residual history CSV is provided via --residuals,
  we compute the rolling mean residual (actual - pred) over the most recent N days
  BEFORE the evaluation window cutoff and subtract that from raw predictions.

Additional Metrics:
  - calibration_slope: OLS slope of actual ~ pred
  - residual_std

Usage:
  python evaluate_models_side_by_side.py --models models_ultra_feature_test/whitelist_lean35_... models_ultra_feature_test/whitelist_clean45_... \
      --residuals outputs/diagnostics/residuals.csv --bias-window 30
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def load_eval(model_dir: Path):
    bundle_path = model_dir / 'bundle.json'
    eval_path = model_dir / 'eval_recent.csv'
    if not bundle_path.exists() or not eval_path.exists():
        raise FileNotFoundError(f"Missing bundle/eval in {model_dir}")
    bundle = json.loads(bundle_path.read_text())
    eval_df = pd.read_csv(eval_path)
    eval_df['date'] = pd.to_datetime(eval_df['date']).dt.date
    return bundle, eval_df


def compute_bias_adjustment(res_df: pd.DataFrame, cutoff_date, window: int):
    if window <= 0:
        return 0.0
    if res_df.empty:
        return 0.0
    res_df['date'] = pd.to_datetime(res_df['date']).dt.date
    # Normalize column names if using residuals.csv from diagnostics
    col_pred = 'pred'
    col_actual = 'actual'
    if 'pred' not in res_df.columns and 'predicted' in res_df.columns:
        col_pred = 'predicted'
    if 'actual' not in res_df.columns and 'total_runs' in res_df.columns:
        col_actual = 'total_runs'
    needed = {col_pred, col_actual}
    if not needed.issubset(res_df.columns):
        return 0.0
    hist = res_df[res_df['date'] < cutoff_date].copy()
    if hist.empty:
        return 0.0
    start_cut = max(hist['date'])
    # Take last N days
    min_allowed = pd.to_datetime(cutoff_date) - pd.Timedelta(days=window)
    recent = hist[pd.to_datetime(hist['date']) >= min_allowed]
    if recent.empty:
        recent = hist.tail(window)
    # We want mean(pred - actual); later we subtract this value from predictions.
    # If model underpredicts (pred < actual) then (pred-actual) is negative and subtracting a negative shifts preds upward.
    return float((recent[col_pred] - recent[col_actual]).mean())


def metrics(df: pd.DataFrame):
    df = df.copy()
    # Flexible column mapping
    if 'actual' not in df.columns and 'total_runs' in df.columns:
        df['actual'] = df['total_runs']
    if 'pred' not in df.columns and 'predicted' in df.columns:
        df['pred'] = df['predicted']
    if 'actual' not in df.columns or 'pred' not in df.columns:
        return {k: float('nan') for k in ['rows','MAE','RMSE','Bias','calibration_slope','residual_std']}
    df['resid'] = df['actual'] - df['pred']
    mae = float(df['resid'].abs().mean())
    rmse = float(np.sqrt((df['resid'] ** 2).mean()))
    bias = float(-df['resid'].mean())  # prediction - actual
    # calibration slope
    X = df[['pred']].values
    y = df['actual'].values
    slope = float(LinearRegression().fit(X, y).coef_[0]) if len(df) > 3 else float('nan')
    resid_std = float(df['resid'].std(ddof=1)) if len(df) > 1 else float('nan')
    return {
        'rows': int(len(df)),
        'MAE': mae,
        'RMSE': rmse,
        'Bias': bias,
        'calibration_slope': slope,
        'residual_std': resid_std
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True, help='Model directories to evaluate')
    ap.add_argument('--residuals', help='Historical residual CSV with columns date, actual, pred (from previous predictions)')
    ap.add_argument('--bias-window', type=int, default=0, help='Days of rolling residual mean to subtract (bias correction)')
    ap.add_argument('--out', default='outputs/diagnostics/model_eval_comparison.csv')
    args = ap.parse_args()

    res_df = pd.read_csv(args.residuals) if args.residuals and Path(args.residuals).exists() else pd.DataFrame()

    rows = []
    for mdir in args.models:
        mpath = Path(mdir)
        try:
            bundle, eval_df = load_eval(mpath)
        except Exception as e:
            print(f"Skip {mdir}: {e}")
            continue
        cutoff = pd.to_datetime(bundle['cutoff_date']).date()
        bias_adj = compute_bias_adjustment(res_df, cutoff, args.bias_window)
        eval_bias_df = eval_df.copy()
        if bias_adj and not np.isnan(bias_adj):
            eval_bias_df['pred'] = eval_bias_df['pred'] - bias_adj
        raw = metrics(eval_df)
        corrected = metrics(eval_bias_df)
        rows.append({
            'model_dir': mdir,
            'feature_sha': bundle['feature_sha'],
            'model_type': bundle['model_type'],
            'whitelist_size': bundle.get('whitelist_size', len(bundle.get('features',[]))),
            'bias_window': args.bias_window,
            'applied_bias_shift': bias_adj,
            'raw_MAE': raw['MAE'],
            'raw_RMSE': raw['RMSE'],
            'raw_Bias': raw['Bias'],
            'raw_calibration_slope': raw['calibration_slope'],
            'raw_residual_std': raw['residual_std'],
            'corr_MAE': corrected['MAE'],
            'corr_RMSE': corrected['RMSE'],
            'corr_Bias': corrected['Bias'],
            'corr_calibration_slope': corrected['calibration_slope'],
            'corr_residual_std': corrected['residual_std']
        })

    if not rows:
        print('No evaluations produced')
        return
    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.sort_values('corr_MAE').to_csv(out_path, index=False)
    print(f"Wrote comparison → {out_path}")
    print(out_df.sort_values('corr_MAE').to_string(index=False))

if __name__ == '__main__':
    main()
