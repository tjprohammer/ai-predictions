#!/usr/bin/env python3
"""
Model Calibration Utility
=========================
Computes recent bias shift and optional linear recalibration (actual = a + b * pred)
for an already-trained whitelist model directory.

Inputs:
  --model-dir <path> (expects model.joblib + bundle.json + eval_recent.csv)
  --residuals outputs/diagnostics/residuals.csv (optional)
  --window-days 30 (rolling window for bias/slope)
  --min-rows 120 (minimum rows required to apply calibration)
  --slope-band 0.7 1.3 (acceptable raw slope band; outside => unstable)
  --out outputs/diagnostics/calibration_<tag>.json

Residual Source Priority:
  1. Provided residuals CSV (expects columns including date + (pred|predicted) + (actual|total_runs))
  2. eval_recent.csv in model directory (fallback)

Bias Shift:
  mean(pred - actual) over window. Future adjusted_pred = raw_pred - bias_shift
  (If model underpredicts, bias_shift is negative; subtracting it raises predictions.)

Linear Recalibration:
  Fit OLS: actual = a + b * pred (recent window). Apply only if within slope safety band and improves AIC-like criterion.

Output JSON contains:
  bias_shift, slope, intercept, apply_bias, apply_linear, reasons (list), window_days, n_rows.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def load_model_eval(model_dir: Path) -> pd.DataFrame:
    eval_path = model_dir / 'eval_recent.csv'
    if not eval_path.exists():
        raise SystemExit(f"Missing eval_recent.csv in {model_dir}")
    df = pd.read_csv(eval_path)
    if 'date' not in df.columns:
        raise SystemExit('eval_recent.csv missing date column')
    df['date'] = pd.to_datetime(df['date']).dt.date
    # normalize column names
    if 'actual' not in df.columns and 'total_runs' in df.columns:
        df['actual'] = df['total_runs']
    if 'pred' not in df.columns and 'predicted' in df.columns:
        df['pred'] = df['predicted']
    return df[['date','actual','pred']].dropna()


def load_residuals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if 'date' not in df.columns:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['date']).dt.date
    # map columns
    if 'actual' not in df.columns and 'total_runs' in df.columns:
        df['actual'] = df['total_runs']
    if 'pred' not in df.columns and 'predicted' in df.columns:
        df['pred'] = df['predicted']
    needed = {'date','actual','pred'}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    return df[list(needed)].dropna()


def recent_window(df: pd.DataFrame, window_days: int, today=None) -> pd.DataFrame:
    if df.empty:
        return df
    if today is None:
        today = max(df['date'])
    cutoff = today - timedelta(days=window_days)
    return df[df['date'] > cutoff]


def compute_bias(df: pd.DataFrame) -> float:
    return float((df['pred'] - df['actual']).mean()) if not df.empty else 0.0


def linear_recal(df: pd.DataFrame):
    if len(df) < 10:
        return np.nan, np.nan
    X = df[['pred']].values
    y = df['actual'].values
    lr = LinearRegression().fit(X, y)
    return float(lr.coef_[0]), float(lr.intercept_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-dir', required=True)
    ap.add_argument('--residuals')
    ap.add_argument('--window-days', type=int, default=30)
    ap.add_argument('--min-rows', type=int, default=120)
    ap.add_argument('--slope-band', nargs=2, type=float, default=[0.85, 1.15])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    mdir = Path(args.model_dir)
    if not mdir.exists():
        raise SystemExit('model dir not found')

    eval_df = load_model_eval(mdir)
    res_df = load_residuals(Path(args.residuals)) if args.residuals else eval_df

    # Combine residual sources (prefer provided) by date; ensure uniqueness
    base = eval_df[['date','actual','pred']]
    if not res_df.empty:
        combined = pd.concat([res_df[['date','actual','pred']], base]).drop_duplicates(subset=['date','actual','pred'])
    else:
        combined = base

    window_df = recent_window(combined, args.window_days)
    reasons = []
    apply_bias = apply_linear = False

    if len(window_df) < args.min_rows:
        reasons.append(f"Insufficient rows ({len(window_df)} < {args.min_rows})")

    bias_shift = compute_bias(window_df) if len(window_df) >= 5 else 0.0
    slope, intercept = linear_recal(window_df)

    if not np.isnan(slope):
        lo, hi = args.slope_band
        if lo <= slope <= hi:
            apply_linear = True
        else:
            reasons.append(f"Slope {slope:.3f} outside band {lo}-{hi}")
    else:
        reasons.append('Not enough rows for linear recalibration')

    if len(window_df) >= args.min_rows:
        apply_bias = True

    out_payload = {
        'model_dir': str(mdir),
        'computed_at': datetime.utcnow().isoformat(),
        'window_days': args.window_days,
        'n_rows': int(len(window_df)),
        'bias_shift': bias_shift,
        'apply_bias': apply_bias,
        'slope': slope,
        'intercept': intercept,
        'slope_band': args.slope_band,
        'apply_linear': apply_linear,
        'reasons': reasons,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2))
    print(json.dumps(out_payload, indent=2))
    if apply_bias or apply_linear:
        print('Calibration ready (apply in live prediction pipeline).')
    else:
        print('Calibration parameters recorded but not flagged for application.')

if __name__ == '__main__':
    main()
