#!/usr/bin/env python3
"""
Feature Drift & Variance Monitor
================================
Checks current (today's) whitelist feature distribution against a rolling
reference (last N days completed games) to flag:
  - Near-zero variance today (possible ingestion failure)
  - Large mean shift (|z| > threshold) vs reference
Outputs JSON + logs warnings; non-fatal unless HARD_FAIL_DRIFT=1.
"""
import os, json
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
ENGINE = create_engine(DATABASE_URL)
WL_PATH = Path(__file__).parent / 'learning_features_v1.json'
WL = json.loads(WL_PATH.read_text()).get('features', []) if WL_PATH.exists() else []
LOOKBACK = int(os.getenv('DRIFT_LOOKBACK_DAYS','30'))
VAR_THRESH = float(os.getenv('MIN_DAILY_VARIANCE','0.0005'))
Z_THRESH = float(os.getenv('DRIFT_Z_THRESHOLD','3.0'))
HARD_FAIL = os.getenv('HARD_FAIL_DRIFT') in ('1','true','TRUE')
TODAY = datetime.utcnow().date().strftime('%Y-%m-%d')
START = (datetime.utcnow().date() - timedelta(days=LOOKBACK)).strftime('%Y-%m-%d')

q_ref = text("""
    SELECT date, game_id, total_runs, *
    FROM enhanced_games
    WHERE date BETWEEN :start AND :today
""")
ref = pd.read_sql(q_ref, ENGINE, params={'start': START, 'today': TODAY})
if ref.empty:
    raise SystemExit('No data for drift baseline')

# Today's slice (upcoming or all rows)
ref_today = ref[ref['date'] == TODAY]
if ref_today.empty:
    print('No today rows to assess')
    raise SystemExit()

metrics = []
for f in WL:
    if f not in ref.columns:
        continue
    series_ref = pd.to_numeric(ref[f], errors='coerce')
    series_today = pd.to_numeric(ref_today[f], errors='coerce')
    if series_today.dropna().empty:
        metrics.append({'feature': f, 'status': 'missing_today'})
        continue
    ref_mean = series_ref.mean()
    ref_std = series_ref.std(ddof=1)
    today_mean = series_today.mean()
    today_std = series_today.std(ddof=1)
    z = None
    if ref_std and ref_std > 1e-9:
        z = (today_mean - ref_mean) / ref_std
    status = 'ok'
    if today_std < VAR_THRESH:
        status = 'flat'
    if z is not None and abs(z) > Z_THRESH:
        status = 'shift' if status == 'ok' else status + '+shift'
    metrics.append({
        'feature': f,
        'ref_mean': round(ref_mean,4) if pd.notna(ref_mean) else None,
        'today_mean': round(today_mean,4) if pd.notna(today_mean) else None,
        'ref_std': round(ref_std,4) if pd.notna(ref_std) else None,
        'today_std': round(today_std,4) if pd.notna(today_std) else None,
        'z': round(z,2) if z is not None else None,
        'status': status
    })

summary = {
    'date': TODAY,
    'lookback_days': LOOKBACK,
    'var_threshold': VAR_THRESH,
    'z_threshold': Z_THRESH,
    'metrics': metrics,
    'counts': {
        'total_features': len(metrics),
        'flat': sum(m['status'].startswith('flat') for m in metrics),
        'shift': sum('shift' in m['status'] for m in metrics),
        'missing_today': sum(m['status'] == 'missing_today' for m in metrics)
    }
}

out = Path(__file__).parent / 'outputs' / f'feature_drift_{TODAY}.json'
out.parent.mkdir(exist_ok=True, parents=True)
out.write_text(json.dumps(summary, indent=2))

print(json.dumps(summary['counts'], indent=2))
if HARD_FAIL and (summary['counts']['flat'] or summary['counts']['shift']):
    raise SystemExit('Drift / flat variance detected (hard fail)')
