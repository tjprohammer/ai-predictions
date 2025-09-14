#!/usr/bin/env python3
"""
Whitelist Feature Audit
=======================
Classifies each desired whitelist feature into:
  - missing_column: not present in table
  - all_null: present but entirely NULL over window
  - sparse: < coverage threshold non-null
  - ok: meets coverage
Also captures basic distribution stats for ok features.
"""
import os, json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text

DB = os.getenv('DATABASE_URL','postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
ENGINE = create_engine(DB)
WL_PATH = Path(__file__).parent / 'learning_features_v1.json'
WL = json.loads(WL_PATH.read_text()).get('features', []) if WL_PATH.exists() else []
LOOKBACK = int(os.getenv('WL_AUDIT_LOOKBACK_DAYS','30'))
COVERAGE_THRESH = float(os.getenv('WL_AUDIT_COVERAGE_THRESH','0.85'))
TODAY = datetime.utcnow().date()
START = TODAY - timedelta(days=LOOKBACK)

q = text("""
    SELECT * FROM enhanced_games
    WHERE date BETWEEN :start AND :today
""")

print(f"Running whitelist feature audit window={START}..{TODAY} features={len(WL)}")

df = pd.read_sql(q, ENGINE, params={'start': START, 'today': TODAY})
if df.empty:
    raise SystemExit('No rows in window')

# Normalize date column if exists
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date']).dt.date

report_rows = []
for f in WL:
    if f not in df.columns:
        report_rows.append({'feature': f, 'status': 'missing_column'})
        continue
    series = pd.to_numeric(df[f], errors='coerce')
    total = len(series)
    non_null = series.notna().sum()
    if non_null == 0:
        report_rows.append({'feature': f, 'status': 'all_null'})
        continue
    coverage = non_null / total
    if coverage < 0.05:
        status = 'all_null'  # effectively unusable
    elif coverage < COVERAGE_THRESH:
        status = 'sparse'
    else:
        status = 'ok'
    row = {'feature': f, 'status': status, 'coverage': round(coverage,3)}
    if status == 'ok':
            row.update({
                'mean': float(round(series.mean(),4)) if series.mean() == series.mean() else None,
                'std': float(round(series.std(ddof=1),4)) if series.std(ddof=1) == series.std(ddof=1) else None,
                'min': float(round(series.min(),4)) if series.min() == series.min() else None,
                'max': float(round(series.max(),4)) if series.max() == series.max() else None
            })
    report_rows.append(row)

report = {
    'window_start': START.isoformat(),
    'window_end': TODAY.isoformat(),
    'coverage_threshold': COVERAGE_THRESH,
    'summary_counts': {
    'missing_column': int(sum(r['status']=='missing_column' for r in report_rows)),
    'all_null': int(sum(r['status']=='all_null' for r in report_rows)),
    'sparse': int(sum(r['status']=='sparse' for r in report_rows)),
    'ok': int(sum(r['status']=='ok' for r in report_rows))
    },
    'features': report_rows
}

out = Path(__file__).parent / 'outputs' / 'whitelist_feature_audit.json'
out.parent.mkdir(exist_ok=True, parents=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report['summary_counts'], indent=2))
