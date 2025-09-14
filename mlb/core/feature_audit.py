#!/usr/bin/env python3
"""
Feature Audit (Phase 2)

Generates coverage, stability, and correlation metrics for candidate model features
sourced from the `enhanced_games` table.

Output: outputs/diagnostics/feature_audit.csv

Metrics:
  - coverage_pct: non-null / total rows (window)
  - null_count
  - mean, std
  - daily_mean_roll14_std: std of 14-day rolling mean of the feature's daily average (stability proxy)
  - stability_ratio: daily_mean_roll14_std / std  (lower = more stable)
  - abs_corr_target: |corr(feature, total_runs)| (completed games only)
  - abs_corr_residual: |corr(feature, residual)| where residual = (total_runs - market_total)
  - vif_group: simple high-correlation grouping label (|corr| > 0.90) among retained features
  - drop_reason: populated with reason (low_coverage / low_variance / collinear_secondary)

CLI:
  python mlb/core/feature_audit.py --days 120 --min-coverage 0.7 --outdir outputs/diagnostics
  Optional: --start-date YYYY-MM-DD --end-date YYYY-MM-DD (overrides --days)

Notes:
  - Excludes identifier / target / obvious non-feature columns automatically.
  - Does not modify database; pure read & CSV export.
"""
from __future__ import annotations
import os
import argparse
from datetime import datetime, timedelta, date
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

EXCLUDE_COLS = {
    'game_id','date','home_team','away_team','prediction_timestamp',
    'total_runs','market_total','predicted_total','predicted_total_whitelist',
    'predicted_total_learning','predicted_total_ultra','over_odds','under_odds',
    'created_at','updated_at'
}

SMALL_STD_EPS = 1e-6


def parse_args():
    ap = argparse.ArgumentParser(description='Feature coverage & stability audit')
    ap.add_argument('--days', type=int, default=120, help='Lookback window (ignored if start/end given)')
    ap.add_argument('--start-date', type=str, default=None)
    ap.add_argument('--end-date', type=str, default=None)
    ap.add_argument('--min-coverage', type=float, default=0.7, help='Minimum coverage threshold to keep feature')
    ap.add_argument('--outdir', type=str, default='outputs/diagnostics')
    ap.add_argument('--limit', type=int, default=None, help='Optional row cap for debugging')
    return ap.parse_args()


def compute_window(args):
    if args.start_date and args.end_date:
        s = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        e = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        e = datetime.utcnow().date()
        s = e - timedelta(days=args.days)
    return s, e


def load_frame(engine, start: date, end: date, limit: int|None) -> pd.DataFrame:
    lim_clause = f'LIMIT {int(limit)}' if limit else ''
    q = text(f'''
        SELECT * FROM enhanced_games
        WHERE "date" BETWEEN :s AND :e
        ORDER BY "date" ASC
        {lim_clause}
    ''')
    df = pd.read_sql(q, engine, params={'s': start, 'e': end})
    return df


def pick_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    features = [c for c in numeric_cols if c not in EXCLUDE_COLS]
    return features


def daily_mean_rolling_std(series: pd.Series, dates: pd.Series) -> float:
    try:
        ddf = pd.DataFrame({'value': series.values, 'date': pd.to_datetime(dates).dt.date})
        grp = ddf.groupby('date')['value'].mean().sort_index()
        if len(grp) < 15:
            return float('nan')
        roll = grp.rolling(window=14, min_periods=7).mean()
        return float(roll.std(ddof=0))
    except Exception:
        return float('nan')


def correlation_safe(a: pd.Series, b: pd.Series) -> float:
    a_valid = a.astype(float)
    b_valid = b.astype(float)
    mask = a_valid.notna() & b_valid.notna()
    if mask.sum() < 10:
        return float('nan')
    try:
        return float(a_valid[mask].corr(b_valid[mask]))
    except Exception:
        return float('nan')


def build_vif_groups(df: pd.DataFrame, feature_list: list[str], coverage_mask: dict[str,bool]) -> dict[str,str]:
    # Simple correlation-based grouping (not true VIF). Features failing coverage already excluded from grouping.
    retained = [f for f in feature_list if coverage_mask.get(f, False)]
    if len(retained) < 2:
        return {f: f for f in feature_list}
    sub = df[retained].copy()
    # Drop columns with all NaN to avoid issues
    sub = sub.dropna(axis=1, how='all')
    corr = sub.corr().abs()
    groups = {}
    assigned = set()
    group_id = 0
    for col in corr.columns:
        if col in assigned:
            continue
        group_members = set([col])
        high_corr = corr.index[(corr[col] >= 0.90) & (corr.index != col)].tolist()
        group_members.update(high_corr)
        gid = f'g{group_id}'
        for m in group_members:
            groups[m] = gid
        assigned.update(group_members)
        group_id += 1
    # Unassigned (if any)
    for f in feature_list:
        if f not in groups:
            groups[f] = f"g{group_id}"; group_id += 1
    return groups


def main():
    args = parse_args()
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    start, end = compute_window(args)
    df = load_frame(engine, start, end, args.limit)
    if df.empty:
        print('No data in window.')
        return
    if 'date' not in df.columns:
        print('Missing date column; aborting.')
        return
    df['date'] = pd.to_datetime(df['date']).dt.date

    features = pick_feature_columns(df)
    if not features:
        print('No candidate numeric feature columns found.')
        return

    # Prepare target & residual
    total_runs = pd.to_numeric(df.get('total_runs'), errors='coerce')
    market_total = pd.to_numeric(df.get('market_total'), errors='coerce')
    residual_target = total_runs - market_total if (total_runs.notna().any() and market_total.notna().any()) else None

    rows = []
    coverage_mask = {}
    for f in features:
        col = pd.to_numeric(df[f], errors='coerce')
        total = len(col)
        non_null = int(col.notna().sum())
        coverage = non_null / total if total else 0.0
        coverage_mask[f] = coverage >= args.min_coverage
        mean = float(col.mean()) if non_null else float('nan')
        std = float(col.std(ddof=0)) if non_null else float('nan')
        daily_roll14_std = daily_mean_rolling_std(col, df['date'])
        stability_ratio = daily_roll14_std / std if std and std > 0 else float('nan')
        abs_corr_target = correlation_safe(col, total_runs) if total_runs is not None else float('nan')
        abs_corr_residual = correlation_safe(col, residual_target) if residual_target is not None else float('nan')
        rows.append({
            'feature': f,
            'coverage_pct': coverage,
            'null_count': total - non_null,
            'mean': mean,
            'std': std,
            'daily_mean_roll14_std': daily_roll14_std,
            'stability_ratio': stability_ratio,
            'abs_corr_target': abs_corr_target if np.isnan(abs_corr_target) else abs(abs_corr_target),
            'abs_corr_residual': abs_corr_residual if np.isnan(abs_corr_residual) else abs(abs_corr_residual),
        })

    audit = pd.DataFrame(rows)

    # VIF-like grouping on retained features only
    groups = build_vif_groups(df, [r['feature'] for r in rows], coverage_mask)
    audit['vif_group'] = audit['feature'].map(groups)

    # Determine primary feature per group by highest coverage then higher abs corr_target
    audit['is_retained_candidate'] = audit.apply(lambda r: coverage_mask.get(r['feature'], False) and (r['std'] or 0) > SMALL_STD_EPS, axis=1)
    group_rank = (
        audit[audit['is_retained_candidate']]
        .assign(rank_key=lambda d: list(zip(-(d['coverage_pct']), -d['abs_corr_target'].fillna(0))))
    )
    # Simpler: choose max coverage, then max abs_corr_target
    primary_map = {}
    for g, sub in audit[audit['is_retained_candidate']].groupby('vif_group'):
        sub_sorted = sub.sort_values(['coverage_pct','abs_corr_target'], ascending=[False, False])
        primary_map[g] = sub_sorted['feature'].iloc[0]
    audit['primary_in_group'] = audit.apply(lambda r: r['feature'] == primary_map.get(r['vif_group']), axis=1)

    # drop_reason assignment
    reasons = []
    for _, r in audit.iterrows():
        if r['coverage_pct'] < args.min_coverage:
            reasons.append('low_coverage')
        elif (r['std'] is np.nan) or (pd.isna(r['std'])) or (r['std'] < SMALL_STD_EPS):
            reasons.append('low_variance')
        elif not r['primary_in_group'] and r['is_retained_candidate']:
            reasons.append('collinear_secondary')
        else:
            reasons.append('')
    audit['drop_reason'] = reasons

    # Sort for readability
    audit.sort_values(['drop_reason','coverage_pct','abs_corr_target'], ascending=[True, False, False], inplace=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / 'feature_audit.csv'
    audit.to_csv(out_path, index=False)

    print(f'Features analyzed: {len(audit)} -> {out_path}')
    kept = audit[(audit['drop_reason'] == '')]
    print(f'Kept candidates: {len(kept)} (coverage >= {args.min_coverage:.0%})')
    print('Top retained by abs_corr_target:')
    print(kept[['feature','coverage_pct','abs_corr_target','abs_corr_residual','stability_ratio']].head(15).to_string(index=False))


if __name__ == '__main__':
    main()
