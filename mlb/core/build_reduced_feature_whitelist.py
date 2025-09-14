#!/usr/bin/env python3
"""
Build Reduced Feature Whitelist (Phase 2)

Reads the feature audit CSV and produces a trimmed whitelist focused on
high coverage, primary (non-collinear) features with strongest target
and residual correlations.

Scoring formula (tunable):
  score = 0.6 * abs_corr_target + 0.4 * abs_corr_residual

Selection steps:
  1. Load feature_audit.csv
  2. Keep rows with drop_reason == '' AND primary_in_group == True
  3. Compute score (NaNs treated as 0)
  4. Rank descending by score
  5. Keep top N (default 60)
  6. Write reduced_feature_whitelist.csv (columns: feature, score, coverage_pct, abs_corr_target, abs_corr_residual, stability_ratio, vif_group)

CLI:
  python mlb/core/build_reduced_feature_whitelist.py --audit-path outputs/diagnostics/feature_audit.csv --outdir outputs/diagnostics --max-features 60

You can adjust weights via --wt-target and --wt-resid (they will be normalized to sum=1).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description='Generate reduced feature whitelist from audit results')
    ap.add_argument('--audit-path', type=str, default='outputs/diagnostics/feature_audit.csv')
    ap.add_argument('--outdir', type=str, default='outputs/diagnostics')
    ap.add_argument('--max-features', type=int, default=60, help='Maximum number of features to keep')
    ap.add_argument('--wt-target', type=float, default=0.6, help='Weight for abs_corr_target')
    ap.add_argument('--wt-resid', type=float, default=0.4, help='Weight for abs_corr_residual')
    ap.add_argument('--exclude-substrings', type=str, default='', help='Comma-separated substrings; any feature containing one will be excluded')
    ap.add_argument('--output-basename', type=str, default='reduced_feature_whitelist', help='Base filename (without extension) for outputs')
    return ap.parse_args()


def main():
    args = parse_args()
    audit_path = Path(args.audit_path)
    if not audit_path.exists():
        print(f'Audit file not found: {audit_path}')
        return
    audit = pd.read_csv(audit_path)
    required_cols = {'feature','drop_reason','primary_in_group','coverage_pct','abs_corr_target','abs_corr_residual','stability_ratio','vif_group'}
    missing = required_cols - set(audit.columns)
    if missing:
        print(f'Missing required columns in audit file: {missing}')
        return

    # Normalize weights
    wt_sum = args.wt_target + args.wt_resid
    if wt_sum <= 0:
        print('Invalid weights (sum <= 0).')
        return
    w_target = args.wt_target / wt_sum
    w_resid = args.wt_resid / wt_sum

    kept = audit[(audit['drop_reason'].fillna('') == '') & (audit['primary_in_group'] == True)].copy()
    if args.exclude_substrings:
        excludes = [s.strip().lower() for s in args.exclude_substrings.split(',') if s.strip()]
        if excludes:
            mask = []
            for f in kept['feature']:
                fl = str(f).lower()
                mask.append(not any(sub in fl for sub in excludes))
            kept = kept[pd.Series(mask, index=kept.index)]
    if kept.empty:
        print('No eligible features after filters.')
        return

    # Score
    for col in ['abs_corr_target','abs_corr_residual']:
        kept[col] = pd.to_numeric(kept[col], errors='coerce').fillna(0.0)
    kept['score'] = w_target * kept['abs_corr_target'] + w_resid * kept['abs_corr_residual']

    kept.sort_values('score', ascending=False, inplace=True)
    reduced = kept.head(args.max_features).copy()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f'{args.output_basename}.csv'
    reduced[['feature','score','coverage_pct','abs_corr_target','abs_corr_residual','stability_ratio','vif_group']].to_csv(out_path, index=False)
    # JSON manifest (list only)
    (outdir / f'{args.output_basename}.json').write_text(pd.Series(reduced['feature']).to_json(orient='values'))

    print(f'Source features (audit total): {len(audit)}')
    print(f'Eligible (post filters): {len(kept)}')
    print(f'Selected (top {args.max_features}): {len(reduced)} -> {out_path}')
    print('Top 10 features:')
    print(reduced[['feature','score','coverage_pct','abs_corr_target','abs_corr_residual']].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
