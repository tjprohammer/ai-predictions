#!/usr/bin/env python3
"""
Historical Backfill Driver
==========================
Backfills newly added real-data whitelist feature columns in enhanced_games over a date range.
Invokes existing ingestion scripts for each date sequentially:
  working_pitcher_ingestor.py
  working_team_ingestor.py
  working_bullpen_ingestor.py
Then computes composites (after base metrics) using compute_composites.py.

Usage:
  python historical_backfill.py --start 2025-03-20 --end 2025-09-04

Notes:
- Skips dates that appear to have all target pitcher metrics already populated (light heuristic) unless --force provided.
- Sleep small interval between calls to avoid API rate limits.
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime, timedelta

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
INGEST_DIR = os.path.join(ROOT, 'mlb', 'ingestion')
PYTHON = sys.executable or 'python'

PITCHER_COLUMNS_CHECK = [
    'home_sp_era','away_sp_era','home_sp_k_per_9','away_sp_k_per_9'
]

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, OperationalError
DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')


def run_script(script_rel_path, date_str, arg_flag='--date'):
    # script_rel_path may be relative from ROOT
    path = os.path.join(ROOT, script_rel_path)
    if not os.path.exists(path):
        print(f"⚠️ Missing script {script_rel_path}; skipping")
        return 0
    cmd = [PYTHON, path, arg_flag, date_str]
    print(f"→ Running {script_rel_path} {date_str}")
    try:
        env = os.environ.copy()
        env.setdefault('PYTHONIOENCODING','utf-8')
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300, env=env)
        if res.returncode != 0:
            print(f"❌ {script_rel_path} failed {date_str}: {res.stderr[:400]}")
        else:
            print(f"✅ {script_rel_path} done {date_str}")
            if res.stdout:
                print(res.stdout[-800:])
        return res.returncode
    except subprocess.TimeoutExpired:
        print(f"⏱️ Timeout {script_rel_path} {date_str}")
        return 1


def needs_backfill(engine, date_str):
    q = text("""
        SELECT COUNT(*) as games, SUM(CASE WHEN home_sp_era IS NOT NULL AND away_sp_era IS NOT NULL THEN 1 ELSE 0 END) as have
        FROM enhanced_games WHERE date = :d
    """)
    try:
        with engine.begin() as conn:
            r = conn.execute(q, {'d': date_str}).mappings().first()
            if not r or r['games'] == 0:
                return True  # no rows means maybe schedule not loaded; still attempt
            return (r['have'] or 0) < max(1, r['games'] // 2)
    except (ProgrammingError, OperationalError):
        # Columns not present yet → treat as needing backfill
        return True


def main():
    parser = argparse.ArgumentParser(description='Historical backfill for whitelist features')
    parser.add_argument('--start', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--force', action='store_true', help='Force re-run even if appears populated')
    parser.add_argument('--sleep', type=float, default=1.0, help='Sleep seconds between scripts')
    parser.add_argument('--stages', type=str, help='Comma list subset: pitcher,team,bullpen,lineup_players,lineup_penalty,composites (default all)')
    args = parser.parse_args()

    start = datetime.strptime(args.start, '%Y-%m-%d')
    end = datetime.strptime(args.end, '%Y-%m-%d')
    if end < start:
        print('End before start')
        return

    engine = create_engine(DATABASE_URL)

    selected = None
    if args.stages:
        selected = {s.strip() for s in args.stages.split(',') if s.strip()}

    def stage_enabled(name):
        return (selected is None) or (name in selected)

    date = start
    while date <= end:
        ds = date.strftime('%Y-%m-%d')
        if not args.force and not needs_backfill(engine, ds):
            print(f"⏭️ Skipping {ds} (appears populated)")
        else:
            # Run ingestion scripts
            if stage_enabled('pitcher'):
                # pitcher ingestor expects --target-date
                run_script(os.path.join('mlb','ingestion','working_pitcher_ingestor.py'), ds, '--target-date')
                time.sleep(args.sleep)
            if stage_enabled('team'):
                run_script(os.path.join('mlb','ingestion','working_team_ingestor.py'), ds, '--target-date')
                time.sleep(args.sleep)
            if stage_enabled('bullpen'):
                # bullpen ingestor expects --target-date
                run_script(os.path.join('mlb','ingestion','working_bullpen_ingestor.py'), ds, '--target-date')
                time.sleep(args.sleep)
            if stage_enabled('lineup_players'):
                run_script(os.path.join('mlb','ingestion','working_lineup_ingestor.py'), ds, '--date')
                time.sleep(args.sleep)
            if stage_enabled('lineup_penalty'):
                run_script(os.path.join('mlb','core','lineup_penalties.py'), ds, '--date')
                time.sleep(args.sleep)
            if stage_enabled('composites'):
                run_script(os.path.join('mlb','core','compute_composites.py'), ds, '--date')
        date += timedelta(days=1)

    print('✅ Historical backfill complete')

if __name__ == '__main__':
    main()
