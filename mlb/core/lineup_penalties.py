#!/usr/bin/env python3
"""
Lineup Penalty Computation
==========================
Computes home_lineup_penalty and away_lineup_penalty for games on a date.
Heuristic placeholder using available offensive season metrics vs team season averages.

Future Enhancement:
- Integrate actual projected vs actual lineup with player-level wRC+ / OPS projections.

Current Heuristic:
- If team_wrc_plus_season < 100 → penalty = (100 - wRC+) * 0.005
- If ISO season below league avg 0.140 → additional penalty = (0.140 - ISO)*0.5
- Clamp between 0 and 1.

Usage:
  python lineup_penalties.py --date 2025-09-04
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')

LINEUP_COLUMNS = {
    'home_lineup_penalty': 'DOUBLE PRECISION',
    'away_lineup_penalty': 'DOUBLE PRECISION',
    # New: store deltas vs season baseline for transparency/usage elsewhere
    'home_lineup_strength_delta': 'DOUBLE PRECISION',
    'away_lineup_strength_delta': 'DOUBLE PRECISION'
}


def get_engine():
    from sqlalchemy import create_engine
    return create_engine(DATABASE_URL)


def ensure_columns(engine):
    with engine.begin() as conn:
        existing = conn.execute(text("""SELECT column_name FROM information_schema.columns WHERE table_name='enhanced_games'""")).fetchall()
        existing_cols = {r[0] for r in existing}
        for col, ddl in LINEUP_COLUMNS.items():
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                print(f"🆕 Added enhanced_games.{col}")


def _get_lineup_strength(row, side_prefix):
    # Prefer explicit lineup_strength, then avg_wrc_plus
    strength = row.get(f'{side_prefix}_lineup_strength')
    if strength is None or not np.isfinite(strength):
        strength = row.get(f'{side_prefix}_lineup_avg_wrc_plus')
    return strength


def compute_penalty_row(row, side_prefix):
    """Compute lineup penalty using lineup_strength vs team season baseline.
    Fallback to heuristic if no lineup_strength available.
    """
    baseline_wrc = row.get(f'{side_prefix}_team_wrc_plus_season')
    iso = row.get(f'{side_prefix}_team_iso_season')
    strength = _get_lineup_strength(row, side_prefix)

    penalty = 0.0
    # Primary: penalize if lineup strength below baseline and/or below league-average (100)
    if strength is not None and np.isfinite(strength):
        # deficit vs baseline
        if baseline_wrc is not None and np.isfinite(baseline_wrc):
            deficit = float(baseline_wrc) - float(strength)
            if deficit > 0:
                penalty += deficit * 0.007  # a bit stronger than generic below-avg
        # also penalize if under league avg
        if strength < 100:
            penalty += (100 - float(strength)) * 0.004
    else:
        # Fallback to team season heuristics
        if baseline_wrc is not None and np.isfinite(baseline_wrc) and baseline_wrc < 100:
            penalty += (100 - baseline_wrc) * 0.005

    # ISO-based light penalty if team lacks power
    if iso is not None and np.isfinite(iso) and iso < 0.140:
        penalty += (0.140 - float(iso)) * 0.5

    return max(0.0, min(1.0, float(penalty)))


def main():
    parser = argparse.ArgumentParser(description='Compute lineup penalties')
    parser.add_argument('--date', type=str, help='Target date YYYY-MM-DD')
    args = parser.parse_args()
    target_date = args.date or datetime.now().strftime('%Y-%m-%d')

    print(f'Computing lineup penalties for {target_date}')
    engine = get_engine()
    ensure_columns(engine)

    with engine.begin() as conn:
        df = pd.read_sql(text("SELECT * FROM enhanced_games WHERE date = :d"), conn, params={'d': target_date})
    if df.empty:
        print('⚠️ No games')
        return

    # Compute penalties and deltas vs baseline for transparency
    df['home_lineup_penalty'] = df.apply(lambda r: compute_penalty_row(r, 'home'), axis=1)
    df['away_lineup_penalty'] = df.apply(lambda r: compute_penalty_row(r, 'away'), axis=1)
    # deltas: lineup_strength - team season wRC+
    def lineup_delta(r, side):
        baseline = r.get(f'{side}_team_wrc_plus_season')
        strength = _get_lineup_strength(r, side)
        if baseline is None or strength is None:
            return None
        try:
            return float(strength) - float(baseline)
        except Exception:
            return None
    df['home_lineup_strength_delta'] = df.apply(lambda r: lineup_delta(r, 'home'), axis=1)
    df['away_lineup_strength_delta'] = df.apply(lambda r: lineup_delta(r, 'away'), axis=1)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            conn.execute(text("""
                UPDATE enhanced_games SET
                    home_lineup_penalty = :hlp,
                    away_lineup_penalty = :alp,
                    home_lineup_strength_delta = :hld,
                    away_lineup_strength_delta = :ald
                WHERE game_id = :gid AND date = :d
            """), {
                'hlp': row['home_lineup_penalty'],
                'alp': row['away_lineup_penalty'],
                'hld': row['home_lineup_strength_delta'],
                'ald': row['away_lineup_strength_delta'],
                'gid': row['game_id'],
                'd': target_date
            })

    # Safe print without potential encoding issues
    try:
        print('Lineup penalties updated ✅')
    except Exception:
        print('Lineup penalties updated')

if __name__ == '__main__':
    main()
