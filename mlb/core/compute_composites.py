#!/usr/bin/env python3
"""
Composite Feature Computation
=============================
Computes missing whitelist composite features using existing base columns in enhanced_games:
  pitcher_strength_composite
  offensive_power_composite
  environmental_impact_composite
Also derives park_effect_recent (rolling 14d delta) and expected_weather_run_impact.

Design:
- Operates per target date (default today) updating only rows for that date.
- Uses standardized z-scores across recent 45-day window for stability.
- Safe fallbacks if insufficient history.

Usage:
  python compute_composites.py --date 2025-09-04
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
WINDOW_DAYS = 45

PARK_EFFECT_FALLBACK = 1.00
LEAGUE_RUN_ENV_BASE = 4.30  # baseline league run scoring environment

COMPOSITE_COLUMNS = {
    'pitcher_strength_composite': 'DOUBLE PRECISION',
    'offensive_power_composite': 'DOUBLE PRECISION',
    'environmental_impact_composite': 'DOUBLE PRECISION',
    'expected_weather_run_impact': 'DOUBLE PRECISION',
    'park_effect_recent': 'DOUBLE PRECISION'
}

PITCHER_INPUTS = [
    'home_sp_era','away_sp_era','home_sp_whip','away_sp_whip',
    'home_sp_k_per_9','away_sp_k_per_9','home_sp_hr_per_9','away_sp_hr_per_9'
]
OFFENSE_INPUTS = [
    'home_team_rpg_season','away_team_rpg_season','home_team_iso_season','away_team_iso_season',
    'home_team_wrc_plus_season','away_team_wrc_plus_season','home_team_power_season','away_team_power_season'
]
ENV_INPUTS = [
    'ballpark_run_factor','temperature','wind_speed','wind_direction_deg'
]

# ----------------- Helpers -----------------

def get_engine():
    return create_engine(DATABASE_URL)

def ensure_columns(engine):
    with engine.begin() as conn:
        existing = conn.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name='enhanced_games'
        """)).fetchall()
        existing_cols = {r[0] for r in existing}
        for col, ddl in COMPOSITE_COLUMNS.items():
            if col not in existing_cols:
                conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                print(f"🆕 Added enhanced_games.{col}")


def zscore(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series([0]*len(s), index=s.index)
    return (s - mu) / sd


def directional_wind_component(df: pd.DataFrame):
    # Convert wind direction degrees to radians (assuming already numeric 0-360)
    wd = pd.to_numeric(df.get('wind_direction_deg'), errors='coerce')
    ws = pd.to_numeric(df.get('wind_speed'), errors='coerce')
    if wd is None:
        return pd.Series([0]*len(df), index=df.index)
    rad = np.deg2rad(wd)
    # Tail-wind (out to center) component ~ cos(theta); treat 0 deg as blowing out
    return ws * np.cos(rad)


def compute_pitcher_strength(row):
    # Lower ERA/WHIP and HR/9 are good → invert; higher K/9 good.
    def inv(x):
        if x is None or (isinstance(x,float) and not np.isfinite(x)):
            return None
        return 1.0 / max(x, 0.01)
    parts = []
    for a,b in [('home_sp_era','away_sp_era'),('home_sp_whip','away_sp_whip'),('home_sp_hr_per_9','away_sp_hr_per_9')]:
        ha = row.get(a); ab = row.get(b)
        parts.extend([inv(ha), inv(ab)])
    parts.extend([row.get('home_sp_k_per_9'), row.get('away_sp_k_per_9')])
    vals = [v for v in parts if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return float(np.mean(vals))


def compute_offensive_power(row):
    vals = []
    # Base team season metrics
    for col in ['home_team_rpg_season','away_team_rpg_season','home_team_iso_season','away_team_iso_season',
                'home_team_wrc_plus_season','away_team_wrc_plus_season','home_team_power_season','away_team_power_season']:
        v = row.get(col)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    # Boost with lineup strength if available (average of home/away)
    ls_cols = ['home_lineup_strength','away_lineup_strength','home_lineup_avg_wrc_plus','away_lineup_avg_wrc_plus']
    ls_vals = [row.get(c) for c in ls_cols]
    ls_vals = [float(v) for v in ls_vals if v is not None and np.isfinite(v)]
    if ls_vals:
        # Convert wRC+-like scale to ~1.x multiplier around 1.0 baseline
        ls_mean = np.mean(ls_vals)
        lineup_mult = 1.0 + (ls_mean - 100.0) / 400.0  # +/- 0.25 at extremes
        vals = [v * lineup_mult for v in vals] if vals else [ls_mean]
    if not vals:
        return None
    return float(np.mean(vals))


def compute_environmental(row):
    park = row.get('ballpark_run_factor') or PARK_EFFECT_FALLBACK
    temp = row.get('temperature')
    wind_component = row.get('_wind_out_component')
    # Normalize components
    comp = []
    if park is not None:
        comp.append((park - 1.0) * 1.0)  # Park factor deviation
    if temp is not None:
        comp.append(((temp - 70)/15.0))
    if wind_component is not None:
        comp.append((wind_component/15.0))
    if not comp:
        return None
    return float(np.mean(comp))


def compute_expected_weather_run_impact(row):
    base = 0.0
    temp = row.get('temperature')
    wind = row.get('_wind_out_component')
    if temp is not None:
        base += (temp - 70) * 0.015  # ~1% run change per 1F difference
    if wind is not None:
        base += wind * 0.02  # scale wind mph tail component
    park = row.get('ballpark_run_factor')
    if park is not None:
        base += (park - 1.0) * 0.5
    return base


def main():
    parser = argparse.ArgumentParser(description='Compute composite whitelist features')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()

    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Computing composites for {target_date}")
    engine = get_engine()
    ensure_columns(engine)

    lookback_start = (datetime.strptime(target_date, '%Y-%m-%d') - timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')

    with engine.begin() as conn:
        df = pd.read_sql(text("""
            SELECT * FROM enhanced_games
            WHERE date = :tdate
        """), conn, params={'tdate': target_date})

        if df.empty:
            print("⚠️ No games for target date")
            return

        # Need history for recent park deltas
        hist = pd.read_sql(text("""
                SELECT date, ballpark_run_factor, game_id
                FROM enhanced_games
                WHERE date BETWEEN :start AND :tdate
            """), conn, params={'start': lookback_start, 'tdate': target_date})
        # Normalize hist date to date objects for comparison
        if not hist.empty:
            hist['date'] = pd.to_datetime(hist['date']).dt.date

    # Precompute wind directional component
    df['_wind_out_component'] = directional_wind_component(df)

    # park_effect_recent: difference between today park factor and mean prior 14 days
    park_recent_map = {}
    for _, row in df.iterrows():
        park_today = row.get('ballpark_run_factor')
        gid = row.get('game_id')
        if park_today is None:
            park_recent_map[gid] = None
            continue
        # Use hist excluding current date
        # Ensure target_date as date
        tdate_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        recent_hist = hist[(hist['date'] < tdate_obj)].tail(14)
        if recent_hist.empty:
            park_recent_map[gid] = None
        else:
            park_recent_map[gid] = park_today - recent_hist['ballpark_run_factor'].mean()

    pitcher_strength = []
    offensive_power = []
    environmental = []
    expected_weather = []
    park_recent = []

    for _, row in df.iterrows():
        ps = compute_pitcher_strength(row)
        op = compute_offensive_power(row)
        env = compute_environmental(row)
        ew = compute_expected_weather_run_impact(row)
        pr = park_recent_map.get(row.get('game_id'))
        pitcher_strength.append(ps)
        offensive_power.append(op)
        environmental.append(env)
        expected_weather.append(ew)
        park_recent.append(pr)

    df['pitcher_strength_composite'] = pitcher_strength
    df['offensive_power_composite'] = offensive_power
    df['environmental_impact_composite'] = environmental
    df['expected_weather_run_impact'] = expected_weather
    df['park_effect_recent'] = park_recent

    # Write back
    with engine.begin() as conn:
        for _, row in df.iterrows():
            update_sql = text("""
                UPDATE enhanced_games SET
                    pitcher_strength_composite = :psc,
                    offensive_power_composite = :opc,
                    environmental_impact_composite = :eic,
                    expected_weather_run_impact = :ewr,
                    park_effect_recent = :per
                WHERE game_id = :gid AND date = :tdate
            """)
            conn.execute(update_sql, {
                'psc': row['pitcher_strength_composite'],
                'opc': row['offensive_power_composite'],
                'eic': row['environmental_impact_composite'],
                'ewr': row['expected_weather_run_impact'],
                'per': row['park_effect_recent'],
                'gid': row['game_id'],
                'tdate': target_date
            })

    print("✅ Composite feature computation complete")

if __name__ == '__main__':
    main()
