#!/usr/bin/env python3
"""
WORKING Lineup (Player-Level) Ingestor
======================================
Collects approximate projected lineup strength metrics for each team appearing
on a target date by querying active roster player season stats, selecting the
Top N hitters by Plate Appearances (PA) up to today as a proxy lineup.

Outputs (written into enhanced_games):
  home_lineup_avg_ops, away_lineup_avg_ops
  home_lineup_avg_wrc_plus, away_lineup_avg_wrc_plus
  home_lineup_strength, away_lineup_strength (same as wRC+ avg for now)

Design Notes:
- True projected lineups require a separate source; this is a first real-data
  approximation replacing heuristic penalties.
- wRC+ approximation: ((OPS - league_ops) * 150) + 100, clamped [60, 170].
- Limits player API calls via simple in-process cache.

Usage:
  python working_lineup_ingestor.py --date 2025-09-04
"""
import os
import sys
import time
import math
import argparse
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
ENGINE = create_engine(DATABASE_URL)
LEAGUE_OPS = 0.705
TOP_N = 9
REQUEST_SLEEP = 0.12  # simple rate spacing

LINEUP_COLUMNS = {
    'home_lineup_avg_ops': 'DOUBLE PRECISION',
    'away_lineup_avg_ops': 'DOUBLE PRECISION',
    'home_lineup_avg_wrc_plus': 'DOUBLE PRECISION',
    'away_lineup_avg_wrc_plus': 'DOUBLE PRECISION',
    'home_lineup_strength': 'DOUBLE PRECISION',
    'away_lineup_strength': 'DOUBLE PRECISION'
}

TEAM_ID_MAP = {
    'Arizona Diamondbacks': 109, 'Atlanta Braves': 144, 'Baltimore Orioles': 110, 'Boston Red Sox': 111,
    'Chicago White Sox': 145, 'Chicago Cubs': 112, 'Cincinnati Reds': 113, 'Cleveland Guardians': 114,
    'Colorado Rockies': 115, 'Detroit Tigers': 116, 'Houston Astros': 117, 'Kansas City Royals': 118,
    'Los Angeles Angels': 108, 'Los Angeles Dodgers': 119, 'Miami Marlins': 146, 'Milwaukee Brewers': 158,
    'Minnesota Twins': 142, 'New York Yankees': 147, 'New York Mets': 121, 'Athletics': 133,
    'Philadelphia Phillies': 143, 'Pittsburgh Pirates': 134, 'San Diego Padres': 135, 'San Francisco Giants': 137,
    'Seattle Mariners': 136, 'St. Louis Cardinals': 138, 'Tampa Bay Rays': 139, 'Texas Rangers': 140,
    'Toronto Blue Jays': 141, 'Washington Nationals': 120
}

_player_cache = {}


def ensure_columns():
    with ENGINE.begin() as conn:
        existing_meta = conn.execute(text("""
            SELECT column_name, data_type FROM information_schema.columns
            WHERE table_name='enhanced_games'
        """))
        existing = {r[0]: r[1] for r in existing_meta}
        for col, ddl in LINEUP_COLUMNS.items():
            if col not in existing:
                conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                print(f"🆕 Added enhanced_games.{col}")
            else:
                # If existing numeric has small precision, widen
                if 'numeric' in existing[col] or 'NUMBER' in existing[col].upper():
                    try:
                        conn.execute(text(f"ALTER TABLE enhanced_games ALTER COLUMN {col} TYPE DOUBLE PRECISION USING {col}::double precision"))
                        print(f"🔧 Altered enhanced_games.{col} to DOUBLE PRECISION")
                    except Exception:
                        pass


def wrc_plus_from_ops(ops: float):
    if ops is None or not math.isfinite(ops):
        return None
    return max(60.0, min(170.0, (ops - LEAGUE_OPS) * 150 + 100))


def fetch_team_roster(team_id: int):
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json().get('roster', [])
        return [p.get('person', {}).get('id') for p in data if p.get('person')]
    except Exception as e:
        print(f"⚠️ roster fetch failed team {team_id}: {e}")
        return []


def fetch_player_season_stats(player_id: int, season: int):
    if player_id in _player_cache:
        return _player_cache[player_id]
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats?stats=season&group=hitting&season={season}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        stats = r.json().get('stats', [])
        splits = stats[0].get('splits', []) if stats else []
        if not splits:
            return None
        st = splits[0].get('stat', {})
        pa = int(st.get('plateAppearances') or 0)
        ops = float(st.get('ops') or 0.0)
        # store minimal subset
        info = {'pa': pa, 'ops': ops}
        _player_cache[player_id] = info
        return info
    except Exception as e:
        print(f"⚠️ player stats fail {player_id}: {e}")
        return None


def aggregate_lineup_metrics(team_name: str, season: int):
    team_id = TEAM_ID_MAP.get(team_name)
    if not team_id:
        return None
    roster = fetch_team_roster(team_id)
    if not roster:
        return None
    players = []
    for pid in roster:
        info = fetch_player_season_stats(pid, season)
        if info:
            players.append(info)
        time.sleep(REQUEST_SLEEP)
    if not players:
        return None
    # sort by PA desc choose top N
    players.sort(key=lambda x: x['pa'], reverse=True)
    core = players[:TOP_N]
    if not core:
        return None
    avg_ops = sum(p['ops'] for p in core)/len(core) if core else None
    avg_wrc = wrc_plus_from_ops(avg_ops) if avg_ops is not None else None
    return {
        'avg_ops': avg_ops,
        'avg_wrc_plus': avg_wrc,
        'strength': avg_wrc
    }


def process_date(target_date: str):
    season_year = datetime.utcnow().year
    with ENGINE.begin() as conn:
        games = pd.read_sql(text("SELECT game_id, home_team, away_team FROM enhanced_games WHERE date = :d"), conn, params={'d': target_date})
    if games.empty:
        print("⚠️ No games for date")
        return

    print(f"Processing {len(games)} games for {target_date}")
    updates = 0
    for _, row in games.iterrows():
        home = row.home_team
        away = row.away_team
        gid = row.game_id
        h_metrics = aggregate_lineup_metrics(home, season_year)
        a_metrics = aggregate_lineup_metrics(away, season_year)
        # Clean numeric ranges to avoid legacy precision overflow
        def clamp(v):
            if v is None: return None
            try:
                if abs(v) > 10000:  # sanity bound
                    return None
                return round(float(v), 6)
            except Exception:
                return None
        with ENGINE.begin() as conn:
            conn.execute(text("""
                UPDATE enhanced_games SET
                    home_lineup_avg_ops = :h_ops,
                    away_lineup_avg_ops = :a_ops,
                    home_lineup_avg_wrc_plus = :h_wrc,
                    away_lineup_avg_wrc_plus = :a_wrc,
                    home_lineup_strength = :h_str,
                    away_lineup_strength = :a_str
                WHERE game_id = :gid AND date = :d
            """), {
                'h_ops': clamp(h_metrics.get('avg_ops')) if h_metrics else None,
                'a_ops': clamp(a_metrics.get('avg_ops')) if a_metrics else None,
                'h_wrc': clamp(h_metrics.get('avg_wrc_plus')) if h_metrics else None,
                'a_wrc': clamp(a_metrics.get('avg_wrc_plus')) if a_metrics else None,
                'h_str': clamp(h_metrics.get('strength')) if h_metrics else None,
                'a_str': clamp(a_metrics.get('strength')) if a_metrics else None,
                'gid': gid,
                'd': target_date
            })
            updates += 1
        print(f"🏏 {away} @ {home} | lineup OPS avg {a_metrics.get('avg_ops') if a_metrics else 'NA'} @ {h_metrics.get('avg_ops') if h_metrics else 'NA'}")
    print(f"✅ Lineup aggregates updated for {updates} games")


def main():
    parser = argparse.ArgumentParser(description='Approximate lineup strength ingestion')
    parser.add_argument('--date', required=False, help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()
    target_date = args.date or datetime.utcnow().strftime('%Y-%m-%d')
    ensure_columns()
    process_date(target_date)

if __name__ == '__main__':
    main()
