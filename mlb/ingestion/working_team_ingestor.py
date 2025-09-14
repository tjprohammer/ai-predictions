#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WORKING Team Stats Ingestor (enhanced for whitelist offensive features)
======================================================================

Collects real team offensive metrics for season, last 30 days, and last 7 days.
Populates enhanced_games with whitelist features: season & L30/L7 runs per game, power, ISO, OPS, wRC+ approximations, combined metrics.
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import argparse
import sys
from datetime import datetime, timedelta

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

WHITELIST_TEAM_COLUMNS = {
    'home_team_rpg_season': 'DOUBLE PRECISION',
    'away_team_rpg_season': 'DOUBLE PRECISION',
    'home_team_rpg_l30': 'DOUBLE PRECISION',
    'away_team_rpg_l30': 'DOUBLE PRECISION',
    'home_team_rpg_l7': 'DOUBLE PRECISION',
    'away_team_rpg_l7': 'DOUBLE PRECISION',
    'home_recent_runs_per_game': 'DOUBLE PRECISION',
    'away_recent_runs_per_game': 'DOUBLE PRECISION',  # last 7 days
    'home_team_power_season': 'DOUBLE PRECISION',
    'away_team_power_season': 'DOUBLE PRECISION',
    'combined_power': 'DOUBLE PRECISION',
    'combined_offense_rpg': 'DOUBLE PRECISION',
    'home_team_iso_season': 'DOUBLE PRECISION',
    'away_team_iso_season': 'DOUBLE PRECISION',
    'home_team_wrc_plus_season': 'DOUBLE PRECISION',
    'away_team_wrc_plus_season': 'DOUBLE PRECISION',
    'home_team_wrc_plus_l30': 'DOUBLE PRECISION',
    'away_team_wrc_plus_l30': 'DOUBLE PRECISION',
    'combined_ops': 'DOUBLE PRECISION'
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

# ---------------- Utility / API helpers ----------------

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def ensure_team_whitelist_columns(engine):
    """Ensure all whitelist columns exist in enhanced_games table"""
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("""SELECT column_name FROM information_schema.columns WHERE table_name='enhanced_games'""")).fetchall()
            existing_cols = {r[0] for r in existing}
            for col, ddl in WHITELIST_TEAM_COLUMNS.items():
                if col not in existing_cols:
                    try:
                        conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                        print(f"🆕 Added enhanced_games.{col}")
                    except Exception as ce:
                        print(f"⚠️ Could not add column {col}: {ce}")
    except Exception as e:
        print(f"⚠️ Column enumeration failed: {e}")

def mlb_team_stats(team_id, season, stat_type='season'):
    """Fetch stats for a team. stat_type: 'season', 'last30', 'last7'. Uses date ranges for rolling windows."""
    if team_id == 0:
        return {}
    try:
        if stat_type == 'season':
            url = f'https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=hitting&season={season}'
        else:
            # derive date range
            today = datetime.utcnow().date()
            if stat_type == 'last30':
                start = today - timedelta(days=30)
            elif stat_type == 'last7':
                start = today - timedelta(days=7)
            else:
                start = today - timedelta(days=30)
            end = today
            url = ("https://statsapi.mlb.com/api/v1/teams/" f"{team_id}/stats?stats=byDateRange&group=hitting&season={season}" f"&startDate={start}&endDate={end}")
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        splits = (data.get('stats') or [{}])[0].get('splits') or []
        if not splits:
            return {}
        # for range, aggregate over splits
        if stat_type in ('last30','last7') and len(splits) > 1:
            agg = {
                'runs':0,'hits':0,'homeRuns':0,'gamesPlayed':0,'ops_sum':0.0,'slg_sum':0.0,'avg_sum':0.0
            }
            for sp in splits:
                st = sp.get('stat', {})
                agg['runs'] += int(st.get('runs',0))
                agg['hits'] += int(st.get('hits',0))
                agg['homeRuns'] += int(st.get('homeRuns',0))
                gp = int(st.get('gamesPlayed',0))
                agg['gamesPlayed'] += gp
                try:
                    agg['ops_sum'] += float(st.get('ops',0.0))*gp
                    agg['slg_sum'] += float(st.get('slg',0.0))*gp
                    agg['avg_sum'] += float(st.get('avg',0.0))*gp
                except:
                    pass
            gp = max(agg['gamesPlayed'],1)
            return {
                'runs': agg['runs'], 'hits': agg['hits'], 'homeRuns': agg['homeRuns'], 'games': gp,
                'ops': agg['ops_sum']/gp if gp else None,
                'slg': agg['slg_sum']/gp if gp else None,
                'avg': agg['avg_sum']/gp if gp else None
            }
        # single split (season)
        st = splits[0].get('stat', {})
        return {
            'runs': int(st.get('runs',0)),
            'hits': int(st.get('hits',0)),
            'homeRuns': int(st.get('homeRuns',0)),
            'games': int(st.get('gamesPlayed',1) or 1),
            'ops': float(st.get('ops',0.0) or 0.0),
            'slg': float(st.get('slg',0.0) or 0.0),
            'avg': float(st.get('avg',0.0) or 0.0)
        }
    except Exception as e:
        print(f"⚠️ MLB API error team {team_id} {stat_type}: {e}")
        return {}

def safe_pg(total, games):
    """Safe per-game calculation"""
    if total is None or games in (None,0):
        return None
    try:
        return float(total)/float(games) if games>0 else None
    except:
        return None

def compute_wrc_plus(ops, league_ops=0.705):
    """Compute wRC+ from OPS"""
    if ops is None:
        return None
    # approximate scaling: center 100 at league_ops, 1 OPS point ~150 scaling
    return max(60, min(160, (ops - league_ops) * 150 + 100))

def collect_team_performance_stats(target_date=None):
    """Collect team performance stats for target date's games"""
    print("Collecting Team Performance Statistics (Whitelist)")
    print("="*44)
    if target_date:
        if len(target_date.split('-')[0]) == 2:  # MM-DD-YYYY
            m,d,y = target_date.split('-')
            game_date = f"{y}-{m}-{d}"
        else:
            game_date = target_date
        date_filter_clause = "date = :gdate"
    else:
        game_date = datetime.now().strftime('%Y-%m-%d')
        date_filter_clause = "date = :gdate"
    engine = get_engine()
    ensure_team_whitelist_columns(engine)
    updated = 0
    season_year = datetime.utcnow().year
    try:
        with engine.begin() as conn:
            games = pd.read_sql(text(f"""
                SELECT game_id, home_team, away_team FROM enhanced_games
                WHERE {date_filter_clause} AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={'gdate': game_date})
            print(f"📅 Game date {game_date} - {len(games)} games")
            for _, row in games.iterrows():
                gid = row.game_id
                home = row.home_team
                away = row.away_team
                hid = TEAM_ID_MAP.get(home,0)
                aid = TEAM_ID_MAP.get(away,0)
                if hid==0 or aid==0:
                    print(f"⚠️ Missing team ID mapping for {home}/{away}")
                # Fetch stats
                h_season = mlb_team_stats(hid, season_year,'season')
                a_season = mlb_team_stats(aid, season_year,'season')
                h_l30 = mlb_team_stats(hid, season_year,'last30')
                a_l30 = mlb_team_stats(aid, season_year,'last30')
                h_l7 = mlb_team_stats(hid, season_year,'last7')
                a_l7 = mlb_team_stats(aid, season_year,'last7')
                # Derive metrics
                h_rpg_season = safe_pg(h_season.get('runs'), h_season.get('games'))
                a_rpg_season = safe_pg(a_season.get('runs'), a_season.get('games'))
                h_rpg_l30 = safe_pg(h_l30.get('runs'), h_l30.get('games')) or h_rpg_season
                a_rpg_l30 = safe_pg(a_l30.get('runs'), a_l30.get('games')) or a_rpg_season
                h_rpg_l7 = safe_pg(h_l7.get('runs'), h_l7.get('games'))
                a_rpg_l7 = safe_pg(a_l7.get('runs'), a_l7.get('games'))
                h_ops = h_season.get('ops')
                a_ops = a_season.get('ops')
                h_ops_l30 = h_l30.get('ops') or h_ops
                a_ops_l30 = a_l30.get('ops') or a_ops
                h_iso = None
                if h_season.get('slg') is not None and h_season.get('avg') is not None:
                    h_iso = h_season['slg'] - h_season['avg']
                a_iso = None
                if a_season.get('slg') is not None and a_season.get('avg') is not None:
                    a_iso = a_season['slg'] - a_season['avg']
                h_power = safe_pg(h_season.get('homeRuns'), h_season.get('games'))
                a_power = safe_pg(a_season.get('homeRuns'), a_season.get('games'))
                combined_power = None
                if h_power is not None and a_power is not None:
                    combined_power = (h_power + a_power)/2.0
                combined_offense_rpg = None
                if h_rpg_l30 and a_rpg_l30:
                    combined_offense_rpg = h_rpg_l30 + a_rpg_l30
                combined_ops = None
                if h_ops is not None and a_ops is not None:
                    combined_ops = (h_ops + a_ops)/2.0
                # wRC+ approximations
                league_ops_est = 0.705
                h_wrc_plus_season = compute_wrc_plus(h_ops, league_ops_est)
                a_wrc_plus_season = compute_wrc_plus(a_ops, league_ops_est)
                h_wrc_plus_l30 = compute_wrc_plus(h_ops_l30, league_ops_est)
                a_wrc_plus_l30 = compute_wrc_plus(a_ops_l30, league_ops_est)
                sql = text("""
                    UPDATE enhanced_games SET
                        home_team_rpg_season = :h_rpg_season,
                        away_team_rpg_season = :a_rpg_season,
                        home_team_rpg_l30 = :h_rpg_l30,
                        away_team_rpg_l30 = :a_rpg_l30,
                        home_team_rpg_l7 = :h_rpg_l7,
                        away_team_rpg_l7 = :a_rpg_l7,
                        home_recent_runs_per_game = :h_rpg_l7,
                        away_recent_runs_per_game = :a_rpg_l7,
                        home_team_power_season = :h_power,
                        away_team_power_season = :a_power,
                        combined_power = :combined_power,
                        combined_offense_rpg = :combined_offense_rpg,
                        home_team_iso_season = :h_iso,
                        away_team_iso_season = :a_iso,
                        home_team_wrc_plus_season = :h_wrc_plus_season,
                        away_team_wrc_plus_season = :a_wrc_plus_season,
                        home_team_wrc_plus_l30 = :h_wrc_plus_l30,
                        away_team_wrc_plus_l30 = :a_wrc_plus_l30,
                        combined_ops = :combined_ops
                    WHERE game_id = :gid AND date = :gdate
                """)
                params = {
                    'gid': gid, 'gdate': game_date,
                    'h_rpg_season': h_rpg_season, 'a_rpg_season': a_rpg_season,
                    'h_rpg_l30': h_rpg_l30, 'a_rpg_l30': a_rpg_l30,
                    'h_rpg_l7': h_rpg_l7, 'a_rpg_l7': a_rpg_l7,
                    'h_rpg_l7': h_rpg_l7, 'a_rpg_l7': a_rpg_l7,
                    'h_power': h_power, 'a_power': a_power,
                    'combined_power': combined_power, 'combined_offense_rpg': combined_offense_rpg,
                    'h_iso': h_iso, 'a_iso': a_iso,
                    'h_wrc_plus_season': h_wrc_plus_season, 'a_wrc_plus_season': a_wrc_plus_season,
                    'h_wrc_plus_l30': h_wrc_plus_l30, 'a_wrc_plus_l30': a_wrc_plus_l30,
                    'combined_ops': combined_ops
                }
                res = conn.execute(sql, params)
                if res.rowcount>0:
                    updated += 1
                    def fmt(x):
                        return f"{x:.2f}" if x is not None else 'NA'
                    print(f"🏟️ {away} @ {home} | RPG Szn {fmt(a_rpg_season)} @ {fmt(h_rpg_season)} | L30 {fmt(a_rpg_l30)} @ {fmt(h_rpg_l30)}")
                else:
                    print(f"⚠️ No row updated for game {gid}")
        return updated
    except Exception as e:
        print(f"❌ Error collecting team stats: {e}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect team whitelist offensive statistics')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()
    print("Team Performance Data Collection (Whitelist)")
    print("="*42)
    updated = collect_team_performance_stats(args.target_date)
    if updated>0:
        print(f"\n✅ Updated offensive whitelist metrics for {updated} games")
    else:
        print("❌ No team metrics updated")

if __name__ == "__main__":
    main()
