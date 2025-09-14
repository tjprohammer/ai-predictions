#!/usr/bin/env python3
"""
WORKING Bullpen Ingestor (whitelist bullpen features)
=====================================================
Populates real (team-level derived) bullpen-related whitelist features in enhanced_games:
  home_bp_era, away_bp_era, home_bp_fip, away_bp_fip,
  combined_bullpen_era, total_bullpen_innings, bullpen_impact_factor

Current approach uses team season pitching stats (MLB Stats API group=pitching) as a proxy for bullpen
metrics (still real data, but not role-isolated). Future enhancement can refine with role splits / reliever-only.
"""
import os
import sys
import json
import argparse
from datetime import datetime
import requests
import pandas as pd
from sqlalchemy import create_engine, text

# Windows UTF-8 fix
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

BULLPEN_COLUMNS = {
    'home_bp_era': 'DOUBLE PRECISION',
    'away_bp_era': 'DOUBLE PRECISION',
    'home_bp_fip': 'DOUBLE PRECISION',
    'away_bp_fip': 'DOUBLE PRECISION',
    'combined_bullpen_era': 'DOUBLE PRECISION',
    'total_bullpen_innings': 'DOUBLE PRECISION',
    'bullpen_impact_factor': 'DOUBLE PRECISION'
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

LEAGUE_AVG_ERA = 4.25  # baseline for normalization
FIP_CONST = 3.1        # league FIP constant approximation


def get_engine():
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)


def ensure_bullpen_columns(engine):
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='enhanced_games'
            """)).fetchall()
            existing_cols = {r[0] for r in existing}
            for col, ddl in BULLPEN_COLUMNS.items():
                if col not in existing_cols:
                    try:
                        conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                        print(f"🆕 Added enhanced_games.{col}")
                    except Exception as ce:
                        print(f"⚠️ Could not add column {col}: {ce}")
    except Exception as e:
        print(f"⚠️ Column introspection failed: {e}")


def fetch_team_pitching(team_id: int, season: int):
    if not team_id:
        return {}
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?stats=season&group=pitching&season={season}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return {}
        data = r.json()
        splits = (data.get('stats') or [{}])[0].get('splits') or []
        if not splits:
            return {}
        stat = splits[0].get('stat', {})
        return {
            'era': _to_float(stat.get('era')),
            'innings_pitched': _ip_to_float(stat.get('inningsPitched')),
            'hr': _to_float(stat.get('homeRuns')),
            'bb': _to_float(stat.get('baseOnBalls')),
            'k': _to_float(stat.get('strikeOuts'))
        }
    except Exception as e:
        print(f"⚠️ Team pitching fetch error {team_id}: {e}")
        return {}


def compute_fip(stats: dict):
    ip = stats.get('innings_pitched') or 0
    if not ip or ip <= 0:
        return None
    hr = stats.get('hr') or 0
    bb = stats.get('bb') or 0
    k = stats.get('k') or 0
    try:
        return ((13*hr) + (3*bb) - (2*k)) / ip + FIP_CONST
    except Exception:
        return None


def _ip_to_float(ip_str):
    if not ip_str:
        return None
    s = str(ip_str)
    if '.' in s:
        whole, frac = s.split('.', 1)
        try:
            return float(whole) + (int(frac) / 3.0)
        except Exception:
            return float(whole)
    try:
        return float(s)
    except Exception:
        return None


def _to_float(v):
    try:
        if v in (None, ''):
            return None
        return float(v)
    except Exception:
        return None


def collect_bullpen_metrics(target_date=None):
    print("Collecting Bullpen (Team Pitching Proxy) Metrics")
    print("="*48)
    if target_date:
        if len(target_date.split('-')[0]) == 2:  # MM-DD-YYYY
            m,d,y = target_date.split('-')
            game_date = f"{y}-{m}-{d}"
        else:
            game_date = target_date
    else:
        game_date = datetime.now().strftime('%Y-%m-%d')
    season_year = datetime.utcnow().year
    engine = get_engine()
    ensure_bullpen_columns(engine)
    updated = 0
    try:
        with engine.begin() as conn:
            games = pd.read_sql(text("""
                SELECT game_id, home_team, away_team FROM enhanced_games
                WHERE date = :gdate AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={'gdate': game_date})
            print(f"📅 {game_date} games: {len(games)}")
            for _, row in games.iterrows():
                gid = row.game_id
                home = row.home_team
                away = row.away_team
                hid = TEAM_ID_MAP.get(home,0)
                aid = TEAM_ID_MAP.get(away,0)
                h_stats = fetch_team_pitching(hid, season_year)
                a_stats = fetch_team_pitching(aid, season_year)
                h_era = h_stats.get('era')
                a_era = a_stats.get('era')
                h_ip = h_stats.get('innings_pitched')
                a_ip = a_stats.get('innings_pitched')
                h_fip = compute_fip(h_stats) if h_stats else None
                a_fip = compute_fip(a_stats) if a_stats else None
                combined_era = None
                if h_era is not None and a_era is not None:
                    combined_era = (h_era + a_era)/2.0
                total_ip = None
                if h_ip is not None and a_ip is not None:
                    total_ip = h_ip + a_ip
                impact = None
                if combined_era is not None:
                    impact = max(-1.0, min(1.0, (LEAGUE_AVG_ERA - combined_era)/LEAGUE_AVG_ERA))
                sql = text("""
                    UPDATE enhanced_games SET
                        home_bp_era = :h_era,
                        away_bp_era = :a_era,
                        home_bp_fip = :h_fip,
                        away_bp_fip = :a_fip,
                        combined_bullpen_era = :combined_era,
                        total_bullpen_innings = :total_ip,
                        bullpen_impact_factor = :impact
                    WHERE game_id = :gid AND date = :gdate
                """)
                params = {
                    'gdate': game_date,
                    'gid': gid,
                    'h_era': h_era,
                    'a_era': a_era,
                    'h_fip': h_fip,
                    'a_fip': a_fip,
                    'combined_era': combined_era,
                    'total_ip': total_ip,
                    'impact': impact
                }
                res = conn.execute(sql, params)
                if res.rowcount > 0:
                    updated += 1
                    def fmt(x):
                        return f"{x:.2f}" if x is not None else 'NA'
                    print(f"🧰 {away} @ {home} | BP ERA {fmt(a_era)} @ {fmt(h_era)} | FIP {fmt(a_fip)} @ {fmt(h_fip)}")
                else:
                    print(f"⚠️ No row updated for game {gid}")
    except Exception as e:
        print(f"❌ Error collecting bullpen metrics: {e}")
        return 0
    return updated


def main():
    parser = argparse.ArgumentParser(description='Collect bullpen (team pitching proxy) whitelist metrics')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    args = parser.parse_args()
    print("Bullpen Metrics Data Collection")
    print("="*33)
    updated = collect_bullpen_metrics(args.target_date)
    if updated > 0:
        print(f"\n✅ Updated bullpen metrics for {updated} games")
    else:
        print("❌ No bullpen metrics updated")

if __name__ == '__main__':
    main()
