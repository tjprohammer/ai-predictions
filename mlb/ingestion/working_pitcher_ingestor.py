#!/usr/bin/env python3
"""
WORKING Pitcher Ingestor (enhanced with whitelist pitcher metrics)
=================================================================

Collects starting pitcher information and real season stats (ERA, WHIP, IP, K, BB, HR) for today's games.
Adds derived per-9 metrics (K/9, BB/9, HR/9) and updates enhanced_games, ensuring whitelist columns exist.
"""

import requests
import pandas as pd
from sqlalchemy import create_engine, text
import os
import json
import argparse
import sys
from datetime import datetime

# Fix encoding issues on Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

PITCHER_WHITELIST_COLUMNS = {
    'home_sp_k_per_9': 'DOUBLE PRECISION',
    'away_sp_k_per_9': 'DOUBLE PRECISION',
    'home_sp_bb_per_9': 'DOUBLE PRECISION',
    'away_sp_bb_per_9': 'DOUBLE PRECISION',
    'home_sp_hr_per_9': 'DOUBLE PRECISION',
    'away_sp_hr_per_9': 'DOUBLE PRECISION'
}

def get_engine():
    """Get database engine"""
    url = os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    return create_engine(url)

def ensure_pitcher_whitelist_columns(engine):
    """Ensure enhanced_games table has all columns for whitelist metrics"""
    try:
        with engine.begin() as conn:
            existing = conn.execute(text("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name='enhanced_games'
            """)).fetchall()
            existing_cols = {r[0] for r in existing}
            for col, ddl in PITCHER_WHITELIST_COLUMNS.items():
                if col not in existing_cols:
                    try:
                        conn.execute(text(f"ALTER TABLE enhanced_games ADD COLUMN IF NOT EXISTS {col} {ddl}"))
                        print(f"🆕 Added enhanced_games.{col}")
                    except Exception as ce:
                        print(f"⚠️ Could not add column {col}: {ce}")
    except Exception as e:
        print(f"⚠️ Column check failed: {e}")

def get_todays_starting_pitchers(target_date=None):
    """Get starting pitcher info for target date's games"""
    print("Collecting Starting Pitcher Information")
    print("=" * 40)
    
    # Use target date if provided, otherwise use today
    if target_date:
        # Convert from MM-DD-YYYY to YYYY-MM-DD if needed
        if len(target_date.split('-')[0]) == 2:  # MM-DD-YYYY format
            month, day, year = target_date.split('-')
            game_date = f"{year}-{month}-{day}"
        else:
            game_date = target_date  # Already YYYY-MM-DD
    else:
        game_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"📅 Collecting data for: {game_date}")
    
    engine = get_engine()
    ensure_pitcher_whitelist_columns(engine)
    pitcher_updates = []
    
    try:
        # Get target date's schedule with probable pitchers (fetch once)
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "startDate": game_date,
            "endDate": game_date, 
            "sportId": 1,
            "hydrate": "probablePitcher,team"
        }
        
        print("🔍 Fetching today's MLB schedule with probable pitchers...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            print(f"❌ API error for schedule: {response.status_code}")
            return []
            
        schedule_data = response.json()
        
        if 'dates' not in schedule_data or len(schedule_data['dates']) == 0:
            print("❌ No schedule data found for today")
            return []
            
        mlb_games = schedule_data['dates'][0].get('games', [])
        print(f"📅 Found {len(mlb_games)} games in MLB schedule")
        
        # Get games from our database
        with engine.begin() as conn:
            db_games = pd.read_sql(text("""
                SELECT game_id, home_team, away_team
                FROM enhanced_games 
                WHERE date = :game_date
                AND game_id IS NOT NULL
                ORDER BY game_id
            """), conn, params={'game_date': game_date})
            
            print(f"🔍 Processing {len(db_games)} games from database...")
            
            for _, db_game in db_games.iterrows():
                game_id = str(db_game['game_id'])
                home_team = db_game['home_team']
                away_team = db_game['away_team']
                
                print(f"📊 {away_team} @ {home_team} (Game {game_id})")
                
                # Find matching game in MLB schedule
                mlb_game = None
                for game in mlb_games:
                    if str(game.get('gamePk')) == game_id:
                        mlb_game = game
                        break
                
                if not mlb_game:
                    print(f"   ⚠️ Game {game_id} not found in MLB schedule")
                    continue
                
                # Extract pitcher information
                pitcher_info = extract_pitcher_info(mlb_game)
                
                if pitcher_info:
                    # Get season stats for both pitchers
                    home_stats = get_pitcher_season_stats(pitcher_info.get('home_pitcher_id'))
                    away_stats = get_pitcher_season_stats(pitcher_info.get('away_pitcher_id'))
                    
                    def per9(val, ip):
                        """Calculate per-9 inning rate"""
                        try:
                            if val is not None and ip and ip > 0:
                                return (float(val) / float(ip)) * 9.0
                        except Exception:
                            pass
                        return None
                    
                    home_k9 = per9(home_stats['strikeouts'], home_stats['innings_pitched'])
                    away_k9 = per9(away_stats['strikeouts'], away_stats['innings_pitched'])
                    home_bb9 = per9(home_stats['walks'], home_stats['innings_pitched'])
                    away_bb9 = per9(away_stats['walks'], away_stats['innings_pitched'])
                    home_hr9 = per9(home_stats.get('home_runs'), home_stats['innings_pitched'])
                    away_hr9 = per9(away_stats.get('home_runs'), away_stats['innings_pitched'])
                    
                    pitcher_updates.append({
                        'date': game_date,  # Include target date for filtering
                        'game_id': game_id,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_sp_id': pitcher_info.get('home_pitcher_id'),
                        'away_sp_id': pitcher_info.get('away_pitcher_id'),
                        'home_pitcher_name': pitcher_info.get('home_pitcher_name'),
                        'away_pitcher_name': pitcher_info.get('away_pitcher_name'),
                        'home_era': home_stats['era'],
                        'away_era': away_stats['era'],
                        'home_whip': home_stats['whip'],
                        'away_whip': away_stats['whip'],
                        'home_innings_pitched': home_stats['innings_pitched'],
                        'away_innings_pitched': away_stats['innings_pitched'],
                        'home_strikeouts': home_stats['strikeouts'],
                        'away_strikeouts': away_stats['strikeouts'],
                        'home_walks': home_stats['walks'],
                        'away_walks': away_stats['walks'],
                        'home_sp_k_per_9': home_k9,
                        'away_sp_k_per_9': away_k9,
                        'home_sp_bb_per_9': home_bb9,
                        'away_sp_bb_per_9': away_bb9,
                        'home_sp_hr_per_9': home_hr9,
                        'away_sp_hr_per_9': away_hr9
                    })
                    
                    def fmt(x):
                        return f"{x:.2f}" if (x is not None and x == x) else "NA"
                    print(f"   Home SP: {pitcher_info.get('home_pitcher_name', 'Unknown')} ERA {fmt(home_stats['era'])} K9 {fmt(home_k9)} BB9 {fmt(home_bb9)} HR9 {fmt(home_hr9)}")
                    print(f"   Away SP: {pitcher_info.get('away_pitcher_name', 'Unknown')} ERA {fmt(away_stats['era'])} K9 {fmt(away_k9)} BB9 {fmt(away_bb9)} HR9 {fmt(away_hr9)}")
                else:
                    print(f"   ⚠️ No pitcher info found for game {game_id}")
                    
    except Exception as e:
        print(f"❌ Error getting pitcher data: {e}")
        return []
    
    return pitcher_updates

def extract_pitcher_info(game_data):
    """Extract pitcher information from MLB game data"""
    try:
        home_pitcher_id = None
        away_pitcher_id = None
        home_pitcher_name = None
        away_pitcher_name = None
        
        # Get probable pitchers
        if 'teams' in game_data:
            home_team = game_data['teams'].get('home', {})
            away_team = game_data['teams'].get('away', {})
            
            # Home pitcher
            if 'probablePitcher' in home_team and home_team['probablePitcher']:
                home_pitcher_id = home_team['probablePitcher']['id']
                home_pitcher_name = home_team['probablePitcher']['fullName']
                
            # Away pitcher  
            if 'probablePitcher' in away_team and away_team['probablePitcher']:
                away_pitcher_id = away_team['probablePitcher']['id']
                away_pitcher_name = away_team['probablePitcher']['fullName']
        
        return {
            'home_pitcher_id': home_pitcher_id,
            'away_pitcher_id': away_pitcher_id,
            'home_pitcher_name': home_pitcher_name,
            'away_pitcher_name': away_pitcher_name
        }
        
    except Exception as e:
        print(f"      ❌ Error extracting pitcher info: {e}")
        return None

def _ip_to_float(ip_str):
    """Convert MLB innings pitched string to float (e.g., '123.1' -> 123.33)"""
    if not ip_str: 
        return None
    s = str(ip_str)
    if "." in s:
        whole, frac = s.split(".", 1)
        try:
            return float(whole) + (int(frac) / 3.0)  # MLB uses .1/.2 = 1/3, 2/3
        except:
            return float(whole)
    try:
        return float(s)
    except:
        return None

def _to_float(x):
    """Safely convert to float, returning None for missing/invalid values"""
    try:
        if x in (None, "", "0", "0.0"): 
            return None
        return float(x)
    except:
        return None

# Replace original get_pitcher_season_stats with extended version

def get_pitcher_season_stats(pitcher_id, season=None):
    """Fetch season ERA and stats for a pitcher, returning None for missing data"""
    season = season or datetime.now().year
    if not pitcher_id:
        return {'era': None, 'whip': None, 'innings_pitched': None, 'strikeouts': None, 'walks': None, 'home_runs': None}
    
    try:
        url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&group=pitching&gameType=R&season={season}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            splits = (data.get('stats') or [{}])[0].get('splits') or []
            if splits:
                stat = splits[0].get('stat', {})
                return {
                    'era': _to_float(stat.get('era')),
                    'whip': _to_float(stat.get('whip')),
                    'innings_pitched': _ip_to_float(stat.get('inningsPitched')),
                    'strikeouts': _to_float(stat.get('strikeOuts')),
                    'walks': _to_float(stat.get('baseOnBalls')),
                    'home_runs': _to_float(stat.get('homeRuns'))
                }
    except Exception as e:
        print(f"      ⚠️ Error fetching stats for pitcher {pitcher_id}: {e}")
    
    return {'era': None, 'whip': None, 'innings_pitched': None, 'strikeouts': None, 'walks': None, 'home_runs': None}

def update_pitcher_ids(pitcher_updates):
    """Update enhanced_games table with starting pitcher IDs and season stats"""
    if not pitcher_updates:
        return 0
    
    engine = get_engine()
    updated_count = 0
    
    try:
        with engine.begin() as conn:
            for u in pitcher_updates:
                # Always update with latest pitcher stats (no COALESCE preservation)
                sql = text("""
                    UPDATE enhanced_games
                    SET
                        home_sp_id           = :home_sp_id,
                        away_sp_id           = :away_sp_id,
                        home_sp_name         = :home_sp_name,
                        away_sp_name         = :away_sp_name,
                        home_sp_season_era   = :home_era,
                        away_sp_season_era   = :away_era,
                        home_sp_whip         = :home_whip,
                        away_sp_whip         = :away_whip,
                        home_sp_season_k     = :home_strikeouts,
                        away_sp_season_k     = :away_strikeouts,
                        home_sp_season_bb    = :home_walks,
                        away_sp_season_bb    = :away_walks,
                        home_sp_season_ip    = :home_innings_pitched,
                        away_sp_season_ip    = :away_innings_pitched,
                        home_sp_k_per_9      = :home_sp_k_per_9,
                        away_sp_k_per_9      = :away_sp_k_per_9,
                        home_sp_bb_per_9     = :home_sp_bb_per_9,
                        away_sp_bb_per_9     = :away_sp_bb_per_9,
                        home_sp_hr_per_9     = :home_sp_hr_per_9,
                        away_sp_hr_per_9     = :away_sp_hr_per_9
                    WHERE game_id = :game_id AND date = :date
                """)
                
                params = {
                    'date': u['date'],
                    'game_id': u['game_id'],
                    'home_sp_id': u['home_sp_id'],
                    'away_sp_id': u['away_sp_id'],
                    'home_sp_name': u['home_pitcher_name'],
                    'away_sp_name': u['away_pitcher_name'],
                    'home_era': u['home_era'],
                    'away_era': u['away_era'],
                    'home_whip': u['home_whip'],
                    'away_whip': u['away_whip'],
                    'home_strikeouts': u['home_strikeouts'],
                    'away_strikeouts': u['away_strikeouts'],
                    'home_walks': u['home_walks'],
                    'away_walks': u['away_walks'],
                    'home_innings_pitched': u['home_innings_pitched'],
                    'away_innings_pitched': u['away_innings_pitched'],
                    'home_sp_k_per_9': u.get('home_sp_k_per_9'),
                    'away_sp_k_per_9': u.get('away_sp_k_per_9'),
                    'home_sp_bb_per_9': u.get('home_sp_bb_per_9'),
                    'away_sp_bb_per_9': u.get('away_sp_bb_per_9'),
                    'home_sp_hr_per_9': u.get('home_sp_hr_per_9'),
                    'away_sp_hr_per_9': u.get('away_sp_hr_per_9'),
                }
                
                result = conn.execute(sql, params)
                
                if result.rowcount > 0:
                    updated_count += 1
                    print(f"✅ Updated pitchers for {u['away_team']} @ {u['home_team']}")
                else:
                    print(f"⚠️ No game found to update for game_id {u['game_id']} on {u['date']}")
        
        return updated_count
        
    except Exception as e:
        print(f"❌ Error updating pitcher data: {e}")
        return 0

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect starting pitcher data (with whitelist metrics)')
    parser.add_argument('--target-date', type=str, help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    print("Starting Pitcher Data Collection (Whitelist Metrics)")
    print("=" * 40)
    
    pitcher_updates = get_todays_starting_pitchers(args.target_date)
    
    if pitcher_updates:
        updated = update_pitcher_ids(pitcher_updates)
        print(f"\n✅ Updated starting pitcher data for {updated} games")
        
        # Save pitcher data to file for reference
        filename = f'daily_starting_pitchers_{args.target_date}.json' if args.target_date else 'daily_starting_pitchers.json'
        with open(filename, 'w') as f:
            json.dump(pitcher_updates, f, indent=2)
        print(f"💾 Saved pitcher data to {filename}")
    else:
        print("❌ No pitcher data to update")

if __name__ == "__main__":
    main()
