#!/usr/bin/env python3
"""
ERA Data Ingestor - Backfill and calculate pitcher ERA statistics

This module provides functions to:
1. Backfill missing IP/ER data for pitchers from MLB API
2. Calculate rolling ERA statistics (L3, L5, L10)
3. Support both build_features.py and infer.py workflows
"""

import psycopg2
import statsapi
from datetime import datetime, date, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
import time
from typing import Optional, Dict, List, Tuple

def extract_pitcher_data_from_game(game_id: str, pitcher_id: str) -> Optional[Dict]:
    """Extract detailed pitching data for a specific pitcher from a game using MLB API"""
    try:
        game_data = statsapi.get('game', {'gamePk': game_id})
        if not game_data:
            return None
            
        # Navigate through liveData.boxscore.teams
        live_data = game_data.get('liveData', {})
        boxscore = live_data.get('boxscore', {})
        teams = boxscore.get('teams', {})
        
        for side in ['home', 'away']:
            if side in teams and 'players' in teams[side]:
                players = teams[side]['players']
                pitcher_key = f"ID{pitcher_id}"
                
                if pitcher_key in players:
                    player_data = players[pitcher_key]
                    
                    if 'stats' in player_data and 'pitching' in player_data['stats']:
                        pitching = player_data['stats']['pitching']
                        ip = pitching.get('inningsPitched', '0')
                        
                        if ip and float(ip) > 0:
                            # Get game info for teams
                            game_info = game_data.get('gameData', {})
                            teams_info = game_info.get('teams', {})
                            game_date = game_info.get('datetime', {}).get('originalDate', '')
                            
                            home_team = teams_info.get('home', {}).get('name', '')
                            away_team = teams_info.get('away', {}).get('name', '')
                            
                            team_name = home_team if side == 'home' else away_team
                            opp_team_name = away_team if side == 'home' else home_team
                            
                            # Parse innings pitched (convert X.Y format to decimal)
                            if '.' in str(ip):
                                whole, frac = str(ip).split('.')
                                ip_float = float(whole) + float(frac) / 3.0
                            else:
                                ip_float = float(ip)
                            
                            return {
                                'game_id': str(game_id),
                                'pitcher_id': str(pitcher_id),
                                'team': team_name,
                                'opp_team': opp_team_name,
                                'is_home': (side == 'home'),
                                'date': game_date,
                                'ip': ip_float,
                                'h': int(pitching.get('hits', 0)),
                                'bb': int(pitching.get('baseOnBalls', 0)),
                                'k': int(pitching.get('strikeOuts', 0)),
                                'hr': int(pitching.get('homeRuns', 0)),
                                'r': int(pitching.get('runs', 0)),
                                'er': int(pitching.get('earnedRuns', 0)),
                                'bf': int(pitching.get('battersFaced', 0)),
                                'pitches': int(pitching.get('pitchesThrown', 0)) if pitching.get('pitchesThrown') else None,
                            }
        
        return None
    except Exception as e:
        print(f"Error extracting data for pitcher {pitcher_id} in game {game_id}: {e}")
        return None

def calculate_rolling_era(pitcher_id: str, reference_date: date, games_back: int = 3, 
                         database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb") -> Optional[Dict]:
    """Calculate rolling ERA for the last N games before a reference date"""
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Get the last N starts before the reference date
            result = conn.execute(text("""
            SELECT ip, er, date
            FROM pitchers_starts 
            WHERE pitcher_id = :pid 
            AND date < :ref_date 
            AND ip IS NOT NULL 
            AND er IS NOT NULL
            ORDER BY date DESC 
            LIMIT :n_games
            """), {"pid": pitcher_id, "ref_date": reference_date, "n_games": games_back})
            
            starts = result.fetchall()
            
            if len(starts) >= games_back:
                total_ip = sum(float(start[0]) for start in starts)
                total_er = sum(int(start[1]) for start in starts)
                
                if total_ip > 0:
                    rolling_era = (total_er * 9) / total_ip
                    return {
                        'era': round(rolling_era, 2),
                        'games': len(starts),
                        'total_ip': round(total_ip, 1),
                        'total_er': total_er,
                        'date_range': f"{starts[-1][2]} to {starts[0][2]}"
                    }
            
            return None
    except Exception as e:
        print(f"Error calculating rolling ERA for pitcher {pitcher_id}: {e}")
        return None

def calculate_vs_opponent_era(pitcher_id: str, opponent_team: str, reference_date: date,
                             database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb") -> Optional[Dict]:
    """Calculate ERA against a specific opponent team"""
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Get all starts against this opponent before the reference date
            result = conn.execute(text("""
            SELECT ip, er, date, opp_team
            FROM pitchers_starts 
            WHERE pitcher_id = :pid 
            AND LOWER(opp_team) LIKE LOWER(:opp_team)
            AND date < :ref_date 
            AND ip IS NOT NULL 
            AND er IS NOT NULL
            ORDER BY date DESC
            """), {"pid": pitcher_id, "ref_date": reference_date, "opp_team": f"%{opponent_team}%"})
            
            starts = result.fetchall()
            
            if len(starts) > 0:
                total_ip = sum(float(start[0]) for start in starts)
                total_er = sum(int(start[1]) for start in starts)
                
                if total_ip > 0:
                    vs_era = (total_er * 9) / total_ip
                    return {
                        'era_vs_opp': round(vs_era, 2),
                        'games_vs_opp': len(starts),
                        'total_ip_vs_opp': round(total_ip, 1),
                        'total_er_vs_opp': total_er,
                        'opponent': opponent_team,
                        'last_vs_opp': starts[0][2] if starts else None
                    }
            
            return None
    except Exception as e:
        print(f"Error calculating vs opponent ERA for pitcher {pitcher_id} vs {opponent_team}: {e}")
        return None

def get_pitcher_era_stats(pitcher_id: str, end_date: date, 
                         database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb",
                         opponent_team: str = None) -> Dict:
    """Get comprehensive ERA statistics for a pitcher, optionally including vs opponent ERA"""
    if not pitcher_id:
        return {}
    
    try:
        # First try to get from our database
        engine = create_engine(database_url)
        with engine.connect() as conn:
            # Season ERA from database
            season_result = conn.execute(text("""
                SELECT ip, er FROM pitchers_starts 
                WHERE pitcher_id = :pid AND ip IS NOT NULL AND er IS NOT NULL
                AND date < :end_date
            """), {"pid": str(pitcher_id), "end_date": end_date})
            
            season_data = season_result.fetchall()
            
            stats = {}
            
            # Calculate season ERA from database if available
            if season_data:
                total_ip = sum(float(row[0]) for row in season_data)
                total_er = sum(int(row[1]) for row in season_data)
                if total_ip > 0:
                    stats['era_season'] = round((total_er * 9.0) / total_ip, 2)
                    stats['data_source'] = 'database'
            
            # Calculate rolling ERAs from database
            for window in [3, 5, 10]:
                rolling_data = calculate_rolling_era(pitcher_id, end_date, window, database_url)
                if rolling_data:
                    stats[f'era_l{window}'] = rolling_data['era']
                    stats[f'games_l{window}'] = rolling_data['games']
            
            # Calculate vs opponent ERA if opponent specified
            if opponent_team:
                vs_opp_data = calculate_vs_opponent_era(pitcher_id, opponent_team, end_date, database_url)
                if vs_opp_data:
                    stats.update(vs_opp_data)
            
            # If no database data, try MLB Stats API
            if not stats:
                try:
                    print(f"  Fetching from MLB API for pitcher {pitcher_id}...")
                    
                    # Get current year for stats
                    current_year = date.today().year
                    
                    # Method 1: Use statsapi.player_stats for current season
                    try:
                        current_year = date.today().year
                        stats_result = statsapi.player_stats(pitcher_id, group='[pitching]', type='season')
                        print(f"    Season stats response: {type(stats_result)}")
                        print(f"    Stats content preview: {str(stats_result)[:200]}...")
                        
                        if stats_result:
                            # Parse the stats from the response
                            import re
                            era_match = re.search(r'ERA:\s*([\d.]+)', str(stats_result))
                            ip_match = re.search(r'IP:\s*([\d.]+)', str(stats_result))
                            er_match = re.search(r'ER:\s*(\d+)', str(stats_result))
                            
                            if era_match:
                                stats['era_season'] = float(era_match.group(1))
                                stats['data_source'] = 'mlb_api_season_parsed'
                                print(f"    Found season ERA: {stats['era_season']}")
                                
                                if ip_match:
                                    stats['ip_season'] = float(ip_match.group(1))
                                if er_match:
                                    stats['er_season'] = int(er_match.group(1))
                    
                    except Exception as season_error:
                        print(f"    Season stats failed: {season_error}")
                    
                    # Method 1b: Try direct API call to MLB Stats API
                    if 'era_season' not in stats:
                        try:
                            import requests
                            api_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&group=pitching&season={current_year}"
                            print(f"    Trying direct API call: {api_url}")
                            
                            response = requests.get(api_url, timeout=10)
                            if response.status_code == 200:
                                data = response.json()
                                print(f"    Direct API response keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
                                
                                if 'stats' in data:
                                    for stat_group in data['stats']:
                                        if 'splits' in stat_group:
                                            for split in stat_group['splits']:
                                                stat = split.get('stat', {})
                                                if 'era' in stat:
                                                    stats['era_season'] = float(stat['era'])
                                                    stats['ip_season'] = float(stat.get('inningsPitched', '0'))
                                                    stats['er_season'] = int(stat.get('earnedRuns', 0))
                                                    stats['wins'] = int(stat.get('wins', 0))
                                                    stats['losses'] = int(stat.get('losses', 0))
                                                    stats['data_source'] = 'mlb_api_direct'
                                                    print(f"    Found direct API season ERA: {stats['era_season']}")
                                                    break
                        
                        except Exception as direct_error:
                            print(f"    Direct API call failed: {direct_error}")
                    
                    # Method 2: If season stats didn't work, try game log
                    if 'era_season' not in stats:
                        try:
                            game_log = statsapi.player_stat_data(pitcher_id, group='pitching', type='gameLog')
                            print(f"    Game log response type: {type(game_log)}")
                            
                            if game_log and isinstance(game_log, dict) and 'stats' in game_log:
                                total_ip = 0.0
                                total_er = 0
                                game_count = 0
                                
                                for stat_group in game_log['stats']:
                                    if 'splits' in stat_group:
                                        for split in stat_group['splits']:
                                            stat = split.get('stat', {})
                                            if 'inningsPitched' in stat and 'earnedRuns' in stat:
                                                # Parse innings pitched properly
                                                ip_str = str(stat['inningsPitched'])
                                                if '.' in ip_str:
                                                    whole, third = ip_str.split('.')
                                                    ip_value = float(whole) + float(third) / 3.0
                                                else:
                                                    ip_value = float(ip_str) if ip_str else 0.0
                                                
                                                total_ip += ip_value
                                                total_er += int(stat['earnedRuns'])
                                                game_count += 1
                                
                                if total_ip > 0:
                                    calculated_era = (total_er * 9.0) / total_ip
                                    stats['era_season'] = round(calculated_era, 2)
                                    stats['ip_season'] = total_ip
                                    stats['er_season'] = total_er
                                    stats['games_season'] = game_count
                                    stats['data_source'] = 'mlb_api_gamelog_calculated'
                                    print(f"    Calculated ERA from {game_count} games: {calculated_era:.2f}")
                        
                        except Exception as gamelog_error:
                            print(f"    Game log failed: {gamelog_error}")
                    
                    # Method 3: Try basic player lookup for identification
                    if 'era_season' not in stats:
                        try:
                            player_lookup = statsapi.lookup_player(pitcher_id)
                            if player_lookup:
                                print(f"    Player lookup found: {player_lookup}")
                                # Try to get their name at least
                                if isinstance(player_lookup, list) and len(player_lookup) > 0:
                                    player = player_lookup[0]
                                    stats['player_name'] = player.get('fullName', 'Unknown')
                                    stats['data_source'] = 'mlb_api_lookup_only'
                        
                        except Exception as lookup_error:
                            print(f"    Player lookup failed: {lookup_error}")
                
                except Exception as e:
                    print(f"    Error fetching MLB API data for pitcher {pitcher_id}: {e}")
                    import traceback
                    print(f"    Detailed error: {traceback.format_exc()[-300:]}")
            
            return stats
            
    except Exception as e:
        print(f"Error getting ERA stats for pitcher {pitcher_id}: {e}")
        return {}

def backfill_pitcher_era_data(pitcher_ids: List[str], 
                             database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb",
                             max_games_per_pitcher: int = 20) -> Dict[str, int]:
    """
    Backfill missing IP/ER data for specific pitchers
    
    Args:
        pitcher_ids: List of pitcher IDs to backfill
        database_url: Database connection string
        max_games_per_pitcher: Maximum number of games to process per pitcher
        
    Returns:
        Dictionary with pitcher_id -> number of games backfilled
    """
    results = {}
    
    try:
        conn = psycopg2.connect(database_url)
        cursor = conn.cursor()
        
        for pitcher_id in pitcher_ids:
            print(f"\nBackfilling data for pitcher {pitcher_id}...")
            
            # Get incomplete starts for this pitcher
            cursor.execute("""
            SELECT start_id, game_id, date FROM pitchers_starts 
            WHERE pitcher_id = %s 
            AND (ip IS NULL OR er IS NULL)
            ORDER BY date DESC
            LIMIT %s
            """, (pitcher_id, max_games_per_pitcher))
            
            incomplete_starts = cursor.fetchall()
            backfilled_count = 0
            
            for start_id, game_id, start_date in incomplete_starts:
                if game_id:  # Only process if we have a game_id
                    print(f"  Processing game {game_id} ({start_date})...")
                    
                    pitcher_data = extract_pitcher_data_from_game(game_id, pitcher_id)
                    
                    if pitcher_data:
                        # Update the existing record
                        cursor.execute("""
                        UPDATE pitchers_starts 
                        SET ip = %s, er = %s, h = %s, bb = %s, k = %s, hr = %s, r = %s, 
                            bf = %s, pitches = %s, team = %s, opp_team = %s, is_home = %s
                        WHERE start_id = %s
                        """, (
                            pitcher_data['ip'], pitcher_data['er'], pitcher_data['h'],
                            pitcher_data['bb'], pitcher_data['k'], pitcher_data['hr'],
                            pitcher_data['r'], pitcher_data['bf'], pitcher_data['pitches'],
                            pitcher_data['team'], pitcher_data['opp_team'], pitcher_data['is_home'],
                            start_id
                        ))
                        
                        backfilled_count += 1
                        print(f"    ✅ Updated: {pitcher_data['er']} ER in {pitcher_data['ip']} IP vs {pitcher_data['opp_team']}")
                    else:
                        print(f"    ❌ No data found for pitcher {pitcher_id} in game {game_id}")
                
                # Rate limiting
                time.sleep(0.5)
            
            results[pitcher_id] = backfilled_count
            print(f"  Completed: {backfilled_count} games backfilled for pitcher {pitcher_id}")
        
        # Commit all changes
        conn.commit()
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        print(f"Error during backfill: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return {}

def get_todays_starting_pitchers(game_date: date = None, 
                                database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb") -> List[str]:
    """Get list of starting pitcher IDs for a given date"""
    if game_date is None:
        game_date = date.today()
    
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("""
            SELECT DISTINCT pitcher_id 
            FROM (
                SELECT home_sp_id::text as pitcher_id FROM games WHERE date = :game_date
                UNION 
                SELECT away_sp_id::text as pitcher_id FROM games WHERE date = :game_date
            ) AS pitchers
            WHERE pitcher_id IS NOT NULL
            """), {"game_date": game_date})
            
            pitcher_ids = [row[0] for row in result.fetchall() if row[0]]
            return pitcher_ids
    except Exception as e:
        print(f"Error getting today's starting pitchers: {e}")
        return []

def backfill_todays_pitchers(game_date: date = None,
                           database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb") -> Dict[str, int]:
    """
    Convenience function to backfill ERA data for today's starting pitchers
    
    Args:
        game_date: Date to get pitchers for (defaults to today)
        database_url: Database connection string
        
    Returns:
        Dictionary with pitcher_id -> number of games backfilled
    """
    if game_date is None:
        game_date = date.today()
    
    print(f"🎯 Getting starting pitchers for {game_date}...")
    pitcher_ids = get_todays_starting_pitchers(game_date, database_url)
    
    if not pitcher_ids:
        print("❌ No starting pitchers found for this date")
        return {}
    
    print(f"📋 Found {len(pitcher_ids)} starting pitchers: {pitcher_ids}")
    
    print(f"🚀 Starting backfill process...")
    results = backfill_pitcher_era_data(pitcher_ids, database_url)
    
    total_backfilled = sum(results.values())
    print(f"\n🎉 Backfill complete! Total games updated: {total_backfilled}")
    
    return results

def validate_era_data(pitcher_ids: List[str] = None, 
                     database_url: str = "postgresql://mlbuser:mlbpass@localhost:5432/mlb") -> pd.DataFrame:
    """
    Validate ERA data availability for pitchers
    
    Args:
        pitcher_ids: List of specific pitcher IDs to check (optional)
        database_url: Database connection string
        
    Returns:
        DataFrame with ERA data availability summary
    """
    try:
        engine = create_engine(database_url)
        
        where_clause = ""
        params = {}
        if pitcher_ids:
            placeholders = ','.join([f':pid_{i}' for i in range(len(pitcher_ids))])
            where_clause = f"WHERE pitcher_id IN ({placeholders})"
            params = {f'pid_{i}': pid for i, pid in enumerate(pitcher_ids)}
        
        query = text(f"""
        SELECT 
            pitcher_id,
            COUNT(*) as total_starts,
            COUNT(CASE WHEN ip IS NOT NULL AND er IS NOT NULL THEN 1 END) as complete_starts,
            COUNT(CASE WHEN ip IS NULL OR er IS NULL THEN 1 END) as incomplete_starts,
            MIN(date) as first_start,
            MAX(date) as last_start
        FROM pitchers_starts 
        {where_clause}
        GROUP BY pitcher_id
        ORDER BY complete_starts DESC, total_starts DESC
        """)
        
        df = pd.read_sql(query, engine, params=params)
        return df
        
    except Exception as e:
        print(f"Error validating ERA data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage
    print("🧪 Testing ERA ingestor functionality...")
    
    # Test with today's date
    today = date.today()
    print(f"📅 Using date: {today}")
    
    # Get today's pitchers
    pitcher_ids = get_todays_starting_pitchers(today)
    print(f"📋 Found {len(pitcher_ids)} starting pitchers for {today}")
    
    # Show ERA stats for each pitcher
    print(f"\n🎯 ERA STATISTICS FOR TODAY'S PITCHERS:")
    print("=" * 70)
    
    pitchers_with_data = 0
    
    for i, pitcher_id in enumerate(pitcher_ids, 1):
        print(f"\n{i:2d}. Pitcher {pitcher_id}:")
        era_stats = get_pitcher_era_stats(pitcher_id, today)
        
        if era_stats and any(era_stats.values()):
            pitchers_with_data += 1
            print("   ✅ ERA DATA:")
            for key, value in era_stats.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.2f}")
                else:
                    print(f"      {key}: {value}")
        else:
            print("   ❌ No ERA data available")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total pitchers: {len(pitcher_ids)}")
    print(f"   With ERA data: {pitchers_with_data}")
    print(f"   Without data: {len(pitcher_ids) - pitchers_with_data}")
    
    # If no data, try backfill and show results
    if pitchers_with_data == 0:
        print(f"\n🚀 Attempting backfill for pitchers without data...")
        backfill_results = backfill_pitcher_era_data(pitcher_ids[:5])  # Test first 5
        
        if any(backfill_results.values()):
            print(f"✅ Some data backfilled! Re-testing ERA calculations...")
            for pitcher_id in list(backfill_results.keys())[:3]:  # Test first 3
                era_stats = get_pitcher_era_stats(pitcher_id, today)
                if era_stats:
                    print(f"\n🎯 Updated Pitcher {pitcher_id}:")
                    for key, value in era_stats.items():
                        if isinstance(value, float):
                            print(f"   {key}: {value:.2f}")
                        else:
                            print(f"   {key}: {value}")
    
    print("\n✅ ERA ingestor test complete!")
