#!/usr/bin/env python3
"""
Enhanced Final Score Collector with Pitcher Stats
=================================================

Extends the existing collect_final_scores.py to also capture:
- Game final scores (existing functionality)
- Starting pitcher game performance stats (NEW)
  - Innings pitched (home_sp_ip, away_sp_ip)
  - Earned runs (home_sp_er, away_sp_er)
  - Strikeouts (home_sp_k, away_sp_k)
  - Walks (home_sp_bb, away_sp_bb)
  - Hits allowed (home_sp_h, away_sp_h)

This should be run as part of the "scores" stage for completed games.
"""

import requests
import pandas as pd
import argparse
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import time
import os

def get_engine():
    """Get database engine"""
    DATABASE_URL = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
    return create_engine(DATABASE_URL)

def get_game_pitcher_stats(game_id):
    """Get detailed pitcher stats for a specific game - both starters AND bullpen"""
    try:
        # Get game boxscore data from MLB API
        url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch boxscore for game {game_id}: {response.status_code}")
            return None
        
        data = response.json()
        
        # Extract ALL pitcher stats (starters + relievers)
        pitcher_stats = {}
        
        # Home team pitchers
        home_pitchers = data.get('teams', {}).get('home', {}).get('pitchers', [])
        home_players = data.get('teams', {}).get('home', {}).get('players', {})
        
        if home_pitchers:
            # Starting pitcher (first in list)
            home_starter_id = f"ID{home_pitchers[0]}"
            if home_starter_id in home_players:
                player = home_players[home_starter_id]
                if 'stats' in player and 'pitching' in player['stats']:
                    stats = player['stats']['pitching']
                    pitcher_stats['home_sp_ip'] = stats.get('inningsPitched', 0)
                    pitcher_stats['home_sp_er'] = stats.get('earnedRuns', 0)
                    pitcher_stats['home_sp_k'] = stats.get('strikeOuts', 0)
                    pitcher_stats['home_sp_bb'] = stats.get('baseOnBalls', 0)
                    pitcher_stats['home_sp_h'] = stats.get('hits', 0)
            
            # Bullpen aggregate (all pitchers except starter)
            home_bp_ip = 0
            home_bp_er = 0
            home_bp_k = 0
            home_bp_bb = 0
            home_bp_h = 0
            
            for pitcher_id in home_pitchers[1:]:  # Skip starter
                player_key = f"ID{pitcher_id}"
                if player_key in home_players:
                    player = home_players[player_key]
                    if 'stats' in player and 'pitching' in player['stats']:
                        stats = player['stats']['pitching']
                        home_bp_ip += float(stats.get('inningsPitched', 0))
                        home_bp_er += stats.get('earnedRuns', 0)
                        home_bp_k += stats.get('strikeOuts', 0)
                        home_bp_bb += stats.get('baseOnBalls', 0)
                        home_bp_h += stats.get('hits', 0)
            
            pitcher_stats['home_bp_ip'] = home_bp_ip
            pitcher_stats['home_bp_er'] = home_bp_er
            pitcher_stats['home_bp_k'] = home_bp_k
            pitcher_stats['home_bp_bb'] = home_bp_bb
            pitcher_stats['home_bp_h'] = home_bp_h
        
        # Away team pitchers
        away_pitchers = data.get('teams', {}).get('away', {}).get('pitchers', [])
        away_players = data.get('teams', {}).get('away', {}).get('players', {})
        
        if away_pitchers:
            # Starting pitcher (first in list)
            away_starter_id = f"ID{away_pitchers[0]}"
            if away_starter_id in away_players:
                player = away_players[away_starter_id]
                if 'stats' in player and 'pitching' in player['stats']:
                    stats = player['stats']['pitching']
                    pitcher_stats['away_sp_ip'] = stats.get('inningsPitched', 0)
                    pitcher_stats['away_sp_er'] = stats.get('earnedRuns', 0)
                    pitcher_stats['away_sp_k'] = stats.get('strikeOuts', 0)
                    pitcher_stats['away_sp_bb'] = stats.get('baseOnBalls', 0)
                    pitcher_stats['away_sp_h'] = stats.get('hits', 0)
            
            # Bullpen aggregate (all pitchers except starter)
            away_bp_ip = 0
            away_bp_er = 0
            away_bp_k = 0
            away_bp_bb = 0
            away_bp_h = 0
            
            for pitcher_id in away_pitchers[1:]:  # Skip starter
                player_key = f"ID{pitcher_id}"
                if player_key in away_players:
                    player = away_players[player_key]
                    if 'stats' in player and 'pitching' in player['stats']:
                        stats = player['stats']['pitching']
                        away_bp_ip += float(stats.get('inningsPitched', 0))
                        away_bp_er += stats.get('earnedRuns', 0)
                        away_bp_k += stats.get('strikeOuts', 0)
                        away_bp_bb += stats.get('baseOnBalls', 0)
                        away_bp_h += stats.get('hits', 0)
            
            pitcher_stats['away_bp_ip'] = away_bp_ip
            pitcher_stats['away_bp_er'] = away_bp_er
            pitcher_stats['away_bp_k'] = away_bp_k
            pitcher_stats['away_bp_bb'] = away_bp_bb
            pitcher_stats['away_bp_h'] = away_bp_h
        
        return pitcher_stats
        
    except Exception as e:
        print(f"Error fetching pitcher stats for game {game_id}: {e}")
        return None

def get_enhanced_game_results(game_date):
    """Fetch game results AND pitcher stats from MLB Stats API for a given date"""
    try:
        # Format date for MLB API
        date_str = game_date.strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            print(f"Failed to fetch data for {date_str}: {response.status_code}")
            return []
        
        data = response.json()
        results = []
        
        for game_date_info in data.get('dates', []):
            for game in game_date_info.get('games', []):
                if game.get('status', {}).get('statusCode') == 'F':  # Final game
                    home_team = game.get('teams', {}).get('home', {}).get('team', {}).get('name', '')
                    away_team = game.get('teams', {}).get('away', {}).get('team', {}).get('name', '')
                    home_score = game.get('teams', {}).get('home', {}).get('score', 0)
                    away_score = game.get('teams', {}).get('away', {}).get('score', 0)
                    total_runs = home_score + away_score
                    game_id = game.get('gamePk')
                    
                    # Get basic game result
                    game_result = {
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': total_runs,
                        'game_id': game_id
                    }
                    
                    # Try to get pitcher stats
                    print(f"   Fetching pitcher stats for {away_team} @ {home_team}...")
                    pitcher_stats = get_game_pitcher_stats(game_id)
                    
                    if pitcher_stats:
                        game_result.update(pitcher_stats)
                        sp_msg = f"SP: {pitcher_stats.get('home_sp_ip', 0)} IP"
                        bp_msg = f"BP: {pitcher_stats.get('home_bp_ip', 0)} IP"
                        print(f"   ✅ Got pitcher stats: {sp_msg}, {bp_msg}")
                    else:
                        print(f"   ⚠️ No pitcher stats available")
                    
                    results.append(game_result)
                    
                    # Be nice to the API
                    time.sleep(0.5)
                    
        return results
        
    except Exception as e:
        print(f"Error fetching results for {game_date}: {e}")
        return []

def update_enhanced_games_with_pitcher_stats(engine, results):
    """Update enhanced_games table with game results AND pitcher stats"""
    if not results:
        return 0
    
    updated_count = 0
    
    for result in results:
        try:
            # Build dynamic UPDATE query based on available data
            set_clauses = ['total_runs = :total_runs', 'home_score = :home_score', 'away_score = :away_score']
            params = {
                'total_runs': result['total_runs'],
                'home_score': result['home_score'],
                'away_score': result['away_score'],
                'game_id': str(result['game_id'])
            }
            
            # Add pitcher stats if available (both starters and bullpen)
            pitcher_fields = ['home_sp_ip', 'home_sp_er', 'home_sp_k', 'home_sp_bb', 'home_sp_h',
                             'away_sp_ip', 'away_sp_er', 'away_sp_k', 'away_sp_bb', 'away_sp_h',
                             'home_bp_ip', 'home_bp_er', 'home_bp_k', 'home_bp_bb', 'home_bp_h',
                             'away_bp_ip', 'away_bp_er', 'away_bp_k', 'away_bp_bb', 'away_bp_h']
            
            for field in pitcher_fields:
                if field in result and result[field] is not None:
                    set_clauses.append(f'{field} = :{field}')
                    params[field] = result[field]
            
            # Try to match by game_id first
            if result.get('game_id'):
                query = text(f"""
                    UPDATE enhanced_games 
                    SET {', '.join(set_clauses)}
                    WHERE game_id = :game_id
                """)
                
                with engine.begin() as conn:
                    result_proxy = conn.execute(query, params)
                    if result_proxy.rowcount > 0:
                        updated_count += 1
                        pitcher_info = ""
                        if 'home_sp_ip' in result:
                            sp_info = f"SP: {result.get('home_sp_ip', 0)} IP / {result.get('away_sp_ip', 0)} IP"
                            bp_info = f"BP: {result.get('home_bp_ip', 0)} IP / {result.get('away_bp_ip', 0)} IP"
                            pitcher_info = f" ({sp_info}, {bp_info})"
                        print(f"✅ Updated game {result['game_id']}: {result['away_team']} @ {result['home_team']} = {result['total_runs']} runs{pitcher_info}")
                        continue
            
            print(f"⚠️ Could not update game {result.get('game_id', 'unknown')}: {result['away_team']} @ {result['home_team']}")
            
        except Exception as e:
            print(f"❌ Error updating game {result.get('game_id', 'unknown')}: {e}")
            continue
    
    return updated_count

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect final game scores and pitcher stats')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=1, help='Days back from today')
    
    args = parser.parse_args()
    
    engine = get_engine()
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = date.today() - timedelta(days=1)  # Yesterday by default
        start_date = end_date - timedelta(days=args.days_back)
    
    print(f"🔍 Collecting enhanced game results from {start_date} to {end_date}")
    print("   This includes final scores AND starting pitcher game stats")
    
    total_updated = 0
    current_date = start_date
    
    while current_date <= end_date:
        print(f"\n📅 Fetching results for {current_date}...")
        results = get_enhanced_game_results(current_date)
        
        if results:
            print(f"🎯 Found {len(results)} completed games")
            updated = update_enhanced_games_with_pitcher_stats(engine, results)
            total_updated += updated
            print(f"✅ Updated {updated} games for {current_date}")
        else:
            print(f"❌ No completed games found for {current_date}")
        
        current_date += timedelta(days=1)
        time.sleep(1)  # Be nice to the API
    
    print(f"\n🎉 [SUCCESS] Total games updated with enhanced data: {total_updated}")

if __name__ == '__main__':
    main()
