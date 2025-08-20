#!/usr/bin/env python3
"""
Collect actual game results for enhanced_games table.
This script fetches final scores from completed games and updates the database.
"""

import requests
import pandas as pd
import argparse
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import time
import os

def get_game_results(game_date):
    """Fetch game results from MLB Stats API for a given date"""
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
                    
                    results.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': total_runs,
                        'game_id': game_id
                    })
                    
        return results
        
    except Exception as e:
        print(f"Error fetching results for {game_date}: {e}")
        return []

def update_enhanced_games_results(engine, results):
    """Update enhanced_games table with game results"""
    if not results:
        return 0
    
    updated_count = 0
    
    for result in results:
        try:
            # First try to match by game_id if available
            if result.get('game_id'):
                query = text("""
                    UPDATE enhanced_games 
                    SET total_runs = :total_runs,
                        home_score = :home_score,
                        away_score = :away_score
                    WHERE game_id = :game_id
                """)
                
                params = {
                    'total_runs': result['total_runs'],
                    'home_score': result['home_score'],
                    'away_score': result['away_score'],
                    'game_id': str(result['game_id'])  # Convert to string
                }
                
                with engine.begin() as conn:
                    result_proxy = conn.execute(query, params)
                    if result_proxy.rowcount > 0:
                        updated_count += 1
                        print(f"Updated by game_id {result['game_id']}: {result['away_team']} @ {result['home_team']} = {result['total_runs']} runs")
                        continue
            
            # Fallback: match by date and teams
            query = text("""
                UPDATE enhanced_games 
                SET total_runs = :total_runs,
                    home_score = :home_score,
                    away_score = :away_score
                WHERE date = :game_date 
                AND (home_team ILIKE :home_team OR home_team ILIKE :home_team_short)
                AND (away_team ILIKE :away_team OR away_team ILIKE :away_team_short)
            """)
            
            # Create shortened team names for matching
            home_short = result['home_team'].split()[-1]  # e.g., "Yankees" from "New York Yankees"
            away_short = result['away_team'].split()[-1]
            
            params = {
                'total_runs': result['total_runs'],
                'home_score': result['home_score'],
                'away_score': result['away_score'],
                'game_date': result['date'],
                'home_team': f"%{result['home_team']}%",
                'home_team_short': f"%{home_short}%",
                'away_team': f"%{result['away_team']}%",
                'away_team_short': f"%{away_short}%"
            }
            
            with engine.begin() as conn:
                result_proxy = conn.execute(query, params)
                if result_proxy.rowcount > 0:
                    updated_count += 1
                    print(f"Updated by teams: {result['away_team']} @ {result['home_team']} = {result['total_runs']} runs")
                else:
                    print(f"No match found for: {result['away_team']} @ {result['home_team']}")
            
        except Exception as e:
            print(f"Error updating game {result}: {e}")
    
    return updated_count

def main():
    # Use environment variable for database URL
    database_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    parser = argparse.ArgumentParser(description='Collect MLB game results for enhanced_games table')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=1, help='Days back from today to collect')
    parser.add_argument('--today', action='store_true', help='Collect today\'s results only')
    
    args = parser.parse_args()
    
    engine = create_engine(database_url)
    
    # Determine date range
    if args.today:
        start_date = end_date = date.today()
    elif args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days_back)
    
    print(f"Collecting game results from {start_date} to {end_date}")
    
    total_updated = 0
    current_date = start_date
    
    while current_date <= end_date:
        print(f"\nFetching results for {current_date}...")
        results = get_game_results(current_date)
        
        if results:
            print(f"Found {len(results)} completed games")
            updated = update_enhanced_games_results(engine, results)
            total_updated += updated
            print(f"Updated {updated} games for {current_date}")
        else:
            print(f"No completed games found for {current_date}")
        
        current_date += timedelta(days=1)
        time.sleep(0.5)  # Be nice to the API
    
    print(f"\n[SUCCESS] Total games updated: {total_updated}")

if __name__ == '__main__':
    main()
