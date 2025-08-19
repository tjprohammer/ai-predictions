#!/usr/bin/env python3
"""
Collect actual game results for model training.
This script fetches final scores from completed games and updates the database.
"""

import requests
import pandas as pd
import argparse
from sqlalchemy import create_engine, text
from datetime import datetime, date, timedelta
import time

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
                    
                    results.append({
                        'date': game_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_runs': total_runs,
                        'game_id': game.get('gamePk')
                    })
                    
        return results
        
    except Exception as e:
        print(f"Error fetching results for {game_date}: {e}")
        return []

def update_game_results(engine, results):
    """Update database with game results"""
    if not results:
        return 0
    
    updated_count = 0
    
    for result in results:
        try:
            # Try to match games by date and teams
            query = text("""
                UPDATE games 
                SET total_runs = :total_runs,
                    home_score = :home_score,
                    away_score = :away_score,
                    updated_at = NOW()
                WHERE date = :game_date 
                AND (home_team ILIKE :home_team OR home_team ILIKE :home_team_short)
                AND (away_team ILIKE :away_team OR away_team ILIKE :away_team_short)
                AND total_runs IS NULL
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
            
            with engine.connect() as conn:
                result_proxy = conn.execute(query, params)
                if result_proxy.rowcount > 0:
                    updated_count += 1
                    print(f"Updated: {result['away_team']} @ {result['home_team']} = {result['total_runs']} runs")
            
        except Exception as e:
            print(f"Error updating game {result}: {e}")
    
    return updated_count

def main():
    parser = argparse.ArgumentParser(description='Collect MLB game results for training')
    parser.add_argument('--database-url', required=True, help='Database connection URL')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--days-back', type=int, default=7, help='Days back from today to collect')
    
    args = parser.parse_args()
    
    engine = create_engine(args.database_url)
    
    # Determine date range
    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d').date()
    else:
        end_date = date.today() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=args.days_back)
    
    print(f"Collecting game results from {start_date} to {end_date}")
    
    total_updated = 0
    current_date = start_date
    
    while current_date <= end_date:
        print(f"Fetching results for {current_date}...")
        results = get_game_results(current_date)
        
        if results:
            updated = update_game_results(engine, results)
            total_updated += updated
            print(f"Updated {updated} games for {current_date}")
        else:
            print(f"No results found for {current_date}")
        
        current_date += timedelta(days=1)
        time.sleep(0.5)  # Be nice to the API
    
    print(f"Total games updated: {total_updated}")

if __name__ == '__main__':
    main()
