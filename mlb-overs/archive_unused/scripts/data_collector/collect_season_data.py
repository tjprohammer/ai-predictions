#!/usr/bin/env python3
"""
Comprehensive season data collector for model training.
Collects games, features, and actual outcomes for the entire 2025 season.
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
import sys

# Add the parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

def fetch_season_schedule(year=2025):
    """Fetch all games from the season."""
    print(f"Fetching {year} season schedule...")
    
    # MLB season typically runs March-October
    start_date = f"{year}-03-01"
    end_date = f"{year}-10-31"
    
    url = f"https://statsapi.mlb.com/api/v1/schedule"
    params = {
        'startDate': start_date,
        'endDate': end_date,
        'sportId': 1,  # MLB
        'hydrate': 'team,game(content(summary,media(epg))),linescore,flags,liveLookin,review,broadcasts(all),decisions,person,probablePitcher,stats,homeRuns,previousPlay,game(content(summary,media(epg))),seriesStatus'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching schedule: {e}")
        return None

def extract_completed_games(schedule_data):
    """Extract games that have been completed with final scores."""
    completed_games = []
    
    if not schedule_data or 'dates' not in schedule_data:
        return completed_games
    
    for date_obj in schedule_data['dates']:
        if 'games' not in date_obj:
            continue
            
        for game in date_obj['games']:
            # Only include completed games
            if game.get('status', {}).get('statusCode') != 'F':
                continue
                
            # Extract basic game info
            game_data = {
                'game_id': game['gamePk'],
                'date': date_obj['date'],
                'home_team': game['teams']['home']['team']['name'],
                'away_team': game['teams']['away']['team']['name'],
                'home_team_id': game['teams']['home']['team']['id'],
                'away_team_id': game['teams']['away']['team']['id'],
            }
            
            # Extract scores if available
            if 'teams' in game and 'home' in game['teams'] and 'away' in game['teams']:
                home_score = game['teams']['home'].get('score')
                away_score = game['teams']['away'].get('score')
                
                if home_score is not None and away_score is not None:
                    game_data['home_score'] = home_score
                    game_data['away_score'] = away_score
                    game_data['total_runs'] = home_score + away_score
                    
                    # Extract pitcher info if available
                    if 'probablePitcher' in game['teams']['home']:
                        game_data['home_probable_pitcher'] = game['teams']['home']['probablePitcher'].get('fullName')
                        game_data['home_pitcher_id'] = game['teams']['home']['probablePitcher'].get('id')
                    
                    if 'probablePitcher' in game['teams']['away']:
                        game_data['away_probable_pitcher'] = game['teams']['away']['probablePitcher'].get('fullName')
                        game_data['away_pitcher_id'] = game['teams']['away']['probablePitcher'].get('id')
                    
                    completed_games.append(game_data)
    
    return completed_games

def collect_historical_betting_lines(games_df):
    """Try to collect historical betting lines for games (if available)."""
    print("Attempting to collect historical betting lines...")
    
    # For now, we'll estimate based on typical MLB totals
    # In a real implementation, you'd integrate with sports betting APIs
    games_df['k_close'] = 8.5  # Default MLB total
    
    # Add some variation based on team strength (rough approximation)
    high_scoring_teams = ['Rockies', 'Rangers', 'Astros', 'Blue Jays']
    low_scoring_teams = ['Rays', 'Giants', 'Marlins', 'Athletics']
    
    for idx, row in games_df.iterrows():
        if any(team in row['home_team'] for team in high_scoring_teams) or \
           any(team in row['away_team'] for team in high_scoring_teams):
            games_df.at[idx, 'k_close'] = 9.0
        elif any(team in row['home_team'] for team in low_scoring_teams) or \
             any(team in row['away_team'] for team in low_scoring_teams):
            games_df.at[idx, 'k_close'] = 8.0
    
    return games_df

def main():
    print("🏆 COMPREHENSIVE SEASON DATA COLLECTION")
    print("="*60)
    
    # Fetch all completed games from 2025 season
    schedule_data = fetch_season_schedule(2025)
    if not schedule_data:
        print("❌ Failed to fetch schedule data")
        return
    
    # Extract completed games with scores
    completed_games = extract_completed_games(schedule_data)
    print(f"✅ Found {len(completed_games)} completed games")
    
    if not completed_games:
        print("❌ No completed games found")
        return
    
    # Convert to DataFrame
    games_df = pd.DataFrame(completed_games)
    games_df['date'] = pd.to_datetime(games_df['date'])
    
    # Add betting lines (estimated for now)
    games_df = collect_historical_betting_lines(games_df)
    
    # Display summary
    print(f"\n📊 SEASON DATA SUMMARY:")
    print(f"Total completed games: {len(games_df)}")
    print(f"Date range: {games_df['date'].min().date()} to {games_df['date'].max().date()}")
    print(f"Average total runs: {games_df['total_runs'].mean():.2f}")
    print(f"Total runs range: {games_df['total_runs'].min()} - {games_df['total_runs'].max()}")
    
    # Show distribution of game totals
    print(f"\n🎯 GAME TOTALS DISTRIBUTION:")
    total_counts = games_df['total_runs'].value_counts().sort_index()
    for total, count in total_counts.head(10).items():
        print(f"{total} runs: {count} games")
    
    # Save the data
    output_path = Path('data') / 'season_games_2025.parquet'
    output_path.parent.mkdir(exist_ok=True)
    games_df.to_parquet(output_path, index=False)
    print(f"\n💾 Saved season data to: {output_path}")
    
    # Also save as CSV for easy viewing
    csv_path = Path('data') / 'season_games_2025.csv'
    games_df.to_csv(csv_path, index=False)
    print(f"💾 Saved CSV version to: {csv_path}")
    
    return games_df

if __name__ == '__main__':
    main()
