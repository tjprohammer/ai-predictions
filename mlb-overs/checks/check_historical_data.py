#!/usr/bin/env python3
"""
Check historical data availability for training
"""
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Connect to database
engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

print('=== Historical Data Inventory ===')

# Check enhanced_games table
with engine.connect() as conn:
    # Count completed games with outcomes
    games_query = 'SELECT COUNT(*) as total, MIN(game_date) as earliest, MAX(game_date) as latest FROM enhanced_games WHERE home_score IS NOT NULL'
    result = pd.read_sql(games_query, conn)
    total_games = result.iloc[0]['total']
    earliest = result.iloc[0]['earliest']
    latest = result.iloc[0]['latest']
    print(f'Enhanced games with outcomes: {total_games} games')
    print(f'Date range: {earliest} to {latest}')
    
    # Check if we have pitcher data
    pitcher_query = 'SELECT COUNT(*) as with_pitchers FROM enhanced_games WHERE home_sp_id IS NOT NULL AND away_sp_id IS NOT NULL AND home_score IS NOT NULL'
    result = pd.read_sql(pitcher_query, conn)
    with_pitchers = result.iloc[0]['with_pitchers']
    print(f'Games with pitcher IDs: {with_pitchers} games')
    
    # Check over/under lines availability
    ou_query = 'SELECT COUNT(*) as with_lines FROM enhanced_games WHERE over_under_line IS NOT NULL AND home_score IS NOT NULL'
    result = pd.read_sql(ou_query, conn)
    with_lines = result.iloc[0]['with_lines']
    print(f'Games with O/U lines: {with_lines} games')
    
    # Sample some recent completed games
    sample_query = '''
    SELECT game_date, home_team, away_team, home_score, away_score, 
           over_under_line, home_sp_id, away_sp_id, venue_id,
           temperature, wind_speed, weather_condition
    FROM enhanced_games 
    WHERE home_score IS NOT NULL 
    ORDER BY game_date DESC 
    LIMIT 5
    '''
    recent_games = pd.read_sql(sample_query, conn)
    print(f'\n=== Recent Completed Games Sample ===')
    for _, game in recent_games.iterrows():
        total_runs = game['home_score'] + game['away_score']
        line = game['over_under_line'] if pd.notna(game['over_under_line']) else 'N/A'
        pitcher_info = f"SP IDs: {game['home_sp_id']}, {game['away_sp_id']}" if pd.notna(game['home_sp_id']) else "No pitcher IDs"
        weather = f"{game['temperature']}°F" if pd.notna(game['temperature']) else "No weather"
        print(f'{game["game_date"]}: {game["away_team"]} @ {game["home_team"]} = {total_runs} runs (O/U: {line})')
        print(f'  {pitcher_info}, {weather}')
    
    # Check what columns we have in enhanced_games
    columns_query = '''
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    ORDER BY ordinal_position
    '''
    columns_df = pd.read_sql(columns_query, conn)
    print(f'\n=== Enhanced Games Table Columns ===')
    print(f'Total columns: {len(columns_df)}')
    key_columns = ['game_date', 'home_team', 'away_team', 'home_score', 'away_score', 
                   'over_under_line', 'home_sp_id', 'away_sp_id', 'venue_id', 
                   'temperature', 'wind_speed', 'weather_condition', 'home_team_id', 'away_team_id']
    print('Key columns for training:')
    for col in key_columns:
        has_col = col in columns_df['column_name'].values
        print(f'  {col}: {"✓" if has_col else "✗"}')

print('\n=== Training Data Strategy ===')
print('We can create training data by:')
print('1. Using completed games from enhanced_games table')
print('2. For each historical game, generate bullpen features using CURRENT season stats')
print('3. This gives us legitimate pre-game features + actual outcomes')
print('4. Train model on this historical data with bullpen enhancement')
