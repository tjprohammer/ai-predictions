#!/usr/bin/env python3

import pandas as pd
from sqlalchemy import create_engine

def check_training_data():
    """Check current training data status"""
    
    engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    
    print('🔍 CHECKING CURRENT TRAINING DATA')
    print('=' * 50)
    
    # Check legitimate_game_features table
    try:
        features_df = pd.read_sql('SELECT * FROM legitimate_game_features ORDER BY date', engine)
        print(f'📊 Training table: {len(features_df):,} total games')
        print(f'📅 Date range: {features_df["date"].min()} to {features_df["date"].max()}')
        print(f'🏃 Avg runs/game: {features_df["total_runs"].mean():.2f}')
        print()
        
        # Check by month/timeframe
        features_df['date'] = pd.to_datetime(features_df['date'])
        print('📅 Games by month:')
        monthly = features_df.groupby(features_df['date'].dt.to_period('M')).size()
        for month, count in monthly.items():
            print(f'   {month}: {count:,} games')
        print()
        
        # Check feature completeness
        print('🎯 Feature completeness (non-null %):')
        important_features = [
            'home_pitcher_season_era', 'away_pitcher_season_era',
            'home_team_runs_per_game', 'away_team_runs_per_game', 
            'ballpark_run_factor', 'temperature', 'wind_speed',
            'home_bullpen_era', 'away_bullpen_era'
        ]
        for feature in important_features:
            if feature in features_df.columns:
                completeness = (features_df[feature].notna().sum() / len(features_df)) * 100
                print(f'   {feature:<25}: {completeness:5.1f}%')
            else:
                print(f'   {feature:<25}: MISSING')
        print()
        
        # Check if we have actual historical data vs just recent games
        recent_games = features_df[features_df['date'] >= '2025-08-01']
        historical_games = features_df[features_df['date'] < '2025-08-01']
        print(f'📈 Historical games (before Aug 1): {len(historical_games):,}')
        print(f'📈 Recent games (Aug 1+): {len(recent_games):,}')
        
    except Exception as e:
        print(f'❌ Error: {e}')

    # Also check if we have enhanced_games with more complete data
    try:
        print()
        print('🔍 CHECKING ENHANCED_GAMES TABLE')
        print('=' * 50)
        enhanced_df = pd.read_sql('SELECT COUNT(*) as total, MIN(date) as min_date, MAX(date) as max_date, AVG(total_runs) as avg_runs FROM enhanced_games WHERE total_runs IS NOT NULL', engine)
        print(f'📊 Enhanced table: {enhanced_df["total"].iloc[0]:,} complete games')
        print(f'📅 Date range: {enhanced_df["min_date"].iloc[0]} to {enhanced_df["max_date"].iloc[0]}')
        print(f'🏃 Avg runs/game: {enhanced_df["avg_runs"].iloc[0]:.2f}')
        
        # Check available features in enhanced_games
        enhanced_sample = pd.read_sql('SELECT * FROM enhanced_games LIMIT 1', engine)
        enhanced_cols = list(enhanced_sample.columns)
        print(f'📋 Available columns: {len(enhanced_cols)}')
        
        # Check for key features
        key_features = ['home_pitcher_era', 'away_pitcher_era', 'ballpark_factor', 
                       'temperature', 'wind_speed', 'home_team_avg_runs', 'away_team_avg_runs']
        print('🎯 Key features in enhanced_games:')
        for feature in key_features:
            if feature in enhanced_cols:
                print(f'   ✅ {feature}')
            else:
                print(f'   ❌ {feature}')
                
    except Exception as e:
        print(f'❌ Error checking enhanced_games: {e}')

if __name__ == "__main__":
    check_training_data()
