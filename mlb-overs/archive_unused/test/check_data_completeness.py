from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np

engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

with engine.begin() as conn:
    # Check today's games data completeness
    print("=== TODAY'S GAMES DATA COMPLETENESS ===")
    today_games = pd.read_sql("SELECT * FROM enhanced_games WHERE date = '2025-08-13'", conn)
    
    print(f"Games found for today: {len(today_games)}")
    if len(today_games) > 0:
        print(f"Total columns: {len(today_games.columns)}")
        print("\nColumn completeness (non-null values):")
        for col in today_games.columns:
            non_null = today_games[col].notna().sum()
            print(f"  {col:25}: {non_null:2d}/{len(today_games):2d} ({non_null/len(today_games)*100:4.0f}%)")
        
        print("\nSample game data:")
        sample = today_games.iloc[0]
        print(f"Game: {sample['away_team']} @ {sample['home_team']}")
        print(f"Home Pitcher: {sample['home_sp_name']} (ERA: {sample['home_sp_season_era']})")
        print(f"Away Pitcher: {sample['away_sp_name']} (ERA: {sample['away_sp_season_era']})")
        print(f"Weather: {sample['weather_condition']}, {sample['temperature']}°F")
        print(f"Venue: {sample['venue_name']}")
    
    # Check historical games for feature availability
    print("\n=== HISTORICAL GAMES DATA COMPLETENESS ===")
    historical = pd.read_sql("SELECT * FROM enhanced_games WHERE total_runs IS NOT NULL LIMIT 100", conn)
    
    print(f"Historical games sample: {len(historical)}")
    if len(historical) > 0:
        print("\nHistorical column completeness:")
        for col in ['home_sp_season_era', 'away_sp_season_era', 'home_team_hits', 'away_team_hits', 
                   'home_team_rbi', 'away_team_rbi', 'weather_condition', 'venue_name']:
            if col in historical.columns:
                non_null = historical[col].notna().sum()
                print(f"  {col:25}: {non_null:2d}/{len(historical):2d} ({non_null/len(historical)*100:4.0f}%)")
            else:
                print(f"  {col:25}: MISSING COLUMN")
