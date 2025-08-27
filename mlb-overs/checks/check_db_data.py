#!/usr/bin/env python3
"""Quick database check script"""

from sqlalchemy import create_engine, text
import pandas as pd
import os

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

print("Enhanced Games Pitcher Data:")
df = pd.read_sql("""
    SELECT game_id, home_team, away_team, 
           home_sp_season_era, away_sp_season_era, 
           home_sp_whip, away_sp_whip,
           home_sp_id, away_sp_id,
           home_team_runs, away_team_runs
    FROM enhanced_games 
    WHERE date = '2025-08-18' 
    LIMIT 5
""", engine)
print(df)

print("\nColumn mapping issue:")
print("Database has: home_sp_season_era, away_sp_season_era")
print("Predictor expects: home_sp_era, away_sp_era")
print("Need to fix the column mapping in the data loading process")
