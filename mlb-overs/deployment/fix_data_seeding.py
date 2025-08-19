#!/usr/bin/env python3
"""Fix data seeding"""

from sqlalchemy import create_engine, text
import pandas as pd
import os

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

print("1. Checking current legitimate_game_features data:")
try:
    df = pd.read_sql("""
        SELECT game_id, home_sp_season_era, away_sp_season_era 
        FROM legitimate_game_features 
        WHERE date = '2025-08-18' 
        LIMIT 5
    """, engine)
    print(df)
except Exception as e:
    print(f"Error reading LGF: {e}")

print("\n2. Clearing and re-seeding legitimate_game_features:")
with engine.begin() as conn:
    # Clear existing data for today
    result = conn.execute(text("DELETE FROM legitimate_game_features WHERE date = '2025-08-18'"))
    print(f"Deleted {result.rowcount} existing rows")
    
    # Re-seed with pitcher data (minimal columns first)
    result = conn.execute(text("""
        INSERT INTO legitimate_game_features (
            game_id, "date", home_team, away_team, market_total,
            home_sp_season_era, away_sp_season_era, 
            home_sp_whip, away_sp_whip
        )
        SELECT 
            eg.game_id, eg."date", eg.home_team, eg.away_team, eg.market_total,
            eg.home_sp_season_era, eg.away_sp_season_era,
            eg.home_sp_whip, eg.away_sp_whip
        FROM enhanced_games eg
        WHERE eg."date" = '2025-08-18'
    """))
    print(f"Inserted {result.rowcount} rows with pitcher data")

print("\n3. Verifying seeded data:")
df2 = pd.read_sql("""
    SELECT game_id, home_team, away_team,
           home_sp_season_era, away_sp_season_era 
    FROM legitimate_game_features 
    WHERE date = '2025-08-18' 
    LIMIT 5
""", engine)
print(df2)
