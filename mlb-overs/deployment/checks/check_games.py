#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text
import pandas as pd

# Database connection
db_url = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(db_url)

# Check total games today
games_query = """
SELECT COUNT(*) as total_games 
FROM games 
WHERE date = '2025-08-13'
"""
games = pd.read_sql(text(games_query), engine)
print(f"Total games in database today: {games.iloc[0]['total_games']}")

# Check games with betting lines
betting_query = """
SELECT COUNT(*) as games_with_betting 
FROM games g
LEFT JOIN (
    SELECT DISTINCT game_id 
    FROM markets_totals 
    WHERE date = '2025-08-13' AND k_total IS NOT NULL
) mt ON g.game_id = mt.game_id
WHERE g.date = '2025-08-13' AND mt.game_id IS NOT NULL
"""
betting = pd.read_sql(text(betting_query), engine)
print(f"Games with betting lines: {betting.iloc[0]['games_with_betting']}")

# Show all games today
all_games_query = """
SELECT g.game_id, g.home_team, g.away_team, 
       COALESCE(mt.k_total, g.close_total) as betting_line
FROM games g
LEFT JOIN (
    SELECT game_id, k_total,
           ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY updated_at DESC) as rn
    FROM markets_totals 
    WHERE date = '2025-08-13'
) mt ON g.game_id = mt.game_id AND mt.rn = 1
WHERE g.date = '2025-08-13'
ORDER BY g.game_id
"""
all_games = pd.read_sql(text(all_games_query), engine)
print(f"\nAll games today ({len(all_games)}):")
for i, game in all_games.iterrows():
    line = game['betting_line'] if pd.notna(game['betting_line']) else 'No Line'
    print(f"  {game['away_team']} @ {game['home_team']}: {line}")
