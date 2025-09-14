import os
from datetime import date
import pandas as pd
from sqlalchemy import create_engine, text

TARGET = '2025-09-12'  # change if needed
DATABASE_URL = os.getenv('DATABASE_URL','postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

q = text("""
SELECT game_id, "date", home_team, away_team, market_total,
       predicted_total, predicted_total_learning,
       over_odds, under_odds
FROM enhanced_games
WHERE "date" = :d
ORDER BY game_id
""")

df = pd.read_sql(q, engine, params={'d': TARGET})
if df.empty:
    print(f'No rows found for {TARGET}')
else:
    # Round numeric prediction columns
    for col in ['market_total','predicted_total','predicted_total_learning']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').round(2)
    print(df.to_string(index=False))
