import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check what predictions exist
query = '''
SELECT 
    home_team,
    away_team,
    market_total,
    predicted_total,
    predicted_total_learning,
    confidence,
    recommendation
FROM enhanced_games 
WHERE date = '2025-08-23'
ORDER BY game_id
'''

df = pd.read_sql(query, engine)
print('Current predictions in database:')
print('=' * 50)
print(df.to_string(index=False))
print(f'\nRows with predicted_total: {(~df["predicted_total"].isna()).sum()}')
print(f'Rows with predicted_total_learning: {(~df["predicted_total_learning"].isna()).sum()}')
