import os
from sqlalchemy import create_engine, text
import pandas as pd

# Use the same connection string as the predictor
engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check bullpen columns
query = """
SELECT column_name 
FROM information_schema.columns 
WHERE table_name = 'enhanced_games' 
AND column_name LIKE '%bp%' 
ORDER BY column_name
"""

with engine.connect() as conn:
    result = conn.execute(text(query))
    print('Bullpen columns in enhanced_games:')
    for row in result:
        print(f'  {row[0]}')

# Also check what the model actually expects
print('\nModel expects these features:')
expected_features = [
    'home_bp_k', 'home_bp_bb', 'home_bp_h', 
    'away_bp_k', 'away_bp_bb', 'away_bp_h'
]
for feat in expected_features:
    print(f'  {feat}')
