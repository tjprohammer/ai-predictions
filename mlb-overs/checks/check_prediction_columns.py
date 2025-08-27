import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check enhanced_games table schema
with engine.connect() as conn:
    result = conn.execute(text('''
    SELECT column_name, data_type, is_nullable 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    AND column_name LIKE '%predicted%'
    ORDER BY column_name
    '''))
    df = pd.DataFrame(result.fetchall(), columns=result.keys())
print('Prediction columns in enhanced_games:')
print('=' * 40)
print(df.to_string(index=False))

# Also check if there are any learning-related columns
with engine.connect() as conn:
    result2 = conn.execute(text('''
    SELECT column_name, data_type, is_nullable 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    AND (column_name LIKE '%learning%' OR column_name LIKE '%original%')
    ORDER BY column_name
    '''))
    df2 = pd.DataFrame(result2.fetchall(), columns=result2.keys())
print('\n\nLearning/original columns in enhanced_games:')
print('=' * 40)
print(df2.to_string(index=False))
