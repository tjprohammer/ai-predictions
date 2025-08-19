#!/usr/bin/env python3

import pandas as pd
import os
from sqlalchemy import create_engine

# Connect to DB
db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(db_url)

# Check what team offense columns actually exist today
sample_query = '''
SELECT *
FROM enhanced_games 
WHERE date = CURRENT_DATE
LIMIT 1
'''

sample = pd.read_sql(sample_query, engine)
print('🔍 Available team offense columns in enhanced_games:')
team_cols = [col for col in sample.columns if 'team' in col and ('woba' in col or 'power' in col or 'avg' in col or 'runs' in col)]
print('Team offense columns:', team_cols)

if team_cols:
    # Now check variance for available team columns
    variance_query = f'''
    SELECT 
        {', '.join(team_cols[:10])}
    FROM enhanced_games 
    WHERE date = CURRENT_DATE
    '''
    
    df = pd.read_sql(variance_query, engine)
    print(f'\n📊 Variance analysis for {len(df)} games:')
    for col in team_cols[:10]:
        if col in df.columns:
            std = df[col].std()
            print(f'  {col}: std={std:.6f}')
else:
    print('No team offense columns found!')
