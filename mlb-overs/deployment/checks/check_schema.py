#!/usr/bin/env python3
import os
import pandas as pd
from sqlalchemy import create_engine, text

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check what SP columns exist in the enhanced_games table
query = text("""SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'enhanced_games' 
                AND (column_name LIKE '%sp_%' OR column_name LIKE '%pitcher%') 
                ORDER BY column_name""")
sp_columns = pd.read_sql(query, engine)
print('Starting Pitcher Columns in enhanced_games:')
for col in sp_columns['column_name']:
    print(f'  {col}')

# Check what gets populated during ingestion
query2 = text("""SELECT column_name FROM information_schema.columns 
                 WHERE table_name = 'enhanced_games' 
                 AND column_name LIKE '%umpire%' 
                 ORDER BY column_name""")
umpire_columns = pd.read_sql(query2, engine)
print('\nUmpire Columns in enhanced_games:')
for col in umpire_columns['column_name']:
    print(f'  {col}')

# Check what gets populated during ingestion  
query3 = text("""SELECT column_name FROM information_schema.columns 
                 WHERE table_name = 'enhanced_games' 
                 AND (column_name LIKE '%bp_%' OR column_name LIKE '%bullpen%') 
                 ORDER BY column_name""")
bp_columns = pd.read_sql(query3, engine)
print('\nBullpen Columns in enhanced_games:')
for col in bp_columns['column_name']:
    print(f'  {col}')
