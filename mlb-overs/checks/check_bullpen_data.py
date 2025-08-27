import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check if there are other tables with bullpen stats
with engine.connect() as conn:
    # Check what bullpen-related tables exist
    tables_result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_name LIKE '%bullpen%' OR table_name LIKE '%bp%'"))
    tables = [row[0] for row in tables_result]
    print('Bullpen-related tables:')
    for table in tables:
        print(f'  {table}')
    
    # Look for any tables with real bullpen statistics
    print('\nSearching for tables with bullpen statistics...')
    search_query = '''
    SELECT table_name, column_name 
    FROM information_schema.columns 
    WHERE column_name LIKE '%bullpen%' OR column_name LIKE '%bp_%' 
    ORDER BY table_name, column_name
    '''
    
    search_result = conn.execute(text(search_query))
    current_table = None
    for row in search_result:
        if row[0] != current_table:
            print(f'\n  {row[0]}:')
            current_table = row[0]
        print(f'    {row[1]}')
        
    # Check if we have any actual data in these columns anywhere
    print('\nChecking for any non-zero bullpen data...')
    try:
        data_check = conn.execute(text("""
        SELECT COUNT(*) as total_rows,
               COUNT(CASE WHEN home_bp_k > 0 THEN 1 END) as home_bp_k_nonzero,
               COUNT(CASE WHEN home_bp_bb > 0 THEN 1 END) as home_bp_bb_nonzero,
               COUNT(CASE WHEN home_bp_h > 0 THEN 1 END) as home_bp_h_nonzero
        FROM enhanced_games 
        WHERE date >= '2025-08-01'
        """))
        
        for row in data_check:
            print(f'  Total rows: {row[0]}')
            print(f'  home_bp_k > 0: {row[1]}')
            print(f'  home_bp_bb > 0: {row[2]}')
            print(f'  home_bp_h > 0: {row[3]}')
    except Exception as e:
        print(f'  Error checking data: {e}')
