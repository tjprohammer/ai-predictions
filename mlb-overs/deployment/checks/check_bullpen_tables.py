import os
from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

# Check bullpen tables for today's data
with engine.connect() as conn:
    print('=== BULLPENS_DAILY TABLE ===')
    try:
        bullpens_query = '''
        SELECT * FROM bullpens_daily 
        WHERE date >= '2025-08-20' 
        ORDER BY date DESC 
        LIMIT 5
        '''
        bullpens_result = conn.execute(text(bullpens_query))
        cols = bullpens_result.keys()
        print(f'Columns: {list(cols)}')
        
        for i, row in enumerate(bullpens_result):
            if i < 3:  # Show first 3 rows
                print(f'Row {i+1}: {dict(zip(cols, row))}')
    except Exception as e:
        print(f'Error: {e}')
    
    print('\n=== BULLPEN_PERFORMANCE TABLE ===')
    try:
        perf_query = '''
        SELECT * FROM bullpen_performance 
        ORDER BY date DESC 
        LIMIT 3
        '''
        perf_result = conn.execute(text(perf_query))
        cols = perf_result.keys()
        print(f'Columns: {list(cols)}')
        
        for i, row in enumerate(perf_result):
            if i < 3:  # Show first 3 rows
                print(f'Row {i+1}: {dict(zip(cols, row))}')
    except Exception as e:
        print(f'Error: {e}')
    
    # Check if today's games need bullpen data populated
    print('\n=== CHECKING TODAYS GAMES BULLPEN STATUS ===')
    today_query = '''
    SELECT 
        home_team,
        away_team,
        home_bp_k,
        home_bp_bb,
        home_bp_h,
        home_bullpen_era_l30,
        away_bullpen_era_l30
    FROM enhanced_games 
    WHERE date = '2025-08-23'
    ORDER BY game_id
    LIMIT 3
    '''
    
    today_result = conn.execute(text(today_query))
    for i, row in enumerate(today_result):
        print(f'Game {i+1}: {row[0]} vs {row[1]}')
        print(f'  BP K/BB/H: {row[2]}/{row[3]}/{row[4]}')
        print(f'  BP ERA L30: Home={row[5]}, Away={row[6]}')
