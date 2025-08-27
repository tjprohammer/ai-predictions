import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

with engine.connect() as conn:
    # Check pitcher_comprehensive_stats columns
    print('Columns in pitcher_comprehensive_stats:')
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'pitcher_comprehensive_stats' ORDER BY column_name"))
    comp_cols = [row[0] for row in result]
    for col in comp_cols:
        print(f'  {col}')
    
    print('\nColumns in pitcher_daily_rolling:')
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'pitcher_daily_rolling' ORDER BY column_name"))
    daily_cols = [row[0] for row in result]
    for col in daily_cols:
        print(f'  {col}')
    
    # Check sample data from pitcher_comprehensive_stats
    print('\nSample data from pitcher_comprehensive_stats:')
    result = conn.execute(text("SELECT * FROM pitcher_comprehensive_stats LIMIT 3"))
    for row in result:
        print(f'  {dict(row)}')
        break
