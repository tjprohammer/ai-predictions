import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

with engine.connect() as conn:
    # Check for team stats tables
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND table_name LIKE '%team%' ORDER BY table_name"))
    print('Team tables:')
    for row in result:
        print(f'  {row[0]}')
    
    # Check what columns are in teams_l30 table if it exists
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'teams_l30' ORDER BY column_name"))
    cols = [row[0] for row in result]
    if cols:
        print('\nColumns in teams_l30:')
        for col in cols:
            print(f'  {col}')
    else:
        print('\nteams_l30 table not found')
