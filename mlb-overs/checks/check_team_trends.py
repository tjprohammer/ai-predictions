import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

with engine.connect() as conn:
    # Check columns in team_recent_trends
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'team_recent_trends' ORDER BY column_name"))
    print('Columns in team_recent_trends:')
    for row in result:
        print(f'  {row[0]}')
    
    # Check sample data
    result = conn.execute(text("SELECT * FROM team_recent_trends LIMIT 3"))
    print('\nSample data:')
    cols = result.keys()
    rows = result.fetchall()
    if rows:
        for i, row in enumerate(rows):
            print(f'Row {i+1}: {dict(zip(cols, row))}')
            if i == 0:  # Only show first row details
                break
