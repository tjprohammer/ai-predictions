import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

# Check what pitcher stats tables we have
with engine.connect() as conn:
    result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' AND (table_name LIKE '%pitcher%' OR table_name LIKE '%player%' OR table_name LIKE '%stats%') ORDER BY table_name"))
    print('Available stats tables:')
    for row in result:
        print(f'  {row[0]}')
    
    # Check what columns pitcher_rolling_stats has
    print('\nColumns in pitcher_rolling_stats:')
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'pitcher_rolling_stats' ORDER BY column_name"))
    for row in result:
        print(f'  {row[0]}')
