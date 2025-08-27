import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

with engine.connect() as conn:
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' AND column_name LIKE '%avg%' ORDER BY column_name"))
    print('Columns with avg:', [row[0] for row in result])
    
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' AND column_name LIKE '%humidity%' ORDER BY column_name"))
    print('Columns with humidity:', [row[0] for row in result])

    # Check for specific columns
    result = conn.execute(text("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' AND column_name IN ('home_team_avg', 'away_team_avg', 'humidity')"))
    print('Specific columns found:', [row[0] for row in result])
