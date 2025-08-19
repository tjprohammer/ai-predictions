from sqlalchemy import create_engine
import pandas as pd

e = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

# Check enhanced_games data
sample = pd.read_sql("SELECT home_team, away_team, home_sp_season_era, away_sp_season_era FROM enhanced_games WHERE date = '2025-08-15' LIMIT 5", e)
print("Enhanced games sample:")
print(sample)

# Find all tables
tables = pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name", e)
game_tables = [t for t in tables['table_name'] if 'game' in t.lower() or 'mlb' in t.lower() or 'api' in t.lower()]
print('\nGame/API tables:', game_tables)

# Check mlb_games_today if it exists
if 'mlb_games_today' in game_tables:
    print('\nChecking mlb_games_today...')
    today_sample = pd.read_sql("SELECT * FROM mlb_games_today LIMIT 1", e)
    pitcher_cols = [col for col in today_sample.columns if 'pitcher' in col.lower() or 'starter' in col.lower()]
    print('Pitcher columns:', pitcher_cols)
    if pitcher_cols:
        print('Sample pitcher data:')
        print(today_sample[pitcher_cols])
