import os
import pandas as pd
from sqlalchemy import create_engine, text

# Connect to database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
engine = create_engine(DATABASE_URL)

# Check if RPG data exists for today
query = text("""
SELECT game_id, home_team, away_team,
       home_team_runs_pg, away_team_runs_pg,
       CASE WHEN home_team_runs_pg IS NOT NULL THEN 1 ELSE 0 END + 
       CASE WHEN away_team_runs_pg IS NOT NULL THEN 1 ELSE 0 END as rpg_count
FROM legitimate_game_features
WHERE date = '2025-08-16'
ORDER BY game_id
""")

df = pd.read_sql(query, engine)
print(f'Games for 2025-08-16: {len(df)}')
if len(df) > 0:
    rpg_sum = df['rpg_count'].sum()
    total_possible = len(df) * 2
    print(f'RPG data coverage: {rpg_sum}/{total_possible} values')
    print()
    print(df[['home_team', 'away_team', 'home_team_runs_pg', 'away_team_runs_pg', 'rpg_count']].head(3))
    print()
    
    # Check RPG ranges
    home_min, home_max = df['home_team_runs_pg'].min(), df['home_team_runs_pg'].max()
    away_min, away_max = df['away_team_runs_pg'].min(), df['away_team_runs_pg'].max()
    combined_min = (df['home_team_runs_pg'] + df['away_team_runs_pg']).min()
    combined_max = (df['home_team_runs_pg'] + df['away_team_runs_pg']).max()
    
    print(f'Home RPG range: {home_min:.2f} - {home_max:.2f}')
    print(f'Away RPG range: {away_min:.2f} - {away_max:.2f}')
    print(f'Combined RPG range: {combined_min:.2f} - {combined_max:.2f}')
    
    # Check variance in combined RPG
    combined_rpg = df['home_team_runs_pg'] + df['away_team_runs_pg']
    print(f'Combined RPG std: {combined_rpg.std():.3f}')
else:
    print('No games found for 2025-08-16')

engine.dispose()
