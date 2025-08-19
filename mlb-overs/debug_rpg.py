import sqlite3
import pandas as pd

# Connect to actual database
conn = sqlite3.connect('data/mlb_data.db')

# List all tables
print('Available tables:')
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(tables['name'].tolist())

# Check for RPG columns in each table  
print('\nTables with RPG/runs_pg columns:')
for table in tables['name']:
    try:
        df = pd.read_sql(f'PRAGMA table_info({table})', conn)
        rpg_cols = df[df['name'].str.contains('rpg|runs_pg', case=False, na=False)]['name'].tolist()
        if rpg_cols:
            print(f'  {table}: {rpg_cols}')
    except Exception as e:
        print(f'  {table}: Error - {e}')

# Check what data exists for 2025-08-16
print('\nData for 2025-08-16:')
try:
    # Check daily_api
    df = pd.read_sql("SELECT home_team, away_team FROM daily_api WHERE game_date='2025-08-16' LIMIT 3", conn)
    print(f"daily_api: {len(df)} games")
    print(df.head())
except Exception as e:
    print(f"daily_api error: {e}")

# Check what columns exist in main tables
for table in ['games', 'team_stats', 'daily_api', 'final_data']:
    try:
        df = pd.read_sql(f'PRAGMA table_info({table})', conn)
        cols = df['name'].tolist()
        rpg_related = [c for c in cols if 'rpg' in c.lower() or 'runs' in c.lower()]
        if rpg_related:
            print(f'\n{table} runs-related columns: {rpg_related}')
    except:
        pass

conn.close()
