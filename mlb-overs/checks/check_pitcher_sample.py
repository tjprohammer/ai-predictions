import os
from sqlalchemy import create_engine, text

engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

with engine.connect() as conn:
    # Check recent data in pitcher_daily_rolling
    result = conn.execute(text("SELECT COUNT(*) as count, MAX(stat_date) as latest_date FROM pitcher_daily_rolling"))
    row = result.fetchone()
    print(f'pitcher_daily_rolling: {row[0]} records, latest: {row[1]}')
    
    # Check sample of recent data
    result = conn.execute(text("SELECT pitcher_id, era, whip, k_per_9, bb_per_9, hr_per_9 FROM pitcher_daily_rolling WHERE stat_date >= '2025-08-10' LIMIT 5"))
    print('Recent pitcher_daily_rolling sample:')
    for row in result:
        print(f'  pitcher {row[0]}: ERA={row[1]}, K/9={row[3]}, BB/9={row[4]}, HR/9={row[5]}')
    
    # Check what pitchers are starting today
    result = conn.execute(text("SELECT DISTINCT home_sp_id, away_sp_id FROM enhanced_games WHERE date = '2025-08-23'"))
    print('\nPitchers starting today:')
    starting_pitchers = set()
    for row in result:
        if row[0]: starting_pitchers.add(row[0])
        if row[1]: starting_pitchers.add(row[1])
    print(f'  {len(starting_pitchers)} unique starting pitchers')
    
    # Check if we have recent data for today's pitchers
    pitcher_list = ','.join(str(p) for p in list(starting_pitchers)[:5])  # Check first 5
    result = conn.execute(text(f"SELECT pitcher_id, era, k_per_9, bb_per_9, hr_per_9, stat_date FROM pitcher_daily_rolling WHERE pitcher_id IN ({pitcher_list}) ORDER BY stat_date DESC LIMIT 10"))
    print('Recent data for today\'s pitchers:')
    for row in result:
        print(f'  pitcher {row[0]}: ERA={row[1]}, K/9={row[2]}, BB/9={row[3]}, HR/9={row[4]} on {row[5]}')
