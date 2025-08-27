#!/usr/bin/env python3

import psycopg2
import pandas as pd

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    database='mlb_data',
    user='postgres',
    password='your_password_here'
)

# Check today's games
today_query = '''
SELECT home_team, away_team, predicted_total, predicted_total_learning, market_total, 
       prediction_timestamp, game_time_utc, date
FROM enhanced_games 
WHERE date = '2025-08-23'
ORDER BY game_time_utc
'''

try:
    df = pd.read_sql(today_query, conn)
    print(f'Found {len(df)} games for today (Aug 23):')
    if len(df) > 0:
        for _, row in df.iterrows():
            away = row['away_team']
            home = row['home_team']
            market = row['market_total']
            enhanced = row['predicted_total']
            learning = row['predicted_total_learning']
            print(f'{away} @ {home} - Market: {market}, Enhanced: {enhanced}, Learning: {learning}')
    else:
        print('No games found for today')
        
    # Check yesterday's games with results for learning
    yesterday_query = '''
    SELECT home_team, away_team, predicted_total, predicted_total_learning, market_total, 
           total_runs, home_score, away_score
    FROM enhanced_games 
    WHERE date = '2025-08-22' AND home_score IS NOT NULL
    ORDER BY game_id
    '''
    
    df_yesterday = pd.read_sql(yesterday_query, conn)
    print(f'\nFound {len(df_yesterday)} completed games from yesterday:')
    for _, row in df_yesterday.iterrows():
        away = row['away_team'] 
        home = row['home_team']
        actual = row['total_runs']
        enhanced = row['predicted_total'] 
        learning = row['predicted_total_learning']
        market = row['market_total']
        print(f'{away} @ {home} - Actual: {actual}, Market: {market}, Enhanced: {enhanced}, Learning: {learning}')
        
    conn.close()
except Exception as e:
    print(f'Error: {e}')
