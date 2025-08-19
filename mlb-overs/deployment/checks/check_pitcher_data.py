#!/usr/bin/env python3
"""
Check the populated pitcher data in the database
"""

from sqlalchemy import create_engine, text
import pandas as pd
import os

# Use PostgreSQL from .env
eng = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

def check_pitcher_data():
    with eng.connect() as cx:
        # Check pitcher starts data
        count_query = text('SELECT COUNT(*) as cnt FROM pitchers_starts')
        count_result = pd.read_sql(count_query, cx)
        print(f'Total pitcher starts records: {count_result.iloc[0]["cnt"]}')
        
        # Check if comprehensive stats table exists
        try:
            comp_count_query = text('SELECT COUNT(*) as cnt FROM pitcher_comprehensive_stats')
            comp_count_result = pd.read_sql(comp_count_query, cx)
            print(f'Total comprehensive pitcher records: {comp_count_result.iloc[0]["cnt"]}')
        except Exception as e:
            print(f'Comprehensive stats table: {e}')
        
        # Check recent pitcher starts with actual data
        recent_query = text('SELECT pitcher_id, date, ip, er, era_game FROM pitchers_starts WHERE ip IS NOT NULL AND er IS NOT NULL ORDER BY date DESC LIMIT 10')
        recent_result = pd.read_sql(recent_query, cx)
        print('\nRecent pitcher starts with actual data:')
        print(recent_result.to_string(index=False))
        
        # Count pitchers with actual performance data
        data_query = text('SELECT COUNT(DISTINCT pitcher_id) as unique_pitchers FROM pitchers_starts WHERE ip IS NOT NULL AND er IS NOT NULL')
        data_result = pd.read_sql(data_query, cx)
        print(f'\nUnique pitchers with actual IP/ER data: {data_result.iloc[0]["unique_pitchers"]}')

if __name__ == "__main__":
    check_pitcher_data()
