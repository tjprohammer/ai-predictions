#!/usr/bin/env python3
"""
Test ERA calculation functions directly with the populated data
"""

from sqlalchemy import create_engine, text
import pandas as pd
import os

# Import the ERA functions from the API
import sys
sys.path.append('api')

# Database connection
eng = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))

def _era_from_rows(rows):
    """Helper function to calculate ERA from pitcher rows"""
    if rows.empty:
        return None
    total_ip = rows['ip'].sum() 
    total_er = rows['er'].sum()
    if total_ip > 0:
        return round((total_er * 9.0) / total_ip, 2)
    return None

def lastN_era(eng, pitcher_id, n=5):
    """Get last N games ERA for a pitcher"""
    with eng.connect() as cx:
        query = text("""
            SELECT ip, er, era_game FROM pitchers_starts 
            WHERE pitcher_id = :pid AND ip IS NOT NULL AND er IS NOT NULL
            ORDER BY date DESC 
            LIMIT :n
        """)
        df = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "n": n})
        return _era_from_rows(df)

def season_era_until(eng, pitcher_id, until_date=None):
    """Get season ERA until a specific date"""
    with eng.connect() as cx:
        if until_date:
            query = text("""
                SELECT ip, er FROM pitchers_starts 
                WHERE pitcher_id = :pid AND date <= :until_date 
                AND ip IS NOT NULL AND er IS NOT NULL
            """)
            df = pd.read_sql(query, cx, params={"pid": str(pitcher_id), "until_date": until_date})
        else:
            query = text("""
                SELECT ip, er FROM pitchers_starts 
                WHERE pitcher_id = :pid 
                AND ip IS NOT NULL AND er IS NOT NULL
            """)
            df = pd.read_sql(query, cx, params={"pid": str(pitcher_id)})
        return _era_from_rows(df)

def test_era_functions():
    print("Testing ERA calculation functions with 2025 data...")
    print("=" * 60)
    
    # Get some test pitchers with actual data
    with eng.connect() as cx:
        test_pitchers_query = text("""
            SELECT pitcher_id, COUNT(*) as games, SUM(ip) as total_ip, SUM(er) as total_er,
                   AVG(era_game) as avg_era_game
            FROM pitchers_starts 
            WHERE ip IS NOT NULL AND er IS NOT NULL 
              AND date >= '2025-08-01'
            GROUP BY pitcher_id 
            HAVING COUNT(*) >= 1
            ORDER BY total_ip DESC 
            LIMIT 5
        """)
        test_pitchers = pd.read_sql(test_pitchers_query, cx)
    
    if test_pitchers.empty:
        print("❌ No pitcher data found for testing")
        return
    
    print(f"Found {len(test_pitchers)} pitchers with data to test:\n")
    
    for _, pitcher in test_pitchers.iterrows():
        pitcher_id = pitcher['pitcher_id']
        games = pitcher['games']
        total_ip = pitcher['total_ip']
        total_er = pitcher['total_er']
        expected_era = (total_er * 9.0) / total_ip if total_ip > 0 else 0
        
        print(f"Pitcher ID: {pitcher_id}")
        print(f"  Games: {games}, IP: {total_ip:.1f}, ER: {total_er}, Expected ERA: {expected_era:.2f}")
        
        # Test season ERA function
        season_era = season_era_until(eng, pitcher_id)
        l5_era = lastN_era(eng, pitcher_id, 5)
        l3_era = lastN_era(eng, pitcher_id, 3)
        
        print(f"  Season ERA (calculated): {season_era}")
        print(f"  Last 5 games ERA: {l5_era}")
        print(f"  Last 3 games ERA: {l3_era}")
        
        # Check if functions return expected values
        if season_era is not None and abs(season_era - expected_era) < 0.01:
            print(f"  ✅ Season ERA calculation is correct!")
        else:
            print(f"  ❌ Season ERA mismatch - expected {expected_era:.2f}, got {season_era}")
        
        print("-" * 40)
    
    print("\nTesting completed!")

if __name__ == "__main__":
    test_era_functions()
