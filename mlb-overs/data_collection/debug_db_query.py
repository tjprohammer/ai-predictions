#!/usr/bin/env python3
"""Debug database query to understand the transaction error"""

import psycopg2

def test_basic_query():
    """Test a basic query to understand the error"""
    try:
        conn = psycopg2.connect(
            host='localhost', 
            database='mlb',
            user='mlbuser', 
            password='mlbpass'
        )
        cursor = conn.cursor()
        
        # Test 1: Simple select
        print("Test 1: Basic select from enhanced_games")
        cursor.execute("SELECT game_id, home_team, away_team, date FROM enhanced_games LIMIT 5")
        results = cursor.fetchall()
        print(f"Found {len(results)} games")
        for row in results[:2]:
            print(f"  {row}")
        
        # Test 2: Check if specific columns exist
        print("\nTest 2: Check recent trends columns")
        cursor.execute("SELECT home_team_runs_l7 FROM enhanced_games WHERE home_team_runs_l7 IS NOT NULL LIMIT 5")
        l7_results = cursor.fetchall()
        print(f"Found {len(l7_results)} games with L7 data")
        
        # Test 3: Try the exact query that's failing
        print("\nTest 3: Test problematic query")
        test_team = "St. Louis Cardinals"
        test_date = "2025-03-20"
        
        cursor.execute("""
            SELECT 
                AVG(CASE WHEN home_team = %s THEN home_score ELSE away_score END) as runs_scored_l7,
                AVG(CASE WHEN home_team = %s THEN away_score ELSE home_score END) as runs_allowed_l7,
                COUNT(*) as games_l7
            FROM enhanced_games 
            WHERE (home_team = %s OR away_team = %s)
            AND date < %s
            AND date >= %s::date - INTERVAL '7 days'
            AND home_score IS NOT NULL
            ORDER BY date DESC
            LIMIT 7
        """, (test_team, test_team, test_team, test_team, test_date, test_date))
        
        result = cursor.fetchone()
        print(f"Query result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e)}")
        
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    test_basic_query()
