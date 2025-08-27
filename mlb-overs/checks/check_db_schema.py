#!/usr/bin/env python3
"""Quick database schema checker"""

import psycopg2

try:
    conn = psycopg2.connect(
        host='localhost', 
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )
    cursor = conn.cursor()
    
    # Check available tables
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
    tables = [row[0] for row in cursor.fetchall()]
    print("Available tables:", tables)
    
    # If enhanced_games exists, check its columns
    if 'enhanced_games' in tables:
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY ordinal_position LIMIT 50")
        columns = [row[0] for row in cursor.fetchall()]
        print("\nFirst 50 columns in enhanced_games:")
        for i, col in enumerate(columns):
            print(f"{i+1:2d}. {col}")
    
    # Check for a sample row to see actual data
    cursor.execute("SELECT * FROM enhanced_games LIMIT 1")
    sample = cursor.fetchone()
    if sample:
        print(f"\nSample row has {len(sample)} values")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    if 'conn' in locals():
        conn.close()
