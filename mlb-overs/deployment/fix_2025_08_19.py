#!/usr/bin/env python3
"""
Quick fix to clear and reload 2025-08-19 data
"""

import psycopg2
import subprocess
import sys

def clear_and_reload():
    """Clear existing data and reload"""
    
    # Connect to database
    conn = psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass'
    )
    
    try:
        with conn.cursor() as cur:
            print("🗑️ Clearing existing data for 2025-08-19...")
            
            # Clear enhanced_games
            cur.execute("DELETE FROM enhanced_games WHERE date = %s", ('2025-08-19',))
            
            # Clear legitimate_game_features  
            cur.execute("DELETE FROM legitimate_game_features WHERE date = %s", ('2025-08-19',))
            
            conn.commit()
            
            # Check what's left
            cur.execute("SELECT COUNT(*) FROM enhanced_games WHERE date = %s", ('2025-08-19',))
            eg_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM legitimate_game_features WHERE date = %s", ('2025-08-19',))
            lgf_count = cur.fetchone()[0]
            
            print(f"✅ Cleared data:")
            print(f"   enhanced_games: {eg_count} rows remaining")
            print(f"   legitimate_game_features: {lgf_count} rows remaining")
            
    finally:
        conn.close()
    
    print("\n🔄 Running games ingestor...")
    result = subprocess.run([
        sys.executable, 
        "ingestion/working_games_ingestor.py", 
        "--target-date", "2025-08-19"
    ], capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

if __name__ == "__main__":
    clear_and_reload()
