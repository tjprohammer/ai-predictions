#!/usr/bin/env python3
"""
Quick status check for Phase 2 recent trends enhancement
"""

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_database_status():
    """Check the current status of trend enhancement"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        # Check what trend columns exist
        cursor.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            AND (column_name LIKE '%l7%' OR column_name LIKE '%l14%' OR column_name LIKE '%l20%')
            ORDER BY column_name
        """)
        
        trend_columns = cursor.fetchall()
        print(f"📋 Trend columns found: {len(trend_columns)}")
        for col in trend_columns:
            print(f"  - {col[0]}")
        
        # Check total games
        cursor.execute("SELECT COUNT(*) FROM enhanced_games")
        total_games = cursor.fetchone()[0]
        
        # Check games with any trend data (try different column patterns)
        if trend_columns:
            # Use first trend column found
            first_col = trend_columns[0][0]
            cursor.execute(f"SELECT COUNT(*) FROM enhanced_games WHERE {first_col} IS NOT NULL")
            trend_games = cursor.fetchone()[0]
            
            print(f"\n📊 Database Status:")
            print(f"Total games: {total_games}")
            print(f"Games with trends: {trend_games}")
            print(f"Games without trends: {total_games - trend_games}")
        
        # Check date range (check what date column exists)
        cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' AND column_name LIKE '%date%'")
        date_cols = cursor.fetchall()
        if date_cols:
            date_col = date_cols[0][0]
            cursor.execute(f"SELECT MIN({date_col}), MAX({date_col}) FROM enhanced_games")
            date_range = cursor.fetchone()
            print(f"\nDate range ({date_col}): {date_range[0]} to {date_range[1]}")
        else:
            print("\nNo date column found")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Database check failed: {e}")

if __name__ == "__main__":
    check_database_status()
