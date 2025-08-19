#!/usr/bin/env python3

import psycopg2

def check_pitcher_columns():
    """Check what pitcher columns exist in enhanced_games table"""
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser',
            password='mlbpass'
        )
        cur = conn.cursor()
        
        # Get all columns that contain 'pitcher' or 'era'
        cur.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        AND (column_name LIKE '%pitcher%' OR column_name LIKE '%era%')
        ORDER BY column_name;
        """)
        
        print('=== PITCHER/ERA COLUMNS IN enhanced_games ===')
        pitcher_cols = cur.fetchall()
        for row in pitcher_cols:
            print(f'{row[0]}: {row[1]}')
        
        if not pitcher_cols:
            print('No pitcher columns found! Let me check all columns...')
            cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games'
            ORDER BY column_name;
            """)
            
            print('\n=== ALL COLUMNS IN enhanced_games ===')
            for row in cur.fetchall():
                print(row[0])
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_pitcher_columns()
