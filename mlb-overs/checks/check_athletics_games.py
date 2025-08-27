#!/usr/bin/env python3
import psycopg2

# Database connection function
def get_connection():
    """Get a database connection"""
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser',
        password='mlbpass',
        port=5432
    )

with get_connection() as conn:
    with conn.cursor() as cursor:
        cursor.execute("""
            SELECT game_id, home_team, away_team 
            FROM legitimate_game_features 
            WHERE date = '2025-08-23' 
              AND (home_team = 'Athletics' OR away_team = 'Athletics')
        """)
        games = cursor.fetchall()
        print(f'Games with Athletics: {games}')
        
        # Get total games for context
        cursor.execute("SELECT COUNT(*) FROM legitimate_game_features WHERE date = '2025-08-23'")
        total = cursor.fetchone()[0]
        print(f'Total games today: {total}')
        print(f'Games with Athletics: {len(games)}')
        print(f'Games we CAN process: {total - len(games)}')
