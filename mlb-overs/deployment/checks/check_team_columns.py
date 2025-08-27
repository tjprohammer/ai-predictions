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
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'teams_offense_daily' 
            ORDER BY ordinal_position
        """)
        cols = [row[0] for row in cursor.fetchall()]
        print('Columns in teams_offense_daily:')
        for i, col in enumerate(cols):
            print(f'  {i+1:2d}. {col}')
