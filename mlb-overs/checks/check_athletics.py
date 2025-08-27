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
        cursor.execute("SELECT DISTINCT team FROM teams_offense_daily WHERE team ILIKE '%athletic%' OR team ILIKE '%oak%';")
        athletics_teams = cursor.fetchall()
        print("Teams with 'athletic' or 'oak':", athletics_teams)
        
        cursor.execute("SELECT DISTINCT team FROM teams_offense_daily WHERE team ILIKE '%a%s%' OR team ILIKE '%bay%';")
        other_teams = cursor.fetchall()
        print("Teams with 'a's or 'bay':", other_teams)
