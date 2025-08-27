#!/usr/bin/env python3
"""
Check which games are missing trend data and why
"""

import psycopg2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_missing_trends():
    """Check which games are missing trend data"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host="localhost",
            database="mlb",
            user="mlbuser", 
            password="mlbpass"
        )
        cursor = conn.cursor()
        
        # Find games without trends
        cursor.execute("""
            SELECT id, date, home_team, away_team, home_score, away_score, total_runs
            FROM enhanced_games 
            WHERE home_team_runs_l7 IS NULL
            ORDER BY date
            LIMIT 20
        """)
        
        missing_games = cursor.fetchall()
        print(f"🔍 Games Missing Trends ({len(missing_games)} found):")
        print("ID | Date | Home vs Away | Score | Total")
        print("-" * 60)
        
        for game in missing_games:
            game_id, date, home, away, home_score, away_score, total = game
            print(f"{game_id} | {date} | {home} vs {away} | {home_score}-{away_score} | {total}")
        
        # Check if these are the earliest games (no previous data for trends)
        cursor.execute("SELECT MIN(date), MAX(date) FROM enhanced_games")
        date_range = cursor.fetchone()
        print(f"\n📅 Full dataset date range: {date_range[0]} to {date_range[1]}")
        
        # Check how many games each team has for trend calculation
        cursor.execute("""
            SELECT home_team, COUNT(*) as games
            FROM enhanced_games 
            WHERE home_team_runs_l7 IS NULL
            GROUP BY home_team
            ORDER BY games DESC
        """)
        
        team_counts = cursor.fetchall()
        if team_counts:
            print(f"\n📊 Teams with missing trends:")
            for team, count in team_counts:
                print(f"  {team}: {count} games")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Missing trends check failed: {e}")

if __name__ == "__main__":
    check_missing_trends()
