#!/usr/bin/env python3
"""
Check what tables exist in the SQLite database
"""

from sqlalchemy import create_engine, text
import pandas as pd

# Use SQLite from .env
eng = create_engine('sqlite:///mlb.db')

def check_tables():
    with eng.connect() as cx:
        # Check what tables exist
        tables_query = text("SELECT name FROM sqlite_master WHERE type='table'")
        tables_result = pd.read_sql(tables_query, cx)
        print("Available tables:")
        print(tables_result.to_string(index=False))
        
        # If pitchers_starts doesn't exist, maybe the data went somewhere else
        # Let's check games table
        try:
            games_query = text("SELECT COUNT(*) as cnt FROM games")
            games_result = pd.read_sql(games_query, cx)
            print(f"\nGames table records: {games_result.iloc[0]['cnt']}")
            
            # Sample games
            sample_games = text("SELECT * FROM games LIMIT 3")
            sample_result = pd.read_sql(sample_games, cx)
            print("\nSample games:")
            print(sample_result.to_string(index=False))
        except Exception as e:
            print(f"Games table issue: {e}")

if __name__ == "__main__":
    check_tables()
