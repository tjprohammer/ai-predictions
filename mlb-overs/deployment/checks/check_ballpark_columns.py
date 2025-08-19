#!/usr/bin/env python3
"""Check ballpark column values."""

from sqlalchemy import create_engine
import pandas as pd
import os

def main():
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    # Check ballpark-related columns
    sample = pd.read_sql("SELECT ballpark, venue, venue_name FROM enhanced_games WHERE date = '2025-08-16' LIMIT 5", engine)
    print('BALLPARK-RELATED COLUMNS:')
    print(sample.to_string())
    
    # Check if venue_name has real data
    venue_query = "SELECT DISTINCT venue_name FROM enhanced_games WHERE date = '2025-08-16' ORDER BY venue_name"
    venues = pd.read_sql(venue_query, engine)
    print('\nVENUE NAMES:')
    for venue in venues['venue_name']:
        print(f'  "{venue}"')

if __name__ == "__main__":
    main()
