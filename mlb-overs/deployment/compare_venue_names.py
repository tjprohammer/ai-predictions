#!/usr/bin/env python3
"""Compare venue names with BALLPARK_FACTORS keys."""

from sqlalchemy import create_engine
import pandas as pd
import os

def main():
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    # Get all venue names for today
    venues = pd.read_sql("SELECT DISTINCT venue_name FROM enhanced_games WHERE date = '2025-08-16' ORDER BY venue_name", engine)
    print('VENUE_NAMES FROM DATABASE:')
    for venue in venues['venue_name']:
        print(f'  "{venue}"')
    
    print('\nBALLPARK_FACTORS KEYS FROM ENHANCED_FEATURE_PIPELINE:')
    ballpark_keys = [
        'Coors Field', 'Fenway Park', 'Chase Field', 'Dodger Stadium',
        'Great American Ball Park', 'Target Field', 'Angel Stadium',
        'Kauffman Stadium', 'loanDepot park', 'Oracle Park', 'Progressive Field',
        'Busch Stadium', 'Citi Field', 'Wrigley Field', 'Nationals Park',
        'Rogers Centre', 'Daikin Park'
    ]
    
    for key in ballpark_keys:
        print(f'  "{key}"')
    
    print('\nMATCHING ANALYSIS:')
    for venue in venues['venue_name']:
        found = venue in ballpark_keys
        print(f'  "{venue}" -> Found: {found}')

if __name__ == "__main__":
    main()
