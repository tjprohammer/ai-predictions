#!/usr/bin/env python3
from sqlalchemy import create_engine, text

engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')

with engine.connect() as conn:
    result = conn.execute(text("SELECT DISTINCT venue_name FROM enhanced_games WHERE date = '2025-08-16' ORDER BY venue_name"))
    venues = [row[0] for row in result]
    
    print('Venue names in database:')
    for v in venues:
        print(f'  "{v}"')
    
    print(f'\nTotal venues: {len(venues)}')
