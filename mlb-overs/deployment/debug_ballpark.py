#!/usr/bin/env python3
"""Check actual ballpark names and debug why ballpark factors aren't being applied."""

from sqlalchemy import create_engine
import pandas as pd
import os

def main():
    db_url = os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
    engine = create_engine(db_url)
    
    # Check actual ballpark names in enhanced_games
    ballpark_query = """
    SELECT DISTINCT ballpark 
    FROM enhanced_games 
    WHERE date = '2025-08-16'
    ORDER BY ballpark
    """
    
    ballparks = pd.read_sql(ballpark_query, engine)
    print('ACTUAL BALLPARK NAMES:')
    for park in ballparks['ballpark']:
        print(f'  "{park}"')
    
    # Test the ballpark matching logic
    ballpark_factors = {
        'Coors Field': {'run': 1.15, 'hr': 1.25},
        'Great American Ball Park': {'run': 1.08, 'hr': 1.12},
        'Yankee Stadium': {'run': 1.02, 'hr': 1.18},
        'Fenway Park': {'run': 1.05, 'hr': 1.10},
        'Minute Maid Park': {'run': 1.03, 'hr': 1.08},
        'Rogers Centre': {'run': 1.01, 'hr': 1.06},
        'Citizens Bank Park': {'run': 1.04, 'hr': 1.09},
        'Oriole Park at Camden Yards': {'run': 1.02, 'hr': 1.07},
        'Progressive Field': {'run': 0.98, 'hr': 0.95},
        'Kauffman Stadium': {'run': 0.97, 'hr': 0.93},
        'Oakland Coliseum': {'run': 0.95, 'hr': 0.88},
        'Tropicana Field': {'run': 0.96, 'hr': 0.90},
        'Marlins Park': {'run': 0.98, 'hr': 0.92},
        'Petco Park': {'run': 0.94, 'hr': 0.85}
    }
    
    def get_ballpark_factor(park_name, factor_type):
        if pd.isna(park_name):
            return 1.0
        # Try exact match first
        if park_name in ballpark_factors:
            return ballpark_factors[park_name][factor_type]
        # Try partial match
        for known_park in ballpark_factors:
            if known_park.lower() in str(park_name).lower() or str(park_name).lower() in known_park.lower():
                return ballpark_factors[known_park][factor_type]
        return 1.0  # Default for unknown parks
    
    print('\nTESTING BALLPARK FACTOR MATCHING:')
    for park in ballparks['ballpark']:
        run_factor = get_ballpark_factor(park, 'run')
        hr_factor = get_ballpark_factor(park, 'hr')
        print(f'  "{park}" -> run: {run_factor}, hr: {hr_factor}')

if __name__ == "__main__":
    main()
