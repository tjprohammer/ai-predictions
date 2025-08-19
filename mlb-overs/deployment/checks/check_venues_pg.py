#!/usr/bin/env python3

import pandas as pd
from sqlalchemy import create_engine, text

def main():
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    with engine.connect() as conn:
        df = pd.read_sql(text('SELECT home_team, venue FROM enhanced_games WHERE date = :date'), 
                        conn, params={'date': '2025-08-16'})
        print("Venues today:")
        for _, row in df.iterrows():
            print(f"  {row['home_team']}: {row['venue']}")
        
        unique_venues = df['venue'].unique()
        print(f"\nUnique venues: {len(unique_venues)}")
        for venue in unique_venues:
            print(f"  {venue}")

if __name__ == "__main__":
    main()
