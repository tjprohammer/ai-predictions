#!/usr/bin/env python3

import pandas as pd
from sqlalchemy import create_engine, text
import os

def main():
    engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))
    
    with engine.connect() as conn:
        df = pd.read_sql(text('SELECT home_sp_whip, away_sp_whip FROM enhanced_games WHERE date = :date'), 
                        conn, params={'date': '2025-08-16'})
        print('Basic WHIP columns from database:')
        print(f'home_sp_whip: min={df["home_sp_whip"].min():.2f}, max={df["home_sp_whip"].max():.2f}, std={df["home_sp_whip"].std():.3f}')
        print(f'away_sp_whip: min={df["away_sp_whip"].min():.2f}, max={df["away_sp_whip"].max():.2f}, std={df["away_sp_whip"].std():.3f}')

if __name__ == "__main__":
    main()
