#!/usr/bin/env python3

from sqlalchemy import create_engine, text
import pandas as pd
import os

def main():
    engine = create_engine(os.environ.get('DATABASE_URL', 'postgresql://mlbuser:mlbpass@localhost:5432/mlb'))

    with engine.connect() as conn:
        # Check the schema first
        result = conn.execute(text('SELECT column_name FROM information_schema.columns WHERE table_name = \'legitimate_game_features\' AND column_name LIKE \'%date%\''))
        print('Date columns in LGF:')
        for row in result:
            print(f'  {row[0]}')
        
        # Try common date column names
        for col in ['date', 'game_date', 'target_date']:
            try:
                result = pd.read_sql(text(f'SELECT COUNT(*) as n FROM legitimate_game_features WHERE {col} = :date'), 
                                    conn, params={'date': '2025-08-17'})
                print(f'LGF count for 2025-08-17 using {col}: {result["n"].iloc[0]}')
                break
            except Exception as e:
                print(f'Failed with {col}: {e}')
                continue

if __name__ == "__main__":
    main()
