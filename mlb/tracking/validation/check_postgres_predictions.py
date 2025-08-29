#!/usr/bin/env python3

from sqlalchemy import create_engine, text
import pandas as pd

def main():
    # Connect to PostgreSQL  
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    # Check available tables
    with engine.connect() as conn:
        result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result]
        print('Available tables:', tables)
        
        # Check for games table with today's data
        if 'games' in tables:
            df = pd.read_sql("SELECT * FROM games WHERE date = '2025-08-27' LIMIT 3", conn)
            print(f"\nColumns in games table: {list(df.columns)}")
            print(f"Sample data:\n{df}")
            
        # Check for enhanced_games table  
        if 'enhanced_games' in tables:
            df = pd.read_sql("SELECT * FROM enhanced_games WHERE date = '2025-08-27' OR game_date = '2025-08-27' LIMIT 3", conn)
            print(f"\nColumns in enhanced_games table: {list(df.columns)}")
            print(f"Sample data:\n{df}")

if __name__ == '__main__':
    main()
