#!/usr/bin/env python3
"""
Find the correct runs columns in enhanced_games
"""
from sqlalchemy import create_engine, text

def check_columns():
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    with engine.connect() as conn:
        # Find runs-related columns
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games' 
              AND column_name LIKE '%run%'
            ORDER BY column_name
        """))
        
        print("Columns with 'run' in enhanced_games:")
        for row in result:
            print(f"  {row.column_name}")
            
        # Check what total_runs looks like (this should be per-game)
        print("\nSample total_runs values:")
        result = conn.execute(text("""
            SELECT date, total_runs, home_team_runs, away_team_runs
            FROM enhanced_games 
            WHERE date = '2025-04-30' 
              AND total_runs IS NOT NULL
            LIMIT 5
        """))
        
        for row in result:
            print(f"  {row.date}: total={row.total_runs}, home_team={row.home_team_runs}, away_team={row.away_team_runs}")

if __name__ == "__main__":
    check_columns()
