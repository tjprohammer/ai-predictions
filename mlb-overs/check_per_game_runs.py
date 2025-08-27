#!/usr/bin/env python3
"""
Check per-game runs columns
"""
from sqlalchemy import create_engine, text

def check_per_game_runs():
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    with engine.connect() as conn:
        # Check if _pg columns are per-game averages or actual game runs
        result = conn.execute(text("""
            SELECT date, total_runs, 
                   home_team_runs_pg, away_team_runs_pg,
                   home_team_runs_l7, away_team_runs_l7
            FROM enhanced_games 
            WHERE date = '2025-04-30' 
              AND total_runs IS NOT NULL 
            LIMIT 5
        """))
        
        print("Per-game vs L7 runs columns:")
        for row in result:
            total = row.total_runs
            home_pg = f"{row.home_team_runs_pg:.2f}" if row.home_team_runs_pg else "N/A"
            away_pg = f"{row.away_team_runs_pg:.2f}" if row.away_team_runs_pg else "N/A"
            home_l7 = f"{row.home_team_runs_l7:.1f}" if row.home_team_runs_l7 else "N/A"
            away_l7 = f"{row.away_team_runs_l7:.1f}" if row.away_team_runs_l7 else "N/A"
            print(f"  total={total}, home_pg={home_pg}, away_pg={away_pg}, home_l7={home_l7}, away_l7={away_l7}")
            
        # The _pg columns look like averages. We need individual game runs.
        # Let's see if we can back-calculate from score or other columns
        result = conn.execute(text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'enhanced_games' 
              AND (column_name LIKE '%score%' OR column_name LIKE '%final%')
            ORDER BY column_name
        """))
        
        print("\nScore/final columns:")
        for row in result:
            print(f"  {row.column_name}")

if __name__ == "__main__":
    check_per_game_runs()
