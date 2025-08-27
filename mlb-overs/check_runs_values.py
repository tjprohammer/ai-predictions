#!/usr/bin/env python3
"""
Check team runs values in enhanced_games
"""
from sqlalchemy import create_engine, text

def check_runs():
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    with engine.connect() as conn:
        # Check actual team runs values
        result = conn.execute(text("""
            SELECT home_team_runs, away_team_runs, date 
            FROM enhanced_games 
            WHERE date = '2025-04-30' 
            LIMIT 5
        """))
        
        print("Sample team runs from enhanced_games:")
        for row in result:
            print(f"  {row.date}: home={row.home_team_runs}, away={row.away_team_runs}")
            
        # Check if these are cumulative or per-game
        result = conn.execute(text("""
            SELECT 
                AVG(home_team_runs::numeric) as avg_home_runs,
                AVG(away_team_runs::numeric) as avg_away_runs,
                MIN(home_team_runs::numeric) as min_home_runs,
                MAX(home_team_runs::numeric) as max_home_runs
            FROM enhanced_games 
            WHERE date BETWEEN '2025-04-01' AND '2025-04-30'
              AND home_team_runs IS NOT NULL
        """))
        
        row = result.fetchone()
        print(f"\nApril 2025 team runs stats:")
        print(f"  Avg: home={row.avg_home_runs:.1f}, away={row.avg_away_runs:.1f}")
        print(f"  Range: {row.min_home_runs:.0f} to {row.max_home_runs:.0f}")
        
        if row.avg_home_runs > 50:
            print("  ⚠️  These look like CUMULATIVE season totals, not per-game!")
            print("  The materialized view is summing season totals, not individual game runs.")

if __name__ == "__main__":
    check_runs()
