#!/usr/bin/env python3
"""
Debug feature alignment between training and backtest serving
"""
import pandas as pd
from sqlalchemy import create_engine, text

def main():
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    print("=== ENHANCED_GAMES COLUMNS ===")
    eg_cols = pd.read_sql("SELECT * FROM enhanced_games LIMIT 1", engine)
    print(f"Enhanced_games columns: {len(eg_cols.columns)}")
    print("Sample columns:", list(eg_cols.columns)[:20])
    
    print("\n=== GAME_CONDITIONS COLUMNS ===")
    try:
        gc_cols = pd.read_sql("SELECT * FROM game_conditions LIMIT 1", engine)
        print(f"Game_conditions columns: {len(gc_cols.columns)}")
        print("Sample columns:", list(gc_cols.columns)[:20])
        
        print("\n=== JOIN COMPARISON ===")
        # Test the actual join query used in backtest
        q = """
          SELECT
            gc.*,
            eg.market_total AS eg_market_total,
            eg.total_runs   AS eg_total_runs,
            eg.date         AS eg_date,
            eg.home_sp_days_rest AS eg_home_sp_rest_days,
            eg.away_sp_days_rest AS eg_away_sp_rest_days,
            eg.roof_status        AS eg_roof_state
          FROM game_conditions gc
          JOIN enhanced_games eg ON gc.game_id::text = eg.game_id::text
          WHERE eg.date = '2025-08-20'
          LIMIT 1
        """
        joined = pd.read_sql(text(q), engine)
        print(f"Joined result columns: {len(joined.columns)}")
        print("Sample joined columns:", list(joined.columns)[:30])
        
    except Exception as e:
        print(f"Error accessing game_conditions: {e}")
        print("game_conditions table may not exist!")
        
        print("\n=== FALLBACK: Using enhanced_games directly ===")
        eg_sample = pd.read_sql("SELECT * FROM enhanced_games WHERE date = '2025-08-20' LIMIT 1", engine)
        print(f"Enhanced_games direct: {len(eg_sample.columns)} columns")

if __name__ == "__main__":
    main()
