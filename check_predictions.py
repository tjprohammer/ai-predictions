#!/usr/bin/env python3

import sqlite3
import pandas as pd

def main():
    conn = sqlite3.connect('mlb.db')
    
    # Check what prediction columns exist in enhanced_games
    cursor = conn.cursor()
    cursor.execute('PRAGMA table_info(enhanced_games)')
    columns = cursor.fetchall()
    pred_columns = [col[1] for col in columns if 'prediction' in col[1].lower()]
    print('Prediction columns:', pred_columns)
    
    # Check today's games with all prediction fields
    query = """
    SELECT game_id, home_team, away_team, 
           incremental_prediction, incremental_confidence,
           enhanced_prediction, enhanced_confidence,
           ultra_prediction, ultra_confidence,
           dual_prediction_original, dual_prediction_learning,
           market_total
    FROM enhanced_games 
    WHERE game_date = '2025-08-27'
    ORDER BY game_id
    """
    
    try:
        df = pd.read_sql_query(query, conn)
        print(f'\nFound {len(df)} games for 2025-08-27:')
        if len(df) > 0:
            print(df.to_string())
        else:
            print("No games found for 2025-08-27")
    except Exception as e:
        print(f"Error: {e}")
        # Try simpler query
        query2 = "SELECT * FROM enhanced_games WHERE game_date = '2025-08-27' LIMIT 3"
        df2 = pd.read_sql_query(query2, conn)
        print(f"\nColumns in enhanced_games: {list(df2.columns)}")
        print(f"Sample data:\n{df2.head()}")
    
    conn.close()

if __name__ == '__main__':
    main()
