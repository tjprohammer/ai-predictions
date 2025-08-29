#!/usr/bin/env python3

from sqlalchemy import create_engine, text
import pandas as pd

def main():
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    with engine.connect() as conn:
        # Check today's games with all prediction columns
        df = pd.read_sql("""
            SELECT home_team, away_team, market_total,
                   predicted_total, confidence,
                   predicted_total_original, predicted_total_learning,
                   predicted_total_ultra,
                   learning_confidence
            FROM enhanced_games 
            WHERE date = '2025-08-27'
            ORDER BY home_team
        """, conn)
        
        print(f'Found {len(df)} games for 2025-08-27')
        if len(df) > 0:
            print('\nPredictions for today:')
            print(df.to_string())
        else:
            print('No games found for today')
            
            # Check what dates we do have
            df_dates = pd.read_sql("SELECT DISTINCT date FROM enhanced_games ORDER BY date DESC LIMIT 10", conn)
            print(f"\nRecent dates in database: {df_dates['date'].tolist()}")

if __name__ == '__main__':
    main()
