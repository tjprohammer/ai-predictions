#!/usr/bin/env python3
"""
Debug script for historical predictions API
"""

import sys
sys.path.append('.')

from sqlalchemy import create_engine
import pandas as pd
from historical_prediction_system import find_similar_games, calculate_prediction_from_history

def debug_historical_predictions():
    """Debug why the API isn't returning predictions"""
    
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')
    
    # Get data
    with engine.begin() as conn:
        current_games_df = pd.read_sql("SELECT * FROM daily_games ORDER BY id", conn)
        historical_games_df = pd.read_sql("SELECT * FROM enhanced_games ORDER BY date DESC", conn)
    
    print(f"Found {len(current_games_df)} current games")
    print(f"Found {len(historical_games_df)} historical games")
    
    # Test first game
    if not current_games_df.empty:
        test_game = current_games_df.iloc[0].to_dict()
        print(f"\nTesting game: {test_game['away_team']} @ {test_game['home_team']}")
        print(f"Game dict keys: {list(test_game.keys())}")
        
        historical_games_df = historical_games_df  # Keep as DataFrame
        
        try:
            # Find similar games - pass DataFrame
            similar_games = find_similar_games(test_game, historical_games_df)
            print(f"Found {len(similar_games)} similar games")
            
            if len(similar_games) >= 3:
                # Calculate prediction
                prediction_data = calculate_prediction_from_history(similar_games, test_game)
                print(f"Prediction: {prediction_data}")
                return True
            else:
                print("Not enough similar games found")
                return False
                
        except Exception as e:
            print(f"Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return False

if __name__ == "__main__":
    debug_historical_predictions()
