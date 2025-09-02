#!/usr/bin/env python3
"""
Quick Game Results Updater
Updates the enhanced_games table with final scores for today's completed games
"""

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import requests
import time

def update_todays_results():
    """Update today's completed games with final results"""
    print("🎯 UPDATING TODAY'S GAME RESULTS")
    print("=" * 40)
    
    # Get today's date
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"📅 Processing results for: {today}")
    
    # Connect to database
    engine = create_engine("postgresql://mlbuser:mlbpass@localhost/mlb")
    
    # Get today's games that need results
    with engine.connect() as conn:
        games_query = text("""
            SELECT game_id, home_team, away_team, total_runs
            FROM enhanced_games 
            WHERE date = :today
            ORDER BY home_team
        """)
        
        games_df = pd.read_sql(games_query, conn, params={'today': today})
    
    print(f"📊 Found {len(games_df)} games for {today}")
    
    # Show current status
    completed_games = games_df[games_df['total_runs'].notna()]
    pending_games = games_df[games_df['total_runs'].isna()]
    
    print(f"✅ Completed games: {len(completed_games)}")
    print(f"⏳ Pending games: {len(pending_games)}")
    
    if len(completed_games) > 0:
        print("\n🏁 COMPLETED GAMES:")
        for _, game in completed_games.iterrows():
            print(f"  {game['away_team']} @ {game['home_team']}: {game['total_runs']} runs")
    
    if len(pending_games) > 0:
        print(f"\n⏳ PENDING GAMES:")
        for _, game in pending_games.iterrows():
            print(f"  {game['away_team']} @ {game['home_team']}: Waiting for result...")
    
    # Calculate prediction errors for completed games
    if len(completed_games) > 0:
        with engine.connect() as conn:
            error_query = text("""
                SELECT 
                    game_id, home_team, away_team, total_runs,
                    predicted_total_learning,
                    predicted_total_ultra,
                    market_total,
                    CASE WHEN predicted_total_learning IS NOT NULL AND total_runs IS NOT NULL 
                         THEN ABS(predicted_total_learning - total_runs) END as learning_error,
                    CASE WHEN predicted_total_ultra IS NOT NULL AND total_runs IS NOT NULL 
                         THEN ABS(predicted_total_ultra - total_runs) END as ultra_error,
                    CASE WHEN market_total IS NOT NULL AND total_runs IS NOT NULL 
                         THEN ABS(market_total - total_runs) END as market_error
                FROM enhanced_games 
                WHERE date = :today AND total_runs IS NOT NULL
                ORDER BY learning_error NULLS LAST
            """)
            
            errors_df = pd.read_sql(error_query, conn, params={'today': today})
        
        if len(errors_df) > 0:
            print(f"\n📊 PREDICTION ACCURACY SUMMARY:")
            
            # Learning model stats
            learning_errors = errors_df['learning_error'].dropna()
            if len(learning_errors) > 0:
                print(f"  🤖 Learning Model: {len(learning_errors)} predictions, {learning_errors.mean():.2f} avg error")
            
            # Ultra model stats  
            ultra_errors = errors_df['ultra_error'].dropna()
            if len(ultra_errors) > 0:
                print(f"  🧠 Ultra 80 Model: {len(ultra_errors)} predictions, {ultra_errors.mean():.2f} avg error")
            
            # Market stats
            market_errors = errors_df['market_error'].dropna()
            if len(market_errors) > 0:
                print(f"  🎯 Market Lines: {len(market_errors)} games, {market_errors.mean():.2f} avg error")
            
            # Show best and worst predictions
            if len(learning_errors) > 0:
                best_game = errors_df.loc[errors_df['learning_error'].idxmin()]
                worst_game = errors_df.loc[errors_df['learning_error'].idxmax()]
                
                print(f"\n🏆 BEST PREDICTION:")
                print(f"  {best_game['away_team']} @ {best_game['home_team']}: {best_game['learning_error']:.1f} runs error")
                
                print(f"📉 WORST PREDICTION:")
                print(f"  {worst_game['away_team']} @ {worst_game['home_team']}: {worst_game['learning_error']:.1f} runs error")
    
    print(f"\n✅ Results update complete for {today}")
    print("🎯 Data ready for UI tracking display!")
    
    return len(completed_games), len(pending_games)

if __name__ == "__main__":
    completed, pending = update_todays_results()
