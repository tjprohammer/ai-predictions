#!/usr/bin/env python3
"""
Check All Games Today
====================
Check all 13 games for today and their prediction status
"""

import os
from sqlalchemy import create_engine, text

def check_all_games():
    engine = create_engine(os.getenv('DATABASE_URL', 'postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb'))
    
    # Check total games for today
    query = text("""
        SELECT 
            COUNT(*) as total_games,
            COUNT(predicted_total_original) as orig_preds,
            COUNT(predicted_total_learning) as learn_preds,
            COUNT(CASE WHEN predicted_total_original IS NOT NULL AND predicted_total_learning IS NOT NULL THEN 1 END) as both_preds
        FROM enhanced_games 
        WHERE date = '2025-08-22'
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()
        print(f"📊 GAME COUNT SUMMARY FOR 2025-08-22:")
        print(f"   Total games: {result[0]}")
        print(f"   Original predictions: {result[1]}")
        print(f"   Learning predictions: {result[2]}")
        print(f"   Both models: {result[3]}")
        print()
        
        # Show all games
        games_query = text("""
            SELECT 
                game_id,
                home_team,
                away_team,
                market_total,
                predicted_total_original,
                predicted_total_learning,
                predicted_total,
                total_runs
            FROM enhanced_games 
            WHERE date = '2025-08-22'
            ORDER BY game_id
        """)
        
        games = conn.execute(games_query).fetchall()
        
        print(f"🏟️ ALL {len(games)} GAMES FOR TODAY:")
        print("="*80)
        
        for i, game in enumerate(games, 1):
            game_id, home, away, market, orig, learn, current, actual = game
            
            print(f"{i:2d}. {home} vs {away}")
            print(f"    Game ID: {game_id}")
            print(f"    Market: {market if market else 'N/A'}")
            print(f"    Original: {orig:.2f}" if orig else "    Original: N/A")
            print(f"    Learning: {learn:.2f}" if learn else "    Learning: N/A")
            print(f"    Current: {current if current else 'N/A'}")
            print(f"    Actual: {actual if actual else 'Pending'}")
            
            # Status
            if orig and learn:
                diff = learn - orig
                print(f"    Status: ✅ Both models ({diff:+.2f} diff)")
            elif orig:
                print(f"    Status: 🔵 Original only")
            elif learn:
                print(f"    Status: 🟢 Learning only")
            else:
                print(f"    Status: ❌ No predictions")
            print()

if __name__ == "__main__":
    check_all_games()
