#!/usr/bin/env python3
"""Check final scores status"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total_games,
            COUNT(total_runs) as with_finals,
            COUNT(market_total) as with_market,
            COUNT(predicted_total) as with_predictions
        FROM enhanced_games 
        WHERE date = '2025-08-14'
    """))
    
    row = result.fetchone()
    print(f"Games on 2025-08-14:")
    print(f"  Total games: {row[0]}")
    print(f"  With final scores: {row[1]}")
    print(f"  With market totals: {row[2]}")
    print(f"  With predictions: {row[3]}")
    
    # Check a sample game
    sample = conn.execute(text("""
        SELECT game_id, home_team, away_team, home_score, away_score, total_runs, market_total, predicted_total
        FROM enhanced_games 
        WHERE date = '2025-08-14'
        LIMIT 1
    """)).fetchone()
    
    if sample:
        print(f"\nSample game:")
        print(f"  {sample[2]} @ {sample[1]} (ID: {sample[0]})")
        print(f"  Score: {sample[4]} - {sample[3]} (Total: {sample[5]})")
        print(f"  Market: {sample[6]}, Predicted: {sample[7]}")
