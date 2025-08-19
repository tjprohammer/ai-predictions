#!/usr/bin/env python3
"""Quick check of backfill progress"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT 
            date,
            COUNT(*) as total_games,
            COUNT(total_runs) as with_finals,
            COUNT(market_total) as with_market,
            COUNT(predicted_total) as with_predictions
        FROM enhanced_games 
        WHERE date BETWEEN '2025-07-15' AND '2025-08-14'
        GROUP BY date
        ORDER BY date
    """))
    
    rows = result.fetchall()
    
    print("Backfill Progress:")
    print("Date       | Games | Finals | Market | Predictions")
    print("-" * 50)
    
    total_games = 0
    total_finals = 0
    total_market = 0
    total_preds = 0
    
    for row in rows:
        date, games, finals, market, preds = row
        print(f"{date} |   {games:2d}  |   {finals:2d}   |   {market:2d}   |     {preds:2d}")
        total_games += games
        total_finals += finals
        total_market += market
        total_preds += preds
    
    print("-" * 50)
    print(f"Totals     |  {total_games:3d}  |   {total_finals:2d}   |   {total_market:2d}   |     {total_preds:2d}")
    print()
    print(f"Coverage: {total_finals}/{total_games} games with final scores ({total_finals/total_games*100:.1f}%)")
    print(f"Market data: {total_market}/{total_games} games ({total_market/total_games*100:.1f}%)")
    print(f"Predictions: {total_preds}/{total_games} games ({total_preds/total_games*100:.1f}%)")
