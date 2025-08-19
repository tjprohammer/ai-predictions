#!/usr/bin/env python3
from sqlalchemy import create_engine, text
import os

engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb')

print("Checking enhanced_games table for today's data...")
with engine.begin() as conn:
    # Check what columns exist
    print("\n1. Table structure:")
    result = conn.execute(text("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        AND column_name IN ('game_id', 'date', 'game_date', 'market_total', 'over_odds', 'under_odds')
        ORDER BY column_name
    """))
    for row in result:
        print(f"   {row.column_name}: {row.data_type}")
    
    # Check today's data
    print("\n2. Today's market data:")
    result = conn.execute(text("""
        SELECT game_id, date, market_total, over_odds, under_odds 
        FROM enhanced_games 
        WHERE date = CURRENT_DATE 
        LIMIT 5
    """))
    for row in result:
        print(f"   Game {row.game_id}: Total={row.market_total}, Over={row.over_odds}, Under={row.under_odds}")
    
    print("\n3. Sample data with any date:")
    result = conn.execute(text("""
        SELECT game_id, date, market_total, over_odds, under_odds 
        FROM enhanced_games 
        WHERE market_total IS NOT NULL 
        ORDER BY date DESC 
        LIMIT 3
    """))
    for row in result:
        print(f"   Game {row.game_id} ({row.date}): Total={row.market_total}, Over={row.over_odds}, Under={row.under_odds}")
