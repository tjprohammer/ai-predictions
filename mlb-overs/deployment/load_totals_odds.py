#!/usr/bin/env python3
"""
Minimal Totals Odds Loader

Loads odds from CSV into totals_odds table to unblock the "price the exact line" logic.
Usage: python load_totals_odds.py odds_2025-08-17.csv

CSV format: game_id,date,book,total,over_odds,under_odds
"""

import sys
import pandas as pd
from sqlalchemy import create_engine, text
import os
import datetime as dt

def main():
    if len(sys.argv) != 2:
        print("Usage: python load_totals_odds.py <odds_file.csv>")
        print("CSV columns: game_id,date,book,total,over_odds,under_odds")
        return 1
    
    csv_file = sys.argv[1]
    
    try:
        # Load CSV
        df = pd.read_csv(csv_file, dtype={"game_id": str, "book": str})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df["collected_at"] = dt.datetime.utcnow()
        
        print(f"📊 Loading {len(df)} odds records from {csv_file}")
        
        # Connect to database
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        
        with engine.begin() as conn:
            # Ensure table exists
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS totals_odds(
                  game_id varchar NOT NULL,
                  "date" date NOT NULL,
                  book text NOT NULL,
                  total numeric NOT NULL,
                  over_odds integer,
                  under_odds integer,
                  collected_at timestamp NOT NULL DEFAULT now(),
                  PRIMARY KEY (game_id, "date", book, total, collected_at)
                )
            """))
            
            # Insert data using executemany for PostgreSQL
            insert_sql = text("""
                INSERT INTO totals_odds (game_id, "date", book, total, over_odds, under_odds, collected_at)
                VALUES (:game_id, :date, :book, :total, :over_odds, :under_odds, :collected_at)
            """)
            
            conn.execute(insert_sql, df.to_dict('records'))
            
        print(f"✅ Successfully loaded {len(df)} odds records")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"📚 Books: {', '.join(df['book'].unique())}")
        
    except Exception as e:
        print(f"❌ Error loading odds: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
