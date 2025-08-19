#!/usr/bin/env python3
"""Check enhanced_games table schema"""
import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT column_name, data_type, is_nullable 
        FROM information_schema.columns 
        WHERE table_name = 'enhanced_games' 
        ORDER BY ordinal_position
    """))
    
    print("Enhanced games table schema:")
    print("-" * 40)
    for row in result:
        print(f"  {row[0]:<25} {row[1]:<15} {row[2]}")
