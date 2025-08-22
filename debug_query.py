#!/usr/bin/env python3
"""Debug script to check database query structure"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

# Connect to database  
db_url = os.getenv('DATABASE_URL')
if db_url and db_url.startswith('postgresql+psycopg2://'):
    db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')
conn = psycopg2.connect(db_url)
cur = conn.cursor()

# Check one game to see the exact structure
cur.execute("""
    SELECT game_id, plate_umpire, first_base_umpire_name, 
           second_base_umpire_name, third_base_umpire_name
    FROM enhanced_games 
    WHERE date >= '2025-03-20' 
    AND plate_umpire IS NOT NULL 
    LIMIT 1
""")
result = cur.fetchone()
print(f'Sample result: {result}')
print(f'Length: {len(result) if result else 0}')

# Check columns that exist
cur.execute("""
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    AND column_name LIKE '%umpire%' 
    ORDER BY column_name
""")
columns = cur.fetchall()
print(f'Umpire columns: {[col[0] for col in columns]}')

conn.close()
