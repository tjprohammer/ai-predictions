#!/usr/bin/env python3
"""Check what umpire data exists in the database"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_umpire_data():
    db_url = os.getenv('DATABASE_URL').replace('postgresql+psycopg2://', 'postgresql://')
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    print('📊 Checking for any umpire data in database:')
    cur.execute("""
        SELECT 
            COUNT(*) as total_games,
            COUNT(plate_umpire) as plate_umpire_count,
            COUNT(first_base_umpire_name) as first_base_count,
            COUNT(second_base_umpire_name) as second_base_count,
            COUNT(third_base_umpire_name) as third_base_count
        FROM enhanced_games 
        WHERE date >= '2025-03-20'
    """)
    
    row = cur.fetchone()
    print(f'Total games: {row[0]}')
    print(f'Games with plate umpire: {row[1]}')  
    print(f'Games with 1B umpire: {row[2]}')
    print(f'Games with 2B umpire: {row[3]}')
    print(f'Games with 3B umpire: {row[4]}')
    
    print('\n🔍 Sample umpire data:')
    cur.execute("""
        SELECT plate_umpire, first_base_umpire_name, date, home_team, away_team
        FROM enhanced_games 
        WHERE date >= '2025-03-20'
        AND (plate_umpire IS NOT NULL OR first_base_umpire_name IS NOT NULL)
        LIMIT 5
    """)
    
    rows = cur.fetchall()
    if len(rows) == 0:
        print("No umpire assignments found in database")
    else:
        for row in rows:
            print(f'Plate: {row[0]}, 1B: {row[1]}, {row[2]} {row[3]} vs {row[4]}')
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    check_umpire_data()
