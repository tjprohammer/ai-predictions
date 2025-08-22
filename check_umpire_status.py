#!/usr/bin/env python3
"""Check current umpire data implementation"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def check_umpire_data():
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    cursor = conn.cursor()

    # Check enhanced_games table for umpire columns
    cursor.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    AND column_name LIKE '%umpire%'
    ORDER BY column_name;
    """)

    print('Umpire columns in enhanced_games:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]}')

    # Check if umpires table exists
    cursor.execute("""
    SELECT EXISTS (
       SELECT FROM information_schema.tables 
       WHERE table_name = 'umpires'
    );
    """)

    umpires_table_exists = cursor.fetchone()[0]
    print(f'\nUmpires table exists: {umpires_table_exists}')

    if umpires_table_exists:
        cursor.execute('SELECT COUNT(*) FROM umpires;')
        count = cursor.fetchone()[0]
        print(f'Umpires in table: {count}')
        
        # Sample umpire data
        cursor.execute('SELECT name, o_u_tendency, sample_size FROM umpires LIMIT 5;')
        print('\nSample umpires:')
        for row in cursor.fetchall():
            print(f'  {row[0]}: O/U={row[1]}, games={row[2]}')

    # Check recent games with umpire data
    cursor.execute("""
    SELECT game_date, home_team, away_team, plate_umpire
    FROM enhanced_games 
    WHERE game_date >= '2025-08-20' 
    AND plate_umpire IS NOT NULL
    ORDER BY game_date DESC
    LIMIT 5;
    """)

    print('\nRecent games with umpire data:')
    for row in cursor.fetchall():
        print(f'  {row[0]}: {row[1]} vs {row[2]} - {row[3]}')

    # Check all umpire columns in enhanced_games
    cursor.execute("""
    SELECT COUNT(*) as total,
           COUNT(plate_umpire) as with_plate,
           COUNT(first_base_umpire) as with_1b,
           COUNT(second_base_umpire) as with_2b,
           COUNT(third_base_umpire) as with_3b
    FROM enhanced_games 
    WHERE game_date >= '2025-08-01';
    """)
    
    stats = cursor.fetchone()
    print(f'\nAugust 2025 games umpire data:')
    print(f'  Total games: {stats[0]}')
    print(f'  With plate umpire: {stats[1]} ({stats[1]/stats[0]*100:.1f}%)')
    print(f'  With 1B umpire: {stats[2]} ({stats[2]/stats[0]*100:.1f}%)')
    print(f'  With 2B umpire: {stats[3]} ({stats[3]/stats[0]*100:.1f}%)')
    print(f'  With 3B umpire: {stats[4]} ({stats[4]/stats[0]*100:.1f}%)')

    conn.close()

if __name__ == "__main__":
    check_umpire_data()
