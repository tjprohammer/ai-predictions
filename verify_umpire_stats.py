#!/usr/bin/env python3
"""Verify umpire statistics population"""

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

# Check umpire stats population
cur.execute("""
    SELECT COUNT(*) as total_games,
           COUNT(CASE WHEN plate_umpire_bb_pct IS NOT NULL THEN 1 END) as with_bb_pct,
           COUNT(CASE WHEN plate_umpire_strike_zone_consistency IS NOT NULL THEN 1 END) as with_zone_consistency,
           COUNT(CASE WHEN umpire_crew_consistency_rating IS NOT NULL THEN 1 END) as with_crew_rating,
           AVG(plate_umpire_bb_pct) as avg_bb_pct,
           AVG(plate_umpire_strike_zone_consistency) as avg_zone_consistency,
           AVG(umpire_crew_consistency_rating) as avg_crew_rating
    FROM enhanced_games 
    WHERE date >= '2025-03-20'
""")
result = cur.fetchone()
print(f'Games: {result[0]}, BB%: {result[1]}, Zone: {result[2]}, Crew: {result[3]}')
print(f'Avg BB%: {result[4]:.1f}, Avg Zone: {result[5]:.1f}, Avg Crew: {result[6]:.1f}')

# Sample a few games to verify data quality
cur.execute("""
    SELECT game_id, plate_umpire, plate_umpire_bb_pct, 
           plate_umpire_strike_zone_consistency, umpire_crew_consistency_rating
    FROM enhanced_games 
    WHERE date >= '2025-03-20' 
    AND plate_umpire IS NOT NULL
    LIMIT 5
""")
samples = cur.fetchall()
print('\nSample game data:')
for sample in samples:
    print(f'  {sample[0]}: {sample[1]} | BB%: {sample[2]} | Zone: {sample[3]} | Crew: {sample[4]}')

conn.close()
