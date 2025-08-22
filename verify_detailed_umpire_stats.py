#!/usr/bin/env python3
"""Verify detailed umpire statistics with RPG values"""

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

# Check detailed umpire stats with RPG values
cur.execute("""
    SELECT plate_umpire,
           plate_umpire_bb_pct,
           plate_umpire_strike_zone_consistency,
           plate_umpire_boost_factor,
           plate_umpire_rpg,
           plate_umpire_ba_against,
           plate_umpire_obp_against,
           plate_umpire_slg_against,
           COUNT(*) as games
    FROM enhanced_games 
    WHERE date >= '2025-03-20' 
    AND plate_umpire IS NOT NULL
    GROUP BY plate_umpire, plate_umpire_bb_pct, plate_umpire_strike_zone_consistency,
             plate_umpire_boost_factor, plate_umpire_rpg, plate_umpire_ba_against,
             plate_umpire_obp_against, plate_umpire_slg_against
    ORDER BY plate_umpire
    LIMIT 10
""")

detailed_stats = cur.fetchall()
print('DETAILED UMPIRE STATISTICS (Sample):')
print('=' * 90)
print(f'{"Umpire":<20} | {"BB%":<4} | {"Zone":<4} | {"Boost":<5} | {"RPG":<5} | {"BA":<5} | {"OBP":<5} | {"SLG":<5} | {"Games":<5}')
print('-' * 90)

for stats in detailed_stats:
    ump, bb, zone, boost, rpg, ba, obp, slg, games = stats
    print(f'{ump:<20} | {bb:<4.1f} | {zone:<4.1f} | {boost:<5.3f} | {rpg:<5.1f} | {ba:<5.3f} | {obp:<5.3f} | {slg:<5.3f} | {games:<5}')

# Check the range of realistic values
cur.execute("""
    SELECT 
        MIN(plate_umpire_bb_pct) as min_bb,
        MAX(plate_umpire_bb_pct) as max_bb,
        MIN(plate_umpire_rpg) as min_rpg,
        MAX(plate_umpire_rpg) as max_rpg,
        MIN(plate_umpire_ba_against) as min_ba,
        MAX(plate_umpire_ba_against) as max_ba,
        MIN(plate_umpire_boost_factor) as min_boost,
        MAX(plate_umpire_boost_factor) as max_boost
    FROM enhanced_games 
    WHERE date >= '2025-03-20'
""")

ranges = cur.fetchone()
print(f'\nREALISTIC STAT RANGES:')
print(f'BB%: {ranges[0]:.1f}% - {ranges[1]:.1f}% (Real MLB: 7.3% - 10.3%)')
print(f'RPG: {ranges[2]:.1f} - {ranges[3]:.1f} (Real MLB: 7.14 - 10.82)')
print(f'BA: {ranges[4]:.3f} - {ranges[5]:.3f} (Real MLB: 0.234 - 0.279)')
print(f'Boost: {ranges[6]:.3f} - {ranges[7]:.3f} (Relative to 8.75 RPG baseline)')

conn.close()
