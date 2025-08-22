import psycopg2
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.getenv('DATABASE_URL')
if 'postgresql+psycopg2://' in db_url:
    db_url = db_url.replace('postgresql+psycopg2://', 'postgresql://')

# Load umpire stats from CSV (update path as needed)
# Expected CSV columns: game_id, home_plate_umpire_name, first_base_umpire_name, etc.
df = pd.read_csv('umpire_stats.csv')

conn = psycopg2.connect(db_url)
cur = conn.cursor()

for _, row in df.iterrows():
    # Update umpire names and position-specific stats
    cur.execute('''
        UPDATE enhanced_games SET
            home_plate_umpire_name = %s,
            first_base_umpire_name = %s,
            second_base_umpire_name = %s,
            third_base_umpire_name = %s,
            plate_umpire_k_pct = %s,
            plate_umpire_bb_pct = %s,
            plate_umpire_strike_zone_consistency = %s,
            plate_umpire_avg_strikes_per_ab = %s,
            plate_umpire_rpg = %s,
            plate_umpire_ba_against = %s,
            plate_umpire_obp_against = %s,
            plate_umpire_slg_against = %s,
            plate_umpire_boost_factor = %s,
            base_umpires_experience_avg = %s,
            base_umpires_error_rate = %s,
            base_umpires_close_call_accuracy = %s,
            umpire_crew_total_experience = %s,
            umpire_crew_consistency_rating = %s
        WHERE game_id = %s
    ''', (
        row['home_plate_umpire_name'], row['first_base_umpire_name'],
        row['second_base_umpire_name'], row['third_base_umpire_name'],
        row['plate_umpire_k_pct'], row['plate_umpire_bb_pct'],
        row['plate_umpire_strike_zone_consistency'], row['plate_umpire_avg_strikes_per_ab'],
        row['plate_umpire_rpg'], row['plate_umpire_ba_against'],
        row['plate_umpire_obp_against'], row['plate_umpire_slg_against'],
        row['plate_umpire_boost_factor'], row['base_umpires_experience_avg'],
        row['base_umpires_error_rate'], row['base_umpires_close_call_accuracy'],
        row['umpire_crew_total_experience'], row['umpire_crew_consistency_rating'],
        row['game_id']
    ))

conn.commit()
cur.close()
conn.close()
print('Position-specific umpire stats ingested successfully.')
