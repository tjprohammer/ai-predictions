import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser',
    password='mlbpass',
    port=5432
)

cur = conn.cursor()

# Check what columns we have in enhanced_games
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'enhanced_games' 
    ORDER BY ordinal_position;
""")

print('Enhanced_games table columns:')
for row in cur.fetchall():
    print(f'  {row[0]}: {row[1]}')

# Check a sample record to see what data we have
cur.execute("""
    SELECT home_team, away_team, home_sp_name, away_sp_name,
           home_sp_season_era, away_sp_season_era, 
           home_sp_k, home_sp_bb, home_sp_ip, home_sp_h,
           away_sp_k, away_sp_bb, away_sp_ip, away_sp_h,
           wind_speed, wind_direction, weather_condition, venue_name,
           predicted_total, market_total, recommendation, confidence, edge
    FROM enhanced_games 
    WHERE date = CURRENT_DATE 
    LIMIT 1;
""")

print('\nSample game data:')
row = cur.fetchone()
if row:
    print(f'Teams: {row[1]} @ {row[0]}')
    print(f'Pitchers: {row[3]} (ERA: {row[5]}) @ {row[2]} (ERA: {row[4]})')
    print(f'Home Pitcher Stats: K={row[6]}, BB={row[7]}, IP={row[8]}, H={row[9]}')
    print(f'Away Pitcher Stats: K={row[10]}, BB={row[11]}, IP={row[12]}, H={row[13]}')
    print(f'Weather: Wind {row[14]} mph {row[15]}, Condition: {row[16]}, Venue: {row[17]}')
    print(f'Prediction: {row[18]} (Market: {row[19]}, Rec: {row[20]}, Conf: {row[21]}%, Edge: {row[22]})')
else:
    print('No games found for today')

conn.close()
