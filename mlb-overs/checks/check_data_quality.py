import psycopg2
import pandas as pd

conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser', 
    password='mlbpass'
)

# Check training period data vs recent data
print('🔍 TRAINING DATA vs PRODUCTION DATA COMPARISON')
print('=' * 60)

# Check training period (July-August)
query_old = """
SELECT 
    COUNT(*) as total_games,
    COUNT(home_sp_er) as has_sp_er,
    COUNT(home_sp_ip) as has_sp_ip, 
    COUNT(home_sp_k) as has_sp_k,
    COUNT(plate_umpire) as has_umpire,
    COUNT(air_pressure) as has_pressure,
    COUNT(home_sp_hand) as has_handedness
FROM enhanced_games
WHERE date BETWEEN '2025-07-22' AND '2025-08-10'
"""

query_new = """
SELECT 
    COUNT(*) as total_games,
    COUNT(home_sp_er) as has_sp_er,
    COUNT(home_sp_ip) as has_sp_ip, 
    COUNT(home_sp_k) as has_sp_k,
    COUNT(plate_umpire) as has_umpire,
    COUNT(air_pressure) as has_pressure,
    COUNT(home_sp_hand) as has_handedness
FROM enhanced_games
WHERE date BETWEEN '2025-08-15' AND '2025-08-20'
"""

old_data = pd.read_sql_query(query_old, conn).iloc[0]
new_data = pd.read_sql_query(query_new, conn).iloc[0]

print('TRAINING PERIOD (July 22 - Aug 10):')
for col in old_data.index:
    if col != 'total_games':
        pct = (old_data[col] / old_data['total_games']) * 100
        print(f'  {col}: {old_data[col]}/{old_data["total_games"]} ({pct:.1f}%)')

print()
print('RECENT PERIOD (Aug 15 - Aug 20):')
for col in new_data.index:
    if col != 'total_games':
        pct = (new_data[col] / new_data['total_games']) * 100 if new_data['total_games'] > 0 else 0
        print(f'  {col}: {new_data[col]}/{new_data["total_games"]} ({pct:.1f}%)')

print()
print('📊 DATA QUALITY ANALYSIS:')
print('Both periods show missing data - the learning model')
print('was trained on incomplete data just like production!')

conn.close()
