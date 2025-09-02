#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('mlb.db')
cursor = conn.execute('''
    SELECT home_team, away_team, predicted_total, predicted_total_learning 
    FROM enhanced_games 
    WHERE date='2025-08-31' 
    LIMIT 5
''')
rows = cursor.fetchall()

print('Sample predictions:')
for row in rows:
    home, away, learning, ultra80 = row
    print(f'{home} vs {away}: Learning={learning}, Ultra80={ultra80}')

# Check if any are different
cursor2 = conn.execute('''
    SELECT COUNT(*) as total,
           SUM(CASE WHEN predicted_total != predicted_total_learning THEN 1 ELSE 0 END) as different
    FROM enhanced_games 
    WHERE date='2025-08-31' 
    AND predicted_total IS NOT NULL 
    AND predicted_total_learning IS NOT NULL
''')
total, different = cursor2.fetchone()
print(f'\nTotal games: {total}, Different predictions: {different}')

conn.close()
