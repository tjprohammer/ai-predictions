import psycopg2
import pandas as pd

conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')

query = '''
SELECT date, COUNT(*) as games,
       SUM(CASE WHEN total_runs IS NULL THEN 1 ELSE 0 END) as null_runs,
       AVG(total_runs) as avg_runs
FROM enhanced_games 
WHERE date >= '2025-08-20'
GROUP BY date 
ORDER BY date DESC;
'''

df = pd.read_sql(query, conn)
print('RECENT TOTAL_RUNS STATUS:')
for _, row in df.iterrows():
    null_pct = (row['null_runs'] / row['games']) * 100
    avg_runs = row['avg_runs'] if pd.notna(row['avg_runs']) else 0
    print(f"{row['date']} | {row['games']} games | {null_pct:.0f}% null runs | avg: {avg_runs:.1f}")
conn.close()
