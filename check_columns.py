import psycopg2

conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser', 
    password='mlbpass'
)

cursor = conn.cursor()
cursor.execute("""
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'enhanced_games'
ORDER BY ordinal_position
""")

print('ENHANCED_GAMES TABLE COLUMNS:')
print('=' * 50)
for col_name, data_type in cursor.fetchall():
    print(f'{col_name:<25} {data_type}')

conn.close()
