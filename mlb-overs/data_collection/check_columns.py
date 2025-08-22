import psycopg2

conn = psycopg2.connect(host='localhost', database='mlb', user='mlbuser', password='mlbpass')
cur = conn.cursor()

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'enhanced_games' ORDER BY column_name")
columns = [row[0] for row in cur.fetchall()]

print("Enhanced_games columns:")
for col in columns:
    print(f"  {col}")

cur.close()
conn.close()
