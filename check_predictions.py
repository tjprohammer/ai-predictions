import psycopg2

# Connect to database
conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser',
    password='mlbpass'
)
cursor = conn.cursor()

# Get today's predictions
cursor.execute('''
    SELECT home_team, away_team, predicted_total, market_total, 
           predicted_total - market_total as pred_vs_market
    FROM enhanced_games 
    WHERE date = '2025-08-20' 
    ORDER BY game_id 
    LIMIT 5
''')

rows = cursor.fetchall()

print("Home Team           | Away Team           | Predicted | Market | Diff")
print("-" * 75)
for row in rows:
    home, away, pred, market, diff = row
    pred_str = f"{pred:.1f}" if pred else "NULL"
    market_str = f"{market:.1f}" if market else "NULL" 
    diff_str = f"{diff:.1f}" if diff else "NULL"
    print(f"{home:<18} | {away:<18} | {pred_str:<9} | {market_str:<6} | {diff_str}")

conn.close()
