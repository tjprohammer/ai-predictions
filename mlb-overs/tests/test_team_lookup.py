import psycopg2

conn = psycopg2.connect(
    host='localhost',
    database='mlb',
    user='mlbuser', 
    password='mlbpass'
)

cursor = conn.cursor()

# Check Angels team name formats
cursor.execute("SELECT DISTINCT home_team FROM enhanced_games WHERE date = '2025-08-20' AND home_team LIKE '%Angels%'")
angels_eg = cursor.fetchall()
print('Angels in enhanced_games:')
for team in angels_eg:
    print(f'  "{team[0]}"')

cursor.execute("SELECT DISTINCT team FROM teams_offense_daily WHERE team LIKE '%Angels%'")
angels_tod = cursor.fetchall()
print('\nAngels in teams_offense_daily:')
for team in angels_tod:
    print(f'  "{team[0]}"')

# Test the function logic directly
def test_team_lookup(team_name):
    cursor.execute("""
        SELECT runs_pg, ba, woba, wrcplus, iso, bb_pct, k_pct
        FROM teams_offense_daily 
        WHERE team = %s 
          AND runs_pg IS NOT NULL
        ORDER BY date DESC 
        LIMIT 1
    """, (team_name,))
    result = cursor.fetchone()
    if result:
        runs_pg, ba, woba, wrcplus, iso, bb_pct, k_pct = result
        print(f'\n✅ Found data for "{team_name}":')
        print(f'   R/G: {runs_pg}, BA: {ba}, wOBA: {woba}')
        return True
    else:
        print(f'\n❌ No data for "{team_name}"')
        return False

# Test different possible team names
test_names = [
    'Los Angeles Angels',
    'Angels', 
    'LAA',
    'Los Angeles Angels of Anaheim'
]

for name in test_names:
    test_team_lookup(name)

conn.close()
