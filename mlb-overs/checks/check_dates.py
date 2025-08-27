from datetime import datetime
import requests

print('🕐 DATE/TIMEZONE ANALYSIS:')
print('=' * 40)

now = datetime.now()
print(f'System date: {now.year}-{now.month:02d}-{now.day:02d} {now.hour:02d}:{now.minute:02d}')

# Test API endpoints
try:
    # Check games for today (2025-08-20)
    r20 = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
    games_20 = r20.json()['games'] if r20.status_code == 200 else []
    
    # Check games for tomorrow (2025-08-21)  
    r21 = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-21')
    games_21 = r21.json()['games'] if r21.status_code == 200 else []
    
    # Check /today endpoint
    r_today = requests.get('http://localhost:8000/api/comprehensive-games/today')
    games_today = r_today.json()['games'] if r_today.status_code == 200 else []
    
    print(f'Games on 2025-08-20: {len(games_20)}')
    print(f'Games on 2025-08-21: {len(games_21)}')
    print(f'Games from /today endpoint: {len(games_today)}')
    
    if games_today:
        print(f'First game from /today has date: {games_today[0]["date"]}')
        
    if games_20:
        first_game = games_20[0]
        print(f'First game on 2025-08-20: {first_game["away_team"]} @ {first_game["home_team"]}')
        if 'start_time' in first_game:
            print(f'Start time: {first_game["start_time"]}')
    
except Exception as e:
    print(f'Error: {e}')
