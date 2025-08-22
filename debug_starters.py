import requests
import json

# Test with a recent game
game_id = 776652  # Cincinnati @ Angels
url = f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore'

response = requests.get(url, timeout=10)
data = response.json()

print('🔍 Looking for starting pitchers...')

# Check pitchers list
home_pitchers = data['teams']['home']['pitchers']
away_pitchers = data['teams']['away']['pitchers'] 

print(f'Home pitchers: {home_pitchers}')
print(f'Away pitchers: {away_pitchers}')

# Get the first pitcher (usually starter)
if home_pitchers:
    home_starter_id = f'ID{home_pitchers[0]}'
    print(f'\nHome starter ID: {home_starter_id}')
    
    if home_starter_id in data['teams']['home']['players']:
        player = data['teams']['home']['players'][home_starter_id]
        print(f'Home starter: {player["person"]["fullName"]}')
        
        if 'stats' in player and 'pitching' in player['stats']:
            pitching = player['stats']['pitching']
            print(f'Pitching stats available:')
            for key, value in pitching.items():
                print(f'  {key}: {value}')
        else:
            print('No pitching stats found')

if away_pitchers:
    away_starter_id = f'ID{away_pitchers[0]}'
    print(f'\nAway starter ID: {away_starter_id}')
    
    if away_starter_id in data['teams']['away']['players']:
        player = data['teams']['away']['players'][away_starter_id]
        print(f'Away starter: {player["person"]["fullName"]}')
        
        if 'stats' in player and 'pitching' in player['stats']:
            pitching = player['stats']['pitching']
            print(f'Pitching stats available:')
            for key, value in pitching.items():
                print(f'  {key}: {value}')
        else:
            print('No pitching stats found')
