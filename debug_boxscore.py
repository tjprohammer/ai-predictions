import requests
import json

# Test with a recent game to see the actual boxscore structure
game_id = 776652  # Cincinnati @ Angels
url = f'https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore'

print('🔍 Testing MLB API boxscore structure...')
response = requests.get(url, timeout=10)

if response.status_code == 200:
    data = response.json()
    
    print('📊 Available top-level keys:')
    for key in data.keys():
        print(f'  {key}')
    
    if 'teams' in data:
        print()
        print('🏠 Home team data structure:')
        home = data['teams']['home']
        for key in home.keys():
            print(f'  {key}: {type(home[key])}')
        
        # Check if we have player stats
        if 'players' in home:
            print()
            print('👥 Home team has player data')
            print(f'   Number of players: {len(home["players"])}')
            
            # Look for pitching stats in players
            for player_id, player_data in list(home['players'].items())[:3]:
                print(f'   Player {player_id}: {player_data.get("person", {}).get("fullName", "Unknown")}')
                if 'stats' in player_data and 'pitching' in player_data['stats']:
                    print(f'     Has pitching stats!')
                    pitching = player_data['stats']['pitching']
                    print(f'     IP: {pitching.get("inningsPitched", "N/A")}')
                    print(f'     ER: {pitching.get("earnedRuns", "N/A")}')
                    print(f'     K: {pitching.get("strikeOuts", "N/A")}')
else:
    print(f'❌ API error: {response.status_code}')
