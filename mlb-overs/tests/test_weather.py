import requests

r = requests.get('http://127.0.0.1:8000/api/comprehensive-games/2025-08-14')
data = r.json()

print('API Response:')
print(f'  Games found: {len(data.get("games", []))}')

if data.get('games'):
    game = data['games'][0]
    print(f'  First game: {game["away_team"]} @ {game["home_team"]}')
    print(f'  Weather: {game["weather"]}')
    
    # Check if Coors Field game exists
    coors_game = None
    for g in data['games']:
        if 'Coors' in g['venue'] or 'Colorado' in g['venue']:
            coors_game = g
            break
    
    if coors_game:
        print(f'\nCoors Field Game:')
        print(f'  {coors_game["away_team"]} @ {coors_game["home_team"]}')
        print(f'  Weather: {coors_game["weather"]}')
