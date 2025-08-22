import requests
import json

try:
    response = requests.get('http://localhost:8000/api/dual-predictions/today')
    if response.status_code == 200:
        data = response.json()
        print('API Response Summary:')
        print(f'  Status: SUCCESS')
        print(f'  Games: {len(data.get("games", []))}')
        if data.get('games'):
            first_game = data['games'][0]
            print(f'  Sample game: {first_game.get("matchup", "N/A")}')
            print(f'  Original: {first_game.get("predictions", {}).get("original", "N/A")}')
            print(f'  Learning: {first_game.get("predictions", {}).get("learning", "N/A")}')
            print(f'  Difference: {first_game.get("comparison", {}).get("difference", "N/A")}')
    else:
        print(f'API Error: {response.status_code} - {response.text}')
except Exception as e:
    print(f'Connection Error: {e}')
