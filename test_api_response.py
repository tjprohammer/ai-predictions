#!/usr/bin/env python3

import requests
import json

def test_api():
    try:
        # Test health first
        response = requests.get('http://localhost:8000/health')
        print(f'Health check: {response.status_code}')
        
        # Test comprehensive games endpoint
        response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-29')
        print(f'Games endpoint status: {response.status_code}')
        
        data = response.json()
        
        if 'error' in data:
            print(f'API Error: {data.get("error")}')
            return
            
        games = data.get('games', [])
        
        if games:
            game = games[0]  # First game
            print('API Response for first game:')
            print(f'  Team: {game.get("away_team")} @ {game.get("home_team")}')
            print(f'  Learning Model: {game.get("predicted_total")}')
            print(f'  Ultra 80: {game.get("predicted_total_learning")}') 
            print(f'  Original: {game.get("predicted_total_original")}')
            print(f'  Ultra Sharp V15: {game.get("predicted_total_ultra")}')
            print(f'  Ultra Confidence: {game.get("ultra_confidence")}')
            print(f'  Market Total: {game.get("market_total")}')
            print()
            
            if len(games) > 1:
                print('Second game:')
                game2 = games[1]
                print(f'  Team: {game2.get("away_team")} @ {game2.get("home_team")}')
                print(f'  Learning Model: {game2.get("predicted_total")}')
                print(f'  Ultra Sharp V15: {game2.get("predicted_total_ultra")}')
                print()
                
            print(f'Total games returned: {len(games)}')
        else:
            print('No games found')
            print(f'Data source: {data.get("data_source")}')
            print(f'Response keys: {list(data.keys())}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_api()
