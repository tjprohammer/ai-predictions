#!/usr/bin/env python3

import requests

def test_comprehensive_endpoint():
    try:
        response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-29')
        data = response.json()
        games = data.get('games', [])
        
        if games:
            game = games[0]
            print('COMPREHENSIVE GAMES ENDPOINT (2025-08-29):')
            print(f'  Team: {game.get("away_team")} @ {game.get("home_team")}')
            print(f'  Learning Model: {game.get("predicted_total")}')
            print(f'  Ultra Sharp V15: {game.get("predicted_total_ultra")}')
            print(f'  Ultra Confidence: {game.get("ultra_confidence")}')
            print(f'  Market Total: {game.get("market_total")}')
            
            # Check a few more games to confirm they're all different
            print('\nChecking more games for variety:')
            for i, g in enumerate(games[:3]):
                learning = g.get('predicted_total')
                ultra = g.get('predicted_total_ultra')
                different = learning != ultra if (learning and ultra) else False
                print(f'  Game {i+1}: Learning={learning}, Ultra={ultra}, Different={different}')
                
            print(f'\nTotal games: {len(games)}')
        else:
            print(f'Error: {data.get("error")}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_comprehensive_endpoint()
