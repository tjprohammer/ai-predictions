#!/usr/bin/env python3

import requests
import json

def test_api():
    try:
        # Test comprehensive games endpoint
        print("=== TESTING COMPREHENSIVE GAMES ENDPOINT ===")
        response = requests.get('http://localhost:8000/api/comprehensive-games/today')
        data = response.json()
        games = data.get('games', [])
        
        if games:
            game = games[0]  # First game
            print('Comprehensive Games API Response for first game:')
            print(f'  Team: {game.get("away_team")} @ {game.get("home_team")}')
            print(f'  Learning Model: {game.get("predicted_total")}')
            print(f'  Ultra 80: {game.get("predicted_total_learning")}') 
            print(f'  Original: {game.get("predicted_total_original")}')
            print(f'  Ultra Sharp V15: {game.get("predicted_total_ultra")}')
            print(f'  Ultra Confidence: {game.get("ultra_confidence")}')
            print(f'  Market Total: {game.get("market_total")}')
            print()
        else:
            print(f'No games found. Error: {data.get("error", "Unknown")}')
            
        # Test dual predictions endpoint
        print("=== TESTING DUAL PREDICTIONS ENDPOINT ===")
        response2 = requests.get('http://localhost:8000/api/model-predictions/2025-08-29')
        data2 = response2.json()
        games2 = data2.get('games', [])
        
        if games2:
            game2 = games2[0]  # First game
            print('Dual Predictions API Response for first game:')
            print(f'  Team: {game2.get("away_team")} @ {game2.get("home_team")}')
            
            learning = game2.get('learning_model', {})
            ultra_80 = game2.get('ultra_80', {})
            ultra_sharp = game2.get('ultra_sharp_v15', {})
            
            print(f'  Learning Model: {learning.get("prediction") if learning else "None"}')
            print(f'  Ultra 80: {ultra_80.get("prediction") if ultra_80 else "None"}')
            print(f'  Ultra Sharp V15: {ultra_sharp.get("prediction") if ultra_sharp else "None"}')
            print(f'  Ultra Sharp V15 Conf: {ultra_sharp.get("confidence") if ultra_sharp else "None"}')
            print(f'  Market Total: {game2.get("market", {}).get("total")}')
            print()
            
            print(f'Total games in dual endpoint: {len(games2)}')
        else:
            print(f'No games found in dual endpoint. Error: {data2.get("error", "Unknown")}')
            
    except Exception as e:
        print(f'Error: {e}')

if __name__ == "__main__":
    test_api()
