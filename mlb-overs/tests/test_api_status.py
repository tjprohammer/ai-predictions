#!/usr/bin/env python3
import requests
import json

def test_api_status():
    """Test API status and enhanced features availability"""
    try:
        response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
        data = response.json()
        games = data.get('games', [])
        
        print(f'API Response: {len(games)} games found')
        print('=' * 50)
        
        if games:
            game = games[0]
            print(f'Sample Game: {game.get("away_team", "N/A")} @ {game.get("home_team", "N/A")}')
            print()
            
            # Check for enhanced features that the UI expects
            enhanced_features = [
                'calibrated_predictions',
                'confidence_level', 
                'is_strong_pick',
                'is_premium_pick',
                'is_high_confidence',
                'ai_analysis'
            ]
            
            print('ENHANCED FEATURES STATUS:')
            for feature in enhanced_features:
                has_feature = feature in game
                value = game.get(feature, 'MISSING')
                status = '✅' if has_feature else '❌'
                print(f'  {feature}: {status} {value if has_feature else "Not found"}')
            
            print()
            print('BASIC PREDICTION DATA:')
            basic_features = ['predicted_total', 'market_total', 'recommendation', 'confidence']
            for feature in basic_features:
                has_feature = feature in game
                value = game.get(feature, 'MISSING')
                status = '✅' if has_feature else '❌'
                print(f'  {feature}: {status} {value}')
                
            # Show all available keys for debugging
            print()
            print('ALL AVAILABLE KEYS:')
            for key in sorted(game.keys()):
                print(f'  {key}')
        else:
            print('No games found in response')
            print('Raw response:', data)
            
    except Exception as e:
        print(f'Error: {e}')
        print('API may not be running on localhost:8000')

if __name__ == "__main__":
    test_api_status()
