import requests
import json

# Test comprehensive games API
print("Testing comprehensive games API...")
r = requests.get('http://127.0.0.1:8000/api/comprehensive-games/2025-08-14')
data = r.json()

print(f"Games count: {len(data.get('games', []))}")
if data.get('games'):
    game = data['games'][0]
    print(f"Sample game has historical_prediction: {'historical_prediction' in game}")
    if 'historical_prediction' in game:
        print(f"Historical prediction: {game['historical_prediction']}")
    print(f"Game keys: {list(game.keys())}")

print("\n" + "="*50 + "\n")

# Test ML predictions API
print("Testing ML predictions API...")
r2 = requests.get('http://127.0.0.1:8000/api/ml-predictions/2025-08-14')
ml_data = r2.json()

print(f"ML Predictions count: {len(ml_data.get('predictions', []))}")
print(f"Summary total games: {ml_data.get('summary', {}).get('total_games')}")
if ml_data.get('predictions'):
    pred = ml_data['predictions'][0]
    print(f"First prediction: {pred['away_team']} @ {pred['home_team']}")
    print(f"Predicted: {pred['predicted_total']}, Market: {pred['market_total']}, Rec: {pred['recommendation']}")
