#!/usr/bin/env python3
"""Test what the frontend sees"""

import requests
import json

response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-19')
data = response.json()
game = data['games'][0]

print('Frontend API Response:')
print(f'Game: {game["away_team"]} @ {game["home_team"]}')
print(f'Has ai_analysis: {"ai_analysis" in game}')

if 'ai_analysis' in game:
    ai = game['ai_analysis']
    print(f'Prediction summary: {ai.get("prediction_summary", "None")}')
    print(f'Primary factors: {len(ai.get("primary_factors", []))} items')
    print(f'Key insights: {len(ai.get("key_insights", []))} items')
    print('\nFull AI Analysis:')
    print(json.dumps(ai, indent=2))
else:
    print('No AI analysis in response!')
    print(f'Available keys: {list(game.keys())}')
