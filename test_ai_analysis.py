#!/usr/bin/env python3
"""Test enhanced API with AI analysis"""

import requests
import json

# Test the enhanced API with AI analysis
response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-19')
data = response.json()

print('🚀 ENHANCED API WITH AI ANALYSIS')
print('=' * 60)
print(f'Total games: {data["count"]}')
print(f'High-confidence games: {data.get("high_confidence_count", 0)}')
print(f'Premium picks: {data.get("premium_pick_count", 0)}')
print()

# Show top 3 games with AI analysis
games = data['games'][:3]
for i, game in enumerate(games, 1):
    conf = game.get('confidence', 0)
    rec = game.get('recommendation', 'HOLD')
    edge = game.get('edge', 0)
    
    print(f'🎯 GAME {i}: {game["away_team"]} @ {game["home_team"]}')
    print(f'Confidence: {conf}% | Recommendation: {rec} | Edge: {float(edge):+.2f}')
    
    if game.get('is_premium_pick'):
        print('⭐ PREMIUM PICK ⭐')
    elif game.get('is_high_confidence'):
        print('🔥 HIGH CONFIDENCE')
    
    ai = game.get('ai_analysis', {})
    print(f'📊 {ai.get("prediction_summary", "No analysis")}')
    print(f'🎯 {ai.get("recommendation_reasoning", "No reasoning")}')
    
    if ai.get('primary_factors'):
        print('🔑 Primary factors:')
        for factor in ai['primary_factors']:
            print(f'   • {factor}')
    
    if ai.get('key_insights'):
        print('💡 Key insights:')
        for insight in ai['key_insights']:
            print(f'   • {insight}')
    
    print('-' * 60)
