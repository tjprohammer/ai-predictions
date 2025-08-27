#!/usr/bin/env python3
"""
Test Dual Predictions API
"""

import requests
import json

def test_dual_predictions_api():
    try:
        response = requests.get('http://localhost:8000/api/dual-predictions/today')
        if response.status_code == 200:
            data = response.json()
            print('🎉 Dual Predictions API Working!')
            summary = data['summary']
            print(f'📊 Summary:')
            print(f'   Total games: {summary["total_games"]}')
            print(f'   Dual predictions available: {summary["dual_predictions_available"]}')
            print(f'   Learning model higher: {summary["learning_higher_count"]} games')
            print(f'   Original model higher: {summary["original_higher_count"]} games')
            print(f'   Average difference: {summary["avg_difference"]} runs')
            print(f'   Model agreement rate: {summary["model_agreement_rate"]}%')
            
            print(f'\n🎯 First few games:')
            for i, game in enumerate(data['games'][:3]):
                orig = game['predictions']['original']
                learn = game['predictions']['learning']
                diff = game['comparison']['difference'] if game['comparison']['difference'] else 0
                print(f'   {i+1}. {game["matchup"]}')
                print(f'      Original: {orig:.2f} | Learning: {learn:.2f} | Diff: {diff:+.2f}')
                
            return True
        else:
            print(f'❌ API returned status code: {response.status_code}')
            print(response.text)
            return False
    except Exception as e:
        print(f'❌ Error testing API: {e}')
        print('Make sure the API server is running on port 8000')
        return False

if __name__ == "__main__":
    test_dual_predictions_api()
