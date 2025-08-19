#!/usr/bin/env python3
"""
Test the MLB API with the populated pitcher data
"""

import requests
import json

def test_api():
    # Test the API with pitchers that have data from today's games
    test_url = 'http://localhost:8000/predict'
    test_params = {
        'home_team': 'NYY',
        'away_team': 'BOS', 
        'home_sp_id': '641302',  # Has data with 0.0 ERA
        'away_sp_id': '685126'   # Has data with 13.5 ERA
    }
    
    print("Testing MLB Predictions API...")
    print(f"URL: {test_url}")
    print(f"Parameters: {test_params}")
    print("-" * 50)
    
    try:
        response = requests.get(test_url, params=test_params, timeout=30)
        print(f'API Response Status: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print('API Response:')
            print(json.dumps(data, indent=2))
            
            # Check if pitcher ERA values are populated
            if 'home_pitcher_era' in data and data['home_pitcher_era'] is not None:
                print(f"\n✅ SUCCESS: Home pitcher ERA = {data['home_pitcher_era']}")
            else:
                print(f"\n❌ ISSUE: Home pitcher ERA still null")
                
            if 'away_pitcher_era' in data and data['away_pitcher_era'] is not None:
                print(f"✅ SUCCESS: Away pitcher ERA = {data['away_pitcher_era']}")
            else:
                print(f"❌ ISSUE: Away pitcher ERA still null")
        else:
            print('Error Response:')
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f'Request failed: {e}')
        print('Make sure the API server is running on localhost:8001')

if __name__ == "__main__":
    test_api()
