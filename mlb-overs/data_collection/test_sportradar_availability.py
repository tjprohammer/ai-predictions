#!/usr/bin/env python3
"""
Test Sportradar Data Availability for Recent Dates
"""

import os
import requests
from dotenv import load_dotenv
from datetime import date, timedelta

# Load environment variables from mlb-overs directory
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

api_key = os.getenv('SPORTRADAR_API_KEY')
base_url = "https://api.sportradar.com/mlb/trial/v8/en"

if not api_key:
    print("ERROR: SPORTRADAR_API_KEY not found")
    exit(1)

print("TESTING SPORTRADAR DATA AVAILABILITY")
print("=" * 50)

# Test dates from recent to older
test_dates = [
    date(2025, 8, 20),  # Yesterday 
    date(2025, 8, 19),  # 2 days ago
    date(2025, 8, 15),  # Last week
    date(2025, 8, 1),   # Start of August
    date(2025, 7, 15),  # Mid July
    date(2025, 6, 15),  # Mid June
    date(2025, 5, 15),  # Mid May
    date(2025, 4, 15),  # Mid April
    date(2024, 10, 15), # 2024 season
    date(2024, 9, 15),  # 2024 season
]

session = requests.Session()
session.params = {'api_key': api_key}

for test_date in test_dates:
    date_str = test_date.strftime('%Y/%m/%d')
    url = f"{base_url}/games/{date_str}/schedule.json"
    
    try:
        response = session.get(url, timeout=10)
        print(f"{test_date}: Status {response.status_code}", end="")
        
        if response.status_code == 200:
            data = response.json()
            games = data.get('league', {}).get('games', [])
            print(f" - {len(games)} games found")
            
            # If we found games, test boxscore for one
            if games:
                game_id = games[0].get('id')
                if game_id:
                    boxscore_url = f"{base_url}/games/{game_id}/boxscore.json"
                    box_response = session.get(boxscore_url, timeout=10)
                    print(f"  -> Boxscore for {game_id}: Status {box_response.status_code}")
                    
                    if box_response.status_code == 200:
                        box_data = box_response.json()
                        # Look for umpire data
                        game_info = box_data.get('game', {})
                        home_team = game_info.get('home_team', {}).get('name', 'Unknown')
                        away_team = game_info.get('away_team', {}).get('name', 'Unknown')
                        print(f"  -> Game: {away_team} @ {home_team}")
                        
                        # Check for officials
                        officials_found = False
                        search_paths = [
                            ['game', 'officials'],
                            ['officials'],
                            ['game', 'umpires'],
                            ['umpires']
                        ]
                        
                        for path in search_paths:
                            current = box_data
                            for key in path:
                                if isinstance(current, dict) and key in current:
                                    current = current[key]
                                else:
                                    break
                            else:
                                if current:
                                    officials_found = True
                                    print(f"  -> Found officials data: {len(current) if isinstance(current, list) else 'dict'}")
                                    if isinstance(current, list) and current:
                                        sample_official = current[0]
                                        print(f"  -> Sample: {sample_official}")
                                    break
                        
                        if not officials_found:
                            print(f"  -> No umpire data found in boxscore")
                            # Show available keys for debugging
                            if isinstance(game_info, dict):
                                available_keys = list(game_info.keys())[:10]
                                print(f"  -> Available game keys: {available_keys}")
        elif response.status_code == 429:
            print(" - RATE LIMITED")
            break
        else:
            print(f" - ERROR")
            
    except Exception as e:
        print(f"{test_date}: ERROR - {e}")
    
    # Small delay between requests
    import time
    time.sleep(1)

print("\nTEST COMPLETE!")
print("If recent dates show games but no umpire data,")
print("the trial API may not include umpire assignments.")
print("Consider using simulated umpire data for Phase 4.")
