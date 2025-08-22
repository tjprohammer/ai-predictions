#!/usr/bin/env python3
"""
Sportradar MLB Umpire Data Collection - FIXED
Using correct date format and recent MLB season dates
"""

import os
import requests
import time
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

# Load from correct .env location
env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(env_path)

api_key = os.getenv('SPORTRADAR_API_KEY')
if not api_key:
    print("ERROR: SPORTRADAR_API_KEY not found")
    exit(1)

base_url = "https://api.sportradar.com/mlb/trial/v8/en"

def make_request(endpoint, delay=3):
    """Make API request with rate limiting"""
    time.sleep(delay)
    url = f"{base_url}{endpoint}?api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Rate limited, waiting 60s...")
            time.sleep(60)
            return make_request(endpoint, delay*2)
        else:
            print(f"API Error {response.status_code} for {endpoint}")
            return None
    except Exception as e:
        print(f"Request failed for {endpoint}: {e}")
        return None

def test_recent_mlb_dates():
    """Test recent MLB season dates with correct format"""
    
    print("TESTING RECENT MLB SEASON DATES")
    print("=" * 50)
    
    # Test recent 2024 MLB season dates (these should have games)
    test_dates = [
        ("2024", "10", "30"),  # 2024 World Series Game 5
        ("2024", "10", "25"),  # 2024 World Series Game 1  
        ("2024", "09", "29"),  # 2024 Regular season end
        ("2024", "09", "15"),  # 2024 Late season
        ("2024", "08", "15"),  # 2024 Mid season
        ("2024", "07", "30"),  # 2024 Trade deadline time
        ("2024", "06", "15"),  # 2024 Mid season
        ("2024", "04", "15"),  # 2024 Early season
    ]
    
    for year, month, day in test_dates:
        date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        print(f"\nTesting {date_str}:")
        print("-" * 30)
        
        # Use correct format: /games/{year}/{month}/{day}/schedule.json
        endpoint = f'/games/{year}/{month}/{day}/schedule.json'
        data = make_request(endpoint)
        
        if not data:
            print(f"  No API response")
            continue
            
        # Look for games
        games = []
        if 'league' in data:
            league_data = data['league']
            if 'games' in league_data:
                games = league_data['games']
            elif 'daily_schedule' in league_data and 'games' in league_data['daily_schedule']:
                games = league_data['daily_schedule']['games']
        
        if not games:
            print(f"  No games found")
            continue
            
        print(f"  SUCCESS: Found {len(games)} games!")
        
        # Test first game for umpire data
        game = games[0]
        game_id = game.get('id')
        home_team = game.get('home_team', {}).get('name', game.get('home', 'Unknown'))
        away_team = game.get('away_team', {}).get('name', game.get('away', 'Unknown'))
        
        print(f"  Sample: {away_team} @ {home_team}")
        print(f"  Game ID: {game_id}")
        
        if game_id:
            # Test boxscore endpoint for umpire data
            print(f"  Testing umpire data...")
            boxscore = make_request(f'/games/{game_id}/boxscore.json')
            
            if boxscore:
                found_umpires = find_umpire_data(boxscore, "boxscore")
                
                if not found_umpires:
                    # Try game summary endpoint
                    print(f"  Trying summary endpoint...")
                    summary = make_request(f'/games/{game_id}/summary.json')
                    if summary:
                        found_umpires = find_umpire_data(summary, "summary")
                
                if found_umpires:
                    print(f"  UMPIRE DATA FOUND!")
                    return True  # Success! We found umpire data
                else:
                    print(f"  No umpire data in this game")
        
        # Don't test all dates if we hit rate limits
        print(f"  Waiting to avoid rate limits...")
        time.sleep(5)
    
    return False

def find_umpire_data(game_data, source):
    """Look for umpire data in game response"""
    
    umpire_paths = [
        ['game', 'officials'],
        ['officials'],
        ['game', 'umpires'], 
        ['umpires'],
        ['game', 'crew'],
        ['crew'],
        ['officials', 'umpires'],
        ['game', 'venue', 'officials']
    ]
    
    for path in umpire_paths:
        current = game_data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                break
        else:
            # Found something at this path
            if current and isinstance(current, (list, dict)):
                print(f"    Found umpire data at {' -> '.join(path)}")
                
                if isinstance(current, list):
                    print(f"    {len(current)} umpires:")
                    for i, ump in enumerate(current[:3]):  # Show first 3
                        if isinstance(ump, dict):
                            name = ump.get('name', ump.get('full_name', 'Unknown'))
                            position = ump.get('position', ump.get('assignment', 'Unknown'))
                            print(f"      {i+1}. {name} - {position}")
                        else:
                            print(f"      {i+1}. {ump}")
                elif isinstance(current, dict):
                    print(f"    Umpire data keys: {list(current.keys())}")
                    # Look for specific umpire positions
                    for pos in ['HP', 'home_plate', 'plate', '1B', 'first_base', '2B', 'second_base', '3B', 'third_base']:
                        if pos in current:
                            print(f"      {pos}: {current[pos]}")
                
                return True
    
    # Show what keys are available for debugging
    if isinstance(game_data, dict):
        print(f"    Available top-level keys in {source}: {list(game_data.keys())}")
        if 'game' in game_data and isinstance(game_data['game'], dict):
            game_keys = list(game_data['game'].keys())
            print(f"    Available game keys: {game_keys}")
    
    return False

def main():
    print("SPORTRADAR MLB UMPIRE DATA INVESTIGATION - FIXED")
    print("=" * 60)
    print(f"Using correct date format: /games/{{year}}/{{month}}/{{day}}/schedule.json")
    print()
    
    # Test officials endpoint first (we know this works)
    print("1. TESTING OFFICIALS ENDPOINT...")
    officials = make_request('/league/officials.json')
    if officials and 'league' in officials and 'officials' in officials['league']:
        print(f"   SUCCESS: {len(officials['league']['officials'])} MLB officials loaded")
    else:
        print("   FAILED: Could not load officials")
        return
    
    print("\n2. TESTING RECENT MLB GAME DATES...")
    success = test_recent_mlb_dates()
    
    print("\n" + "="*60)
    print("INVESTIGATION RESULTS:")
    print("="*60)
    
    if success:
        print("SUCCESS: Found MLB games with umpire assignment data!")
        print("Next step: Implement full historical collection")
    else:
        print("ISSUE: Games found but no umpire assignments detected")
        print("Possible causes:")
        print("1. Trial API doesn't include umpire assignments")
        print("2. Umpire data is in different location/format")
        print("3. Need different endpoints or parameters")
        print("\nRecommendation: Use simulated umpire data for Phase 4")

if __name__ == "__main__":
    main()
