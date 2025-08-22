#!/usr/bin/env python3
"""
Sportradar API Test - Find Real Game Dates with Umpire Data

Purpose: Test different date ranges to find where umpire data actually exists
Context: MLB v8 API confirmed to have umpire data, need to find correct dates
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

def make_request(endpoint, delay=2):
    """Make API request with rate limiting"""
    time.sleep(delay)
    url = f"{base_url}{endpoint}?api_key={api_key}"
    
    try:
        response = requests.get(url, timeout=15)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            print(f"Rate limited, waiting 30s...")
            time.sleep(30)
            return make_request(endpoint, delay)
        else:
            print(f"API Error {response.status_code} for {endpoint}")
            return None
    except Exception as e:
        print(f"Request failed for {endpoint}: {e}")
        return None

def test_date_ranges():
    """Test different date ranges to find where games/umpires exist"""
    
    print("TESTING MULTIPLE DATE RANGES FOR MLB GAMES")
    print("=" * 60)
    
    # Test ranges - focusing on 2024 MLB season
    test_ranges = [
        ("2024-04-01", "2024 Opening Day area"),
        ("2024-05-15", "2024 Mid-May"),
        ("2024-07-15", "2024 Mid-July"), 
        ("2024-08-15", "2024 Mid-August"),
        ("2024-09-15", "2024 Late September"),
        ("2024-10-01", "2024 Playoffs start"),
        ("2025-04-01", "2025 Season (if available)"),
        ("2025-08-20", "Current date area")
    ]
    
    for test_date, description in test_ranges:
        print(f"\nTesting {test_date} ({description}):")
        print("-" * 40)
        
        # Get schedule for this date
        data = make_request(f'/games/{test_date.replace("-", "/")}/schedule.json')
        
        if not data:
            print("  No API response")
            continue
            
        # Look for games in response
        games = []
        if 'league' in data:
            if 'games' in data['league']:
                games = data['league']['games']
            elif 'daily_schedule' in data['league']:
                games = data['league']['daily_schedule'].get('games', [])
        
        if not games:
            print(f"  No games found for {test_date}")
            continue
            
        print(f"  Found {len(games)} games!")
        
        # Test first game for umpire data
        if games:
            game = games[0]
            game_id = game.get('id', game.get('game_id'))
            home_team = game.get('home_team', {}).get('name', game.get('home', 'Unknown'))
            away_team = game.get('away_team', {}).get('name', game.get('away', 'Unknown'))
            
            print(f"  Sample game: {away_team} @ {home_team} (ID: {game_id})")
            
            if game_id:
                print(f"  Testing game umpire data...")
                
                # Try boxscore endpoint for this game
                boxscore = make_request(f'/games/{game_id}/boxscore.json')
                if boxscore:
                    # Look for umpire data in various locations
                    umpire_locations = [
                        ['game', 'officials'],
                        ['officials'], 
                        ['game', 'umpires'],
                        ['umpires'],
                        ['game', 'crew']
                    ]
                    
                    found_umpires = False
                    for location in umpire_locations:
                        current = boxscore
                        for key in location:
                            if isinstance(current, dict) and key in current:
                                current = current[key]
                            else:
                                break
                        else:
                            if current and isinstance(current, (list, dict)):
                                print(f"    FOUND UMPIRE DATA at {' -> '.join(location)}")
                                if isinstance(current, list) and len(current) > 0:
                                    sample = current[0]
                                    print(f"    Sample umpire: {sample}")
                                elif isinstance(current, dict):
                                    print(f"    Umpire data keys: {list(current.keys())}")
                                found_umpires = True
                                break
                    
                    if not found_umpires:
                        print(f"    No umpire data found in boxscore")
                        # Show what keys are available
                        if isinstance(boxscore, dict):
                            print(f"    Available keys: {list(boxscore.keys())}")
                            if 'game' in boxscore and isinstance(boxscore['game'], dict):
                                print(f"    Game keys: {list(boxscore['game'].keys())}")
                else:
                    print(f"    Failed to get boxscore data")
        
        # Rate limiting
        time.sleep(3)

def test_officials_endpoint():
    """Test the Officials endpoint"""
    print("\n" + "="*60)
    print("TESTING OFFICIALS ENDPOINT")
    print("="*60)
    
    data = make_request('/league/officials.json')
    if data and 'league' in data and 'officials' in data['league']:
        officials = data['league']['officials']
        print(f"SUCCESS: Found {len(officials)} MLB officials")
        
        # Show sample officials
        print("\nSample officials:")
        for i, official in enumerate(officials[:5], 1):
            name = official.get('name', official.get('full_name', 'Unknown'))
            ump_id = official.get('id', 'No ID')
            tenure = official.get('tenure', 'No tenure')
            print(f"  {i}. {name} (ID: {ump_id}, Tenure: {tenure})")
            
        return officials
    else:
        print("FAILED: Could not load officials data")
        return []

def main():
    print("SPORTRADAR MLB UMPIRE DATA INVESTIGATION")
    print("=" * 60)
    print(f"API Key: {api_key[:15]}...")
    print(f"Base URL: {base_url}")
    print()
    
    # Test officials endpoint first
    officials = test_officials_endpoint()
    
    # Test date ranges for games with umpire data
    test_date_ranges()
    
    print("\n" + "="*60)
    print("INVESTIGATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. If officials loaded but games have no umpire data -> Try different endpoints")
    print("2. If specific dates work -> Focus collection on those dates")
    print("3. If no umpire data found -> API might not include umpire assignments")

if __name__ == "__main__":
    main()
