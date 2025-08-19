#!/usr/bin/env python3
"""
Test The Odds API for Real Market Totals
========================================
"""

import os
import requests
import json
from datetime import datetime

def test_odds_api():
    """Test The Odds API with real key"""
    
    api_key = os.getenv("THE_ODDS_API_KEY")
    
    if not api_key:
        print("❌ THE_ODDS_API_KEY environment variable not set")
        print("Set it with: $env:THE_ODDS_API_KEY = 'your_key_here'")
        return
    
    print(f"🔑 Using API key: {api_key[:8]}...")
    
    # Test API connection
    url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
        "dateFormat": "iso"
    }
    
    try:
        print("🎯 Fetching MLB odds from The Odds API...")
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ SUCCESS: Retrieved {len(data)} games")
        
        # Show real market totals
        for game in data[:3]:  # Show first 3 games
            away_team = game.get('away_team', 'Unknown')
            home_team = game.get('home_team', 'Unknown')
            
            print(f"\n📊 {away_team} @ {home_team}")
            
            bookmakers = game.get('bookmakers', [])
            if bookmakers:
                for book in bookmakers[:2]:  # Show first 2 bookmakers
                    book_name = book.get('title', 'Unknown')
                    
                    for market in book.get('markets', []):
                        if market.get('key') == 'totals':
                            for outcome in market.get('outcomes', []):
                                point = outcome.get('point')
                                price = outcome.get('price')
                                name = outcome.get('name')
                                
                                if point:
                                    print(f"   {book_name}: {name} {point} ({price:+d})")
            else:
                print("   No bookmaker data available")
        
        # Check remaining requests
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')
        
        if remaining:
            print(f"\n📈 API Usage: {used} used, {remaining} remaining")
        
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🎯 Testing The Odds API for Real Market Totals")
    print("=" * 50)
    
    data = test_odds_api()
    
    if data:
        print(f"\n✅ The Odds API is working! Found {len(data)} games")
        print("Ready to collect real market totals 🎯")
    else:
        print("\n❌ The Odds API test failed")
