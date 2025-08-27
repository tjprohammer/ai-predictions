#!/usr/bin/env python3
"""
Quick Sportradar API Test
"""

import os
import requests
from dotenv import load_dotenv

# Load .env from the mlb-overs directory
env_path = os.path.join(os.path.dirname(__file__), 'mlb-overs', '.env')
load_dotenv(env_path)

api_key = os.getenv('SPORTRADAR_API_KEY')
if not api_key:
    print("❌ SPORTRADAR_API_KEY not found in .env file")
    exit(1)

print("🔑 API Key found:", api_key[:10] + "..." if len(api_key) > 10 else api_key)

# Test basic API access
base_url = "https://api.sportradar.com/mlb/trial/v8/en"
test_url = f"{base_url}/league/officials.json?api_key={api_key}"

print(f"🌐 Testing API endpoint: {base_url}/league/officials.json")

try:
    response = requests.get(test_url, timeout=10)
    print(f"📡 Response Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        # Check for umpire data
        if 'league' in data and 'officials' in data['league']:
            officials = data['league']['officials']
            print(f"✅ Success! Found {len(officials)} MLB officials")
            
            # Show sample umpires
            sample_umpires = officials[:5]
            print("\n👨‍⚖️ Sample umpires:")
            for ump in sample_umpires:
                name = ump.get('name', ump.get('full_name', 'Unknown'))
                ump_id = ump.get('id', 'No ID')
                print(f"   {name} (ID: {ump_id})")
        else:
            print("⚠️ Unexpected response structure")
            print("Response keys:", list(data.keys())[:5])
            
    elif response.status_code == 401:
        print("❌ Authentication failed - check your API key")
    elif response.status_code == 429:
        print("⚠️ Rate limit exceeded")
    else:
        print(f"❌ API request failed: {response.status_code}")
        print("Response:", response.text[:200])
        
except Exception as e:
    print(f"❌ Request failed: {e}")

print("\n🎯 If successful, you're ready to run the full umpire collector!")
