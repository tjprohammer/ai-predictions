#!/usr/bin/env python3
"""
Test OpenWeather API Key
"""

import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

def test_api_key():
    """Test if the OpenWeather API key is valid"""
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        print("❌ No API key found in environment variables!")
        return False
    
    print(f"🔑 API Key found: {api_key[:10]}...{api_key[-5:]}")
    
    # Test the API key with a simple request
    url = f'http://api.openweathermap.org/data/2.5/weather?q=Denver,CO&appid={api_key}&units=imperial'
    
    try:
        response = requests.get(url)
        print(f"📡 API Response Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            condition = data['weather'][0]['main']
            print(f"✅ API Working! Denver Weather: {temp}°F, {condition}")
            return True
        elif response.status_code == 401:
            print("❌ API Key is INVALID or EXPIRED!")
            print(f"Response: {response.text}")
            return False
        else:
            print(f"⚠️ API Error {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Request Error: {e}")
        return False

def check_env_file():
    """Check what's in the .env file"""
    print("\n📄 Checking .env file content...")
    
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines, 1):
                if 'OPENWEATHER' in line:
                    print(f"Line {i}: {line.strip()}")
    else:
        print("❌ No .env file found!")

if __name__ == "__main__":
    print("🧪 Testing OpenWeather API Key...")
    print("=" * 40)
    
    check_env_file()
    test_api_key()
