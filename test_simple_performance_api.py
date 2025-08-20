#!/usr/bin/env python3
"""
Simple Performance API Test
"""

import requests
import json

def test_simple_performance():
    """Test a simple performance endpoint"""
    try:
        response = requests.get("http://localhost:8000/api/simple-performance?days=7")
        if response.status_code == 200:
            data = response.json()
            print("✅ Simple performance API working:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ API error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_simple_performance()
