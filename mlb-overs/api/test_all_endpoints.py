import requests
import json

def test_endpoint(url, name):
    print(f"\n🧪 Testing {name}...")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success! {name}")
            if 'summary' in data:
                print(f"   Games: {data['summary']['total_games']}")
                print(f"   Dual predictions: {data['summary']['dual_predictions_available']}")
            elif 'total_games' in data:
                print(f"   Games: {data['total_games']}")
            return True
        else:
            print(f"❌ Failed! Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

# Test all dual prediction endpoints
base_url = "http://localhost:8000"

endpoints = [
    (f"{base_url}/api/dual-predictions/today", "Today's Dual Predictions"),
    (f"{base_url}/api/dual-predictions/tomorrow", "Tomorrow's Dual Predictions"),  
    (f"{base_url}/api/dual-predictions/2025-08-22", "Specific Date Dual Predictions"),
    (f"{base_url}/api/dual-performance/7", "7-Day Performance Summary"),
    (f"{base_url}/api/dual-historical/2025-08-20/2025-08-22", "Historical Analysis")
]

success_count = 0
for url, name in endpoints:
    if test_endpoint(url, name):
        success_count += 1

print(f"\n📊 Results: {success_count}/{len(endpoints)} endpoints working")

if success_count == len(endpoints):
    print("🎉 All dual prediction API endpoints are working perfectly!")
else:
    print("⚠️ Some endpoints need attention")
