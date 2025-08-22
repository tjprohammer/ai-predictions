import requests
import json

try:
    response = requests.get('http://localhost:8000/api/dual-predictions/today')
    print(f'Status code: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print('✅ Success! API endpoint working')
        print('Summary:')
        print(json.dumps(data['summary'], indent=2))
        print('First game:', data["games"][0]["matchup"])
        print('  Original:', data["games"][0]["predictions"]["original"])
        print('  Learning:', data["games"][0]["predictions"]["learning"])
    else:
        print('❌ Error response:', response.text)
except Exception as e:
    print('❌ Request failed:', e)
