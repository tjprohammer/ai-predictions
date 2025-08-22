from datetime import datetime
import requests

today_str = datetime.now().strftime('%Y-%m-%d')
print(f'Today string: {today_str}')

# Test direct call
r_direct = requests.get(f'http://localhost:8000/api/comprehensive-games/{today_str}')
print(f'Direct call to /{today_str}: {len(r_direct.json()["games"])} games')

# Test /today endpoint
r_today = requests.get('http://localhost:8000/api/comprehensive-games/today')
print(f'Call to /today: {r_today.status_code}')
if r_today.status_code == 200:
    today_data = r_today.json()
    print(f'/today games: {len(today_data["games"])}')
    if "date" in today_data:
        print(f'/today date field: {today_data["date"]}')
else:
    print(f'/today error: {r_today.text}')
