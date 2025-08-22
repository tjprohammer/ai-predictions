import requests

response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
if response.status_code == 200:
    games = response.json()['games']
    game = games[0]
    
    print('🎯 Enhanced Team Stats Test:')
    print('=' * 50)
    
    home_stats = game['team_stats']['home']
    away_stats = game['team_stats']['away']
    
    print(f'HOME ({game["home_team"]}):')
    print(f'  Season BA: {home_stats.get("season_ba", "N/A")}')
    print(f'  Last 5 BA: {home_stats.get("last_5_ba", "N/A")}')  
    print(f'  Season R/G: {home_stats.get("season_rpg", "N/A")}')
    print(f'  Last 5 R/G: {home_stats.get("last_5_rpg", "N/A")}')
    print(f'  Form: {home_stats.get("form_status", "unknown")} - {home_stats.get("form_description", "")}')
    print()
    print(f'AWAY ({game["away_team"]}):')
    print(f'  Season BA: {away_stats.get("season_ba", "N/A")}')
    print(f'  Last 5 BA: {away_stats.get("last_5_ba", "N/A")}')
    print(f'  Season R/G: {away_stats.get("season_rpg", "N/A")}')
    print(f'  Last 5 R/G: {away_stats.get("last_5_rpg", "N/A")}')
    print(f'  Form: {away_stats.get("form_status", "unknown")} - {away_stats.get("form_description", "")}')
    
    # Test a few more games for hot/cold teams
    print(f'\n🔥 HOT/COLD TEAMS TODAY:')
    for g in games[:5]:
        home = g['team_stats']['home']
        away = g['team_stats']['away']
        
        if home.get('form_status') == 'hot':
            print(f'  🔥 {g["home_team"]}: {home.get("form_description", "")}')
        if away.get('form_status') == 'hot':
            print(f'  🔥 {g["away_team"]}: {away.get("form_description", "")}')
        if home.get('form_status') == 'cold':
            print(f'  ❄️ {g["home_team"]}: {home.get("form_description", "")}')
        if away.get('form_status') == 'cold':
            print(f'  ❄️ {g["away_team"]}: {away.get("form_description", "")}')
    
else:
    print(f'❌ API Error: {response.status_code}')
