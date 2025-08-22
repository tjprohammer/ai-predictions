import requests

response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
data = response.json()
games = data.get('games', [])

print('🔍 API CONFIDENCE VALUE DEBUG')
print('=' * 40)

if games:
    game = games[0]  # First game
    print(f'Game: {game.get("away_team")} @ {game.get("home_team")}')
    print(f'Raw confidence from API: {game.get("confidence")} (type: {type(game.get("confidence"))})')
    
    # Look for games with higher confidence
    high_conf_games = [g for g in games if g.get('confidence', 0) > 50]
    print(f'Games with confidence > 50: {len(high_conf_games)}')
    
    if high_conf_games:
        hc_game = high_conf_games[0]
        print(f'High conf game: {hc_game.get("away_team")} @ {hc_game.get("home_team")}')
        print(f'Confidence: {hc_game.get("confidence")}')
        print(f'Edge: {hc_game.get("edge")}')
        print(f'Recommendation: {hc_game.get("recommendation")}')
        print(f'Confidence Level: {hc_game.get("confidence_level")}')
        print(f'Is Strong Pick: {hc_game.get("is_strong_pick")}')
    
    print(f'\nAll game confidences:')
    for i, g in enumerate(games[:5]):
        print(f'  {i+1}. {g.get("away_team")} @ {g.get("home_team")}: {g.get("confidence")}% conf, {g.get("edge"):+.2f} edge, {g.get("recommendation")}')
