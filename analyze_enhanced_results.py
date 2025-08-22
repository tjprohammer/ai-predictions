import requests

response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
data = response.json()
games = data.get('games', [])

print('🏆 HIGH CONFIDENCE GAMES ANALYSIS')
print('=' * 45)

# Find games with confidence > 60
high_conf_games = [g for g in games if g.get('confidence', 0) > 60]
print(f'Games with >60% confidence: {len(high_conf_games)}')

for i, game in enumerate(high_conf_games):
    print(f'\n🔥 HIGH CONFIDENCE GAME {i+1}:')
    print(f'   {game.get("away_team")} @ {game.get("home_team")}')
    print(f'   Confidence: {game.get("confidence")}%')
    print(f'   Edge: {game.get("edge"):+.2f} runs')
    print(f'   Recommendation: {game.get("recommendation")}')
    print(f'   Confidence Level: {game.get("confidence_level")}')
    print(f'   Strong Pick: {"✅" if game.get("is_strong_pick") else "❌"}')
    print(f'   Premium Pick: {"⭐" if game.get("is_premium_pick") else "❌"}')
    
    if game.get('calibrated_predictions'):
        cal = game['calibrated_predictions']
        print(f'   📈 Calibrated: {cal.get("predicted_total")} runs, {cal.get("recommendation")}, {cal.get("edge"):+.2f} edge')
        if cal.get('team_adjustment'):
            print(f'   🏈 Team Adjustment: {cal.get("team_adjustment"):+.2f} runs')

print(f'\nSUMMARY:')
actionable = len([g for g in games if g.get('recommendation') != 'HOLD'])
strong_picks = len([g for g in games if g.get('is_strong_pick')])
premium_picks = len([g for g in games if g.get('is_premium_pick')])

print(f'✅ Total actionable games: {actionable}/{len(games)} ({actionable/len(games)*100:.1f}%)')
print(f'🔥 Strong picks: {strong_picks}')
print(f'⭐ Premium picks: {premium_picks}')

# Show team data integration working
print(f'\n🏈 TEAM DATA INTEGRATION TEST:')
sample_game = games[0] if games else None
if sample_game:
    print(f'Sample game: {sample_game.get("away_team")} @ {sample_game.get("home_team")}')
    if sample_game.get('ai_analysis'):
        ai = sample_game['ai_analysis']
        team_factors = [f for f in ai.get('primary_factors', []) + ai.get('supporting_factors', []) if 'recent' in f.lower() or 'form' in f.lower()]
        if team_factors:
            print(f'✅ Team form factors detected: {len(team_factors)}')
            for factor in team_factors[:2]:
                print(f'   • {factor}')
        else:
            print(f'⚠️ No team form factors detected - check team data integration')
