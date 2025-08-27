import requests

def test_enhanced_features():
    response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
    data = response.json()
    games = data.get('games', [])

    if games:
        game = games[0]
        print('🎯 ENHANCED FEATURES VALIDATION:')
        print('=' * 50)
        
        print(f'📊 Game: {game["away_team"]} @ {game["home_team"]}')
        
        # Check for new features
        new_features = ['calibrated_predictions', 'confidence_level', 'is_strong_pick', 'is_premium_pick', 'is_high_confidence']
        for feature in new_features:
            if feature in game:
                print(f'✅ {feature}: {game[feature]}')
            else:
                print(f'❌ {feature}: Missing')
        
        # Check enhanced AI analysis
        if 'ai_analysis' in game:
            ai = game['ai_analysis']
            print(f'\n🤖 ENHANCED AI ANALYSIS:')
            print(f'   Prediction Summary: {ai.get("prediction_summary", "N/A")}')
            print(f'   Confidence Level: {ai.get("confidence_level", "N/A")}')
            print(f'   Primary Factors: {len(ai.get("primary_factors", []))} items')
            print(f'   Supporting Factors: {len(ai.get("supporting_factors", []))} items')
            print(f'   Key Insights: {len(ai.get("key_insights", []))} items')
            print(f'   Risk Factors: {len(ai.get("risk_factors", []))} items')
            
            if ai.get('primary_factors'):
                print(f'   First Primary Factor: {ai["primary_factors"][0]}')
        
        # Check calibrated predictions
        if 'calibrated_predictions' in game:
            cal = game['calibrated_predictions']
            print(f'\n🔄 CALIBRATED PREDICTIONS:')
            print(f'   Original: {game.get("predicted_total", "N/A")} runs')
            print(f'   Calibrated: {cal.get("predicted_total", "N/A")} runs')
            print(f'   Calibration Applied: {cal.get("calibration_applied", False)}')
            print(f'   Calibrated Recommendation: {cal.get("recommendation", "N/A")}')
    else:
        print('❌ No games found')

if __name__ == "__main__":
    test_enhanced_features()
