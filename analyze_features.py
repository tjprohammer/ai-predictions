import requests
import json

def analyze_current_features():
    response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
    data = response.json()
    games = data.get('games', [])

    print('🔍 AI ANALYSIS CONTENT INSPECTION:')
    print('=' * 50)

    if games and 'ai_analysis' in games[0]:
        ai_analysis = games[0]['ai_analysis']
        print('📊 AI Analysis Structure:')
        for key, value in ai_analysis.items():
            if isinstance(value, list):
                print(f'  📋 {key}: {len(value)} items')
                if value:
                    sample = value[0][:60] + '...' if len(value[0]) > 60 else value[0]
                    print(f'      Sample: {sample}')
            else:
                print(f'  📝 {key}: {value}')
        
        away_team = games[0]['away_team']
        home_team = games[0]['home_team']
        print(f'\n🎯 SAMPLE AI ANALYSIS FOR {away_team} @ {home_team}:')
        print(f'   Prediction Summary: {ai_analysis.get("prediction_summary", "N/A")}')
        print(f'   Recommendation Reasoning: {ai_analysis.get("recommendation_reasoning", "N/A")}')
        print(f'   Primary Factors: {len(ai_analysis.get("primary_factors", []))} items')
        print(f'   Supporting Factors: {len(ai_analysis.get("supporting_factors", []))} items')
        print(f'   Key Insights: {len(ai_analysis.get("key_insights", []))} items')
        print(f'   Risk Factors: {len(ai_analysis.get("risk_factors", []))} items')
    else:
        print('❌ No AI analysis found in games')

    print(f'\n📈 MISSING ENHANCEMENT OPPORTUNITIES:')
    missing_fields = ['calibrated_predictions', 'confidence_level', 'is_strong_pick', 'is_premium_pick', 'is_high_confidence']
    for field in missing_fields:
        if not any(field in game for game in games):
            print(f'   ❌ {field}: Not implemented')
        else:
            print(f'   ✅ {field}: Implemented')

    return games

if __name__ == "__main__":
    analyze_current_features()
