#!/usr/bin/env python3
"""
Test Enhanced API with Team Data Integration
"""
import requests
import json

def test_enhanced_team_integration():
    """Test the new team data integration features"""
    try:
        print("🚀 TESTING ENHANCED API WITH TEAM DATA INTEGRATION")
        print("=" * 60)
        
        response = requests.get('http://localhost:8000/api/comprehensive-games/2025-08-20')
        data = response.json()
        games = data.get('games', [])
        
        print(f"📊 API Response: {len(games)} games found")
        
        if games:
            # Test with first few games
            for i, game in enumerate(games[:3]):
                print(f"\n🎯 GAME {i+1}: {game.get('away_team', 'N/A')} @ {game.get('home_team', 'N/A')}")
                print("-" * 50)
                
                # Original prediction
                print(f"   Original Prediction: {game.get('predicted_total', 'N/A')} runs")
                print(f"   Market Total: {game.get('market_total', 'N/A')} runs")
                print(f"   Edge: {game.get('edge', 'N/A'):+.2f} runs" if game.get('edge') else "   Edge: N/A")
                print(f"   Confidence: {game.get('confidence', 'N/A')}%")
                print(f"   Recommendation: {game.get('recommendation', 'N/A')}")
                
                # Enhanced features
                if game.get('calibrated_predictions'):
                    cal = game['calibrated_predictions']
                    print(f"\n   📈 CALIBRATED WITH TEAM DATA:")
                    print(f"      Calibrated Total: {cal.get('predicted_total', 'N/A')} runs")
                    print(f"      Calibrated Edge: {cal.get('edge', 'N/A'):+.2f} runs" if cal.get('edge') else "      Calibrated Edge: N/A")
                    print(f"      Calibrated Confidence: {cal.get('confidence', 'N/A')}%")
                    print(f"      Calibrated Recommendation: {cal.get('recommendation', 'N/A')}")
                    if cal.get('team_adjustment'):
                        print(f"      Team Form Adjustment: {cal.get('team_adjustment', 0):+.2f} runs")
                    print(f"      Reason: {cal.get('calibration_reason', 'N/A')}")
                
                # Pick classifications
                print(f"\n   🏆 PICK CLASSIFICATION:")
                print(f"      Confidence Level: {game.get('confidence_level', 'N/A')}")
                print(f"      Strong Pick: {'✅' if game.get('is_strong_pick') else '❌'}")
                print(f"      Premium Pick: {'⭐' if game.get('is_premium_pick') else '❌'}")
                print(f"      High Confidence: {'🔥' if game.get('is_high_confidence') else '❌'}")
                
                # AI Analysis with team data
                if game.get('ai_analysis'):
                    ai = game['ai_analysis']
                    print(f"\n   🧠 AI ANALYSIS:")
                    print(f"      Summary: {ai.get('prediction_summary', 'N/A')}")
                    print(f"      Reasoning: {ai.get('recommendation_reasoning', 'N/A')}")
                    
                    if ai.get('primary_factors'):
                        print(f"      🔑 Primary Factors:")
                        for factor in ai['primary_factors'][:2]:  # Show first 2
                            print(f"         • {factor}")
                    
                    if ai.get('key_insights'):
                        print(f"      💡 Key Insights:")
                        for insight in ai['key_insights'][:2]:  # Show first 2
                            print(f"         • {insight}")
        
        # Summary of improvements
        print(f"\n📈 ENHANCEMENT SUMMARY:")
        actionable_games = len([g for g in games if g.get('recommendation', 'HOLD') != 'HOLD'])
        strong_picks = len([g for g in games if g.get('is_strong_pick')])
        premium_picks = len([g for g in games if g.get('is_premium_pick')])
        
        print(f"   Total Games: {len(games)}")
        print(f"   Actionable (Not HOLD): {actionable_games} ({actionable_games/len(games)*100:.1f}%)")
        print(f"   Strong Picks: {strong_picks}")
        print(f"   Premium Picks: {premium_picks}")
        print(f"\n🎯 SUCCESS: Team data integration working!")
        
    except Exception as e:
        print(f"❌ Error testing enhanced API: {e}")

if __name__ == "__main__":
    test_enhanced_team_integration()
