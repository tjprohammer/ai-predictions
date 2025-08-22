#!/usr/bin/env python3
"""
Today's Game Analysis - Check how our predictions performed
==========================================================
Quick analysis of today's completed games vs our predictions
"""

import requests
import json
from datetime import datetime

def analyze_todays_games():
    """Analyze today's game predictions vs actual results"""
    
    today = datetime.now().strftime('%Y-%m-%d')
    print(f"🎯 ANALYZING TODAY'S PREDICTIONS - {today}")
    print("=" * 55)
    
    try:
        # Get today's games from API
        response = requests.get(f"http://localhost:8000/api/comprehensive-games/{today}")
        data = response.json()
        
        if 'games' not in data:
            print("❌ No games data found")
            return
            
        games = data['games']
        completed_games = [g for g in games if g.get('home_score') is not None and g.get('away_score') is not None]
        
        if not completed_games:
            print("⏳ No completed games found yet today")
            return
        
        print(f"✅ Found {len(completed_games)} completed games")
        print()
        
        # Analyze each game
        total_prediction_error = 0
        total_market_error = 0
        games_with_market = 0
        perfect_predictions = 0
        beat_market_count = 0
        
        results = []
        
        for game in completed_games:
            home_score = game.get('home_score', 0)
            away_score = game.get('away_score', 0)
            actual_total = home_score + away_score
            
            predicted_total = game.get('predicted_total', 0)
            market_total = game.get('market_total')
            
            prediction_error = abs(predicted_total - actual_total)
            total_prediction_error += prediction_error
            
            market_error = None
            if market_total:
                market_error = abs(market_total - actual_total)
                total_market_error += market_error
                games_with_market += 1
                
                # Check if we beat the market
                if prediction_error < market_error:
                    beat_market_count += 1
            
            # Check for perfect predictions (within 0.5 runs)
            if prediction_error <= 0.5:
                perfect_predictions += 1
            
            results.append({
                'matchup': f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}",
                'predicted': predicted_total,
                'market': market_total,
                'actual': actual_total,
                'pred_error': prediction_error,
                'market_error': market_error,
                'beat_market': market_error and prediction_error < market_error,
                'confidence': game.get('confidence', 0),
                'recommendation': game.get('recommendation', 'HOLD')
            })
        
        # Sort by prediction accuracy (best first)
        results.sort(key=lambda x: x['pred_error'])
        
        # Print summary
        avg_prediction_error = total_prediction_error / len(completed_games)
        avg_market_error = total_market_error / games_with_market if games_with_market > 0 else 0
        
        print("📊 PERFORMANCE SUMMARY:")
        print(f"  • Games Analyzed: {len(completed_games)}")
        print(f"  • Average Prediction Error: {avg_prediction_error:.2f} runs")
        if games_with_market > 0:
            print(f"  • Average Market Error: {avg_market_error:.2f} runs")
            print(f"  • Beat Market: {beat_market_count}/{games_with_market} games ({beat_market_count/games_with_market:.1%})")
        print(f"  • Perfect Predictions (≤0.5): {perfect_predictions} ({perfect_predictions/len(completed_games):.1%})")
        print(f"  • Accuracy within 1 run: {sum(1 for r in results if r['pred_error'] <= 1)}/{len(completed_games)} ({sum(1 for r in results if r['pred_error'] <= 1)/len(completed_games):.1%})")
        print()
        
        # Show detailed results
        print("🎯 DETAILED GAME RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(results):
            status = "✅" if result['pred_error'] <= 1 else "⚠️" if result['pred_error'] <= 2 else "❌"
            market_info = f"Market: {result['market']:.1f} (±{result['market_error']:.1f})" if result['market'] else "No market data"
            beat_market = " 🏆 BEAT MARKET" if result.get('beat_market') else ""
            
            print(f"{status} {result['matchup']}")
            print(f"   Predicted: {result['predicted']:.1f} | Actual: {result['actual']:.0f} | Error: ±{result['pred_error']:.1f}")
            print(f"   {market_info}{beat_market}")
            print(f"   Confidence: {result['confidence']:.0f}% | Rec: {result['recommendation']}")
            print()
        
        # Show best and worst predictions
        if results:
            print("🏆 BEST PREDICTION:")
            best = results[0]
            print(f"   {best['matchup']} - Predicted: {best['predicted']:.1f}, Actual: {best['actual']}, Error: ±{best['pred_error']:.1f}")
            
            print("💥 WORST PREDICTION:")
            worst = results[-1]
            print(f"   {worst['matchup']} - Predicted: {worst['predicted']:.1f}, Actual: {worst['actual']}, Error: ±{worst['pred_error']:.1f}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error connecting to API: {e}")
    except Exception as e:
        print(f"❌ Error analyzing data: {e}")

if __name__ == "__main__":
    analyze_todays_games()
