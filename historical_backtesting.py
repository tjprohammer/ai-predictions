#!/usr/bin/env python3
"""
Historical Model Backtesting
=============================

Re-run our current model on historical games to:
1. See how curren    # Error analysis
    avg_prediction_error = sum(r['prediction_error'] for r in results) / total_games
    avg_market_error = sum(r['market_error'] for r in results) / total_games
    
    print(f"\n📈 ERROR ANALYSIS:")
    print(f"Avg Prediction Error: {avg_prediction_error:.2f} runs")
    print(f"Avg Market Error: {avg_market_error:.2f} runs")
    
    # Convert to float to avoid Decimal arithmetic issues
    error_diff = float(avg_prediction_error) - float(avg_market_error)
    print(f"Model vs Market: {'+' if error_diff > 0 else ''}{error_diff:.2f} runs")
    
    if error_diff < 0:
        print(f"   🎯 Your model is MORE ACCURATE than market by {abs(error_diff):.2f} runs!")
    elif error_diff < 0.5:
        print(f"   📊 Your model performs similarly to market")
    else:
        print(f"   📈 Market is more accurate by {error_diff:.2f} runs")would have performed in the past
2. Identify model improvements over time
3. Generate additional training data with current features
"""

import psycopg2
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

def connect_db():
    return psycopg2.connect(
        host='localhost',
        database='mlb',
        user='mlbuser', 
        password='mlbpass'
    )

def get_historical_games_for_backtest(start_date, end_date, limit=50):
    """Get historical games that have actual results but could use fresh predictions"""
    conn = connect_db()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT game_id, date, home_team, away_team, total_runs, market_total
    FROM enhanced_games
    WHERE date BETWEEN %s AND %s
    AND total_runs IS NOT NULL
    AND market_total IS NOT NULL
    ORDER BY date DESC
    LIMIT %s
    ''', (start_date, end_date, limit))
    
    games = cursor.fetchall()
    conn.close()
    
    return games

def backtest_current_model(start_date, end_date):
    """Run current model predictions on historical games"""
    print(f"🔄 BACKTESTING CURRENT MODEL: {start_date} to {end_date}")
    print("-" * 60)
    
    # Get historical games
    historical_games = get_historical_games_for_backtest(start_date, end_date, 100)
    
    print(f"Found {len(historical_games)} historical games to backtest")
    
    backtest_results = []
    
    for i, (game_id, date, home_team, away_team, actual_total, market_total) in enumerate(historical_games):
        print(f"\\rProcessing game {i+1}/{len(historical_games)}: {away_team} @ {home_team}", end="")
        
        try:
            # Make prediction using current API
            response = requests.get(
                f"http://localhost:8000/api/comprehensive-games/{date}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('games', [])
                
                # Find matching game
                matching_game = None
                for game in games:
                    if (game.get('home_team') == home_team and 
                        game.get('away_team') == away_team):
                        matching_game = game
                        break
                
                if matching_game:
                    current_prediction = matching_game.get('predicted_total')
                    current_confidence = matching_game.get('confidence')
                    current_recommendation = matching_game.get('recommendation')
                    
                    if current_prediction:
                        # Calculate performance metrics
                        prediction_error = abs(current_prediction - actual_total)
                        market_error = abs(market_total - actual_total)
                        beat_market = prediction_error < market_error
                        
                        # Direction accuracy
                        predicted_direction = 'OVER' if current_prediction > market_total else 'UNDER'
                        actual_direction = 'OVER' if actual_total > market_total else 'UNDER'
                        direction_correct = predicted_direction == actual_direction
                        
                        # Betting result
                        bet_won = None
                        if current_recommendation in ['OVER', 'UNDER']:
                            if current_recommendation == 'OVER':
                                bet_won = actual_total > market_total
                            else:
                                bet_won = actual_total < market_total
                        
                        backtest_results.append({
                            'game_id': game_id,
                            'date': date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'actual_total': actual_total,
                            'market_total': market_total,
                            'current_prediction': current_prediction,
                            'current_confidence': current_confidence,
                            'current_recommendation': current_recommendation,
                            'prediction_error': prediction_error,
                            'market_error': market_error,
                            'beat_market': beat_market,
                            'direction_correct': direction_correct,
                            'bet_won': bet_won
                        })
            
        except Exception as e:
            # Skip failed requests
            pass
    
    print(f"\\n✅ Completed backtest on {len(backtest_results)} games")
    
    return backtest_results

def analyze_backtest_results(results):
    """Analyze the backtest results"""
    if not results:
        print("❌ No backtest results to analyze")
        return
    
    print(f"\\n📊 BACKTEST ANALYSIS")
    print("-" * 40)
    
    # Overall accuracy
    total_games = len(results)
    direction_correct = sum(1 for r in results if r['direction_correct'])
    beat_market_count = sum(1 for r in results if r['beat_market'])
    
    print(f"Total Games Analyzed: {total_games}")
    print(f"Direction Accuracy: {direction_correct}/{total_games} ({direction_correct/total_games:.1%})")
    print(f"Beat Market: {beat_market_count}/{total_games} ({beat_market_count/total_games:.1%})")
    
    # Betting performance
    betting_games = [r for r in results if r['bet_won'] is not None]
    if betting_games:
        betting_wins = sum(1 for r in betting_games if r['bet_won'])
        betting_total = len(betting_games)
        
        print(f"\\n💰 BETTING PERFORMANCE:")
        print(f"Betting Record: {betting_wins}/{betting_total} ({betting_wins/betting_total:.1%})")
        
        # Calculate profit
        profit = (betting_wins * 90.91) - ((betting_total - betting_wins) * 100)
        print(f"Estimated Profit: ${profit:.0f} (@$100/bet)")
    
    # Error analysis - convert to float to avoid Decimal issues
    avg_prediction_error = float(sum(r['prediction_error'] for r in results)) / total_games
    avg_market_error = float(sum(r['market_error'] for r in results)) / total_games
    
    print(f"\\n📈 ERROR ANALYSIS:")
    print(f"Avg Prediction Error: {avg_prediction_error:.2f} runs")
    print(f"Avg Market Error: {avg_market_error:.2f} runs")
    print(f"Model vs Market: {'+' if avg_prediction_error < avg_market_error else ''}{avg_prediction_error - avg_market_error:.2f} runs")
    
    # Confidence analysis
    high_conf_games = [r for r in results if r['current_confidence'] and r['current_confidence'] >= 70]
    if high_conf_games:
        high_conf_correct = sum(1 for r in high_conf_games if r['direction_correct'])
        print(f"\\n🎯 HIGH CONFIDENCE GAMES (70%+):")
        print(f"Accuracy: {high_conf_correct}/{len(high_conf_games)} ({high_conf_correct/len(high_conf_games):.1%})")

def save_backtest_results(results, output_file="backtest_results.json"):
    """Save backtest results for further analysis"""
    if not results:
        return
    
    # Convert datetime objects to strings for JSON serialization
    for result in results:
        if isinstance(result['date'], datetime):
            result['date'] = result['date'].isoformat()
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n💾 Saved backtest results to: {output_file}")

def create_model_comparison_dataset():
    """Create a dataset comparing historical predictions vs current model"""
    print(f"\\n🔬 CREATING MODEL COMPARISON DATASET")
    print("-" * 50)
    
    conn = connect_db()
    cursor = conn.cursor()
    
    # Get games where we have both historical and potential current predictions
    cursor.execute('''
    SELECT 
        game_id,
        date,
        home_team,
        away_team,
        predicted_total as historical_prediction,
        market_total,
        total_runs as actual_total,
        confidence as historical_confidence,
        recommendation as historical_recommendation
    FROM enhanced_games
    WHERE predicted_total IS NOT NULL
    AND market_total IS NOT NULL  
    AND total_runs IS NOT NULL
    AND date >= %s
    AND date <= %s
    ORDER BY date DESC
    LIMIT 200
    ''', (datetime.now().date() - timedelta(days=30), datetime.now().date() - timedelta(days=1)))
    
    historical_data = cursor.fetchall()
    conn.close()
    
    print(f"Found {len(historical_data)} games with historical predictions")
    
    # This could be expanded to re-run current model on these games
    # and compare performance differences
    
    return historical_data

def main():
    """Run complete historical backtesting analysis"""
    print("📈 HISTORICAL MODEL BACKTESTING")
    print("=" * 60)
    
    # Backtest last 7 days
    end_date = datetime.now().date() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=7)  # Last week
    
    print(f"Backtesting period: {start_date} to {end_date}")
    
    # Run backtest
    results = backtest_current_model(start_date, end_date)
    
    # Analyze results
    analyze_backtest_results(results)
    
    # Save results
    save_backtest_results(results, "historical_backtest_results.json")
    
    # Create comparison dataset
    comparison_data = create_model_comparison_dataset()
    
    print(f"\\n✅ BACKTESTING COMPLETE")
    print(f"🎯 Analyzed {len(results)} games")
    print(f"📊 Historical comparison data: {len(comparison_data)} games")
    print(f"🚀 Use this data to identify model improvements!")

if __name__ == "__main__":
    main()
