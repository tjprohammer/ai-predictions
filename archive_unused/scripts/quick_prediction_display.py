#!/usr/bin/env python3
"""
Quick Prediction Results Display
===============================

Shows recent predictions vs actual outcomes with pitcher matchups
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, date, timedelta
from sqlalchemy import create_engine, text

def display_recent_predictions_vs_results():
    """Display recent predictions against actual results"""
    print("🎯 MLB Prediction Tracking - Recent Results")
    print("=" * 60)
    
    engine = create_engine('postgresql://mlbuser:mlbpass@localhost:5432/mlb', echo=False)
    
    # Get recent completed games
    with engine.begin() as conn:
        query = """
        SELECT 
            date, game_id, home_team, away_team, home_score, away_score, 
            total_runs, venue_name, temperature, weather_condition,
            home_sp_id, away_sp_id, home_sp_er, away_sp_er, 
            home_sp_ip, away_sp_ip
        FROM enhanced_games 
        WHERE total_runs IS NOT NULL 
            AND home_score IS NOT NULL 
            AND away_score IS NOT NULL
            AND date >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY date DESC
        LIMIT 20
        """
        
        completed_games = pd.read_sql(text(query), conn)
        
        if completed_games.empty:
            print("❌ No recent completed games found")
            return
        
        print(f"✅ Found {len(completed_games)} recent completed games")
        print()
        
        # Display results in a table format
        print("📊 RECENT GAME RESULTS & PITCHER MATCHUPS")
        print("-" * 80)
        print(f"{'Date':<12} {'Game':<25} {'Score':<10} {'Total':<6} {'Venue':<20}")
        print("-" * 80)
        
        for _, game in completed_games.iterrows():
            date_str = game['date'].strftime('%m/%d')
            game_str = f"{game['away_team'][:10]} @ {game['home_team'][:10]}"
            score_str = f"{int(game['away_score'])}-{int(game['home_score'])}"
            total_str = str(int(game['total_runs']))
            venue_str = game['venue_name'][:20] if game['venue_name'] else "Unknown"
            
            print(f"{date_str:<12} {game_str:<25} {score_str:<10} {total_str:<6} {venue_str:<20}")
        
        print("-" * 80)
        print(f"📈 Average Total Runs: {completed_games['total_runs'].mean():.1f}")
        print(f"🎯 Total Games Analyzed: {len(completed_games)}")
    
    # Load current predictions
    try:
        with open('S:/Projects/AI_Predictions/daily_predictions.json', 'r') as f:
            predictions_data = json.load(f)
        
        print("\\n📅 TODAY'S MODEL PREDICTIONS")
        print("-" * 60)
        
        if isinstance(predictions_data, list):
            predictions = predictions_data
        else:
            predictions = predictions_data.get('predictions', [])
        
        for pred in predictions[:10]:  # Show first 10
            home_team = pred.get('home_team', 'Unknown')[:15]
            away_team = pred.get('away_team', 'Unknown')[:15]
            predicted_total = pred.get('predicted_total', pred.get('model_prediction', 0))
            recommendation = pred.get('recommendation', 'UNKNOWN')
            confidence = pred.get('confidence', pred.get('model_confidence', 0))
            
            print(f"🏟️  {away_team} @ {home_team}")
            print(f"   Model Prediction: {predicted_total} runs | {recommendation} | Confidence: {confidence}%")
            
            # Show pitcher matchup if available
            home_pitcher = pred.get('home_pitcher', 'Unknown')
            away_pitcher = pred.get('away_pitcher', 'Unknown')
            if home_pitcher != 'Unknown' or away_pitcher != 'Unknown':
                print(f"   Pitchers: {away_pitcher} vs {home_pitcher}")
            print()
    
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
    
    # Load betting odds
    try:
        with open('S:/Projects/AI_Predictions/betting_odds_today.json', 'r') as f:
            betting_odds = json.load(f)
        
        print("\\n💰 TODAY'S MARKET TOTALS")
        print("-" * 40)
        
        for odds in betting_odds[:10]:  # Show first 10
            home_team = odds.get('home_team', 'Unknown')[:15]
            away_team = odds.get('away_team', 'Unknown')[:15]
            market_total = odds.get('total', odds.get('market_total', 8.5))
            over_odds = odds.get('over_odds', -110)
            under_odds = odds.get('under_odds', -110)
            
            print(f"🎲 {away_team} @ {home_team}")
            print(f"   Market Total: {market_total} | Over: {over_odds} | Under: {under_odds}")
            print()
    
    except Exception as e:
        print(f"❌ Error loading betting odds: {e}")

def check_enhanced_validation_results():
    """Check the enhanced validation results from the ML training"""
    print("\\n🤖 ENHANCED ML MODEL VALIDATION RESULTS")
    print("=" * 60)
    
    try:
        # Load validation results
        validation_df = pd.read_csv('S:/Projects/AI_Predictions/enhanced_validation_results.csv')
        
        print(f"✅ Validation Results: {len(validation_df)} games analyzed")
        print()
        
        # Show summary statistics
        avg_error = validation_df['prediction_error'].mean()
        max_error = validation_df['prediction_error'].max()
        min_error = validation_df['prediction_error'].min()
        
        print(f"📊 PREDICTION ACCURACY SUMMARY")
        print(f"   Average Error: {avg_error:.2f} runs")
        print(f"   Best Prediction: {min_error:.1f} runs error")
        print(f"   Worst Prediction: {max_error:.1f} runs error")
        print()
        
        # Show top 5 best predictions
        best_predictions = validation_df.nsmallest(5, 'prediction_error')
        print("🟢 TOP 5 BEST PREDICTIONS:")
        print("-" * 70)
        print(f"{'Date':<12} {'Game':<25} {'Predicted':<10} {'Actual':<8} {'Error':<8}")
        print("-" * 70)
        
        for _, row in best_predictions.iterrows():
            date_str = str(row['date'])[:10]
            game_str = f"{row['away_team'][:10]} @ {row['home_team'][:10]}"
            pred_str = str(row['predicted_total'])
            actual_str = str(row['actual_total'])
            error_str = str(row['prediction_error'])
            
            print(f"{date_str:<12} {game_str:<25} {pred_str:<10} {actual_str:<8} {error_str:<8}")
        
        # Show top 5 worst predictions
        worst_predictions = validation_df.nlargest(5, 'prediction_error')
        print("\\n🔴 TOP 5 WORST PREDICTIONS:")
        print("-" * 70)
        print(f"{'Date':<12} {'Game':<25} {'Predicted':<10} {'Actual':<8} {'Error':<8}")
        print("-" * 70)
        
        for _, row in worst_predictions.iterrows():
            date_str = str(row['date'])[:10]
            game_str = f"{row['away_team'][:10]} @ {row['home_team'][:10]}"
            pred_str = str(row['predicted_total'])
            actual_str = str(row['actual_total'])
            error_str = str(row['prediction_error'])
            
            print(f"{date_str:<12} {game_str:<25} {pred_str:<10} {actual_str:<8} {error_str:<8}")
        
    except FileNotFoundError:
        print("❌ Enhanced validation results not found. Run enhanced ML trainer first.")
    except Exception as e:
        print(f"❌ Error loading validation results: {e}")

def check_model_feature_importance():
    """Display the feature importance from the enhanced model"""
    print("\\n🧠 MODEL FEATURE IMPORTANCE")
    print("=" * 40)
    
    try:
        with open('S:/Projects/AI_Predictions/enhanced_validation_summary.json', 'r') as f:
            summary = json.load(f)
        
        print(f"📅 Model trained on: {summary.get('validation_date', 'Unknown')[:10]}")
        print(f"🎯 Total games: {summary.get('total_games_validated', 0)}")
        print(f"📊 Average error: {summary.get('average_prediction_error', 0):.2f} runs")
        print(f"🎲 R-squared: {summary.get('r_squared', 0):.3f}")
        print(f"📈 Recommendation accuracy: {summary.get('recommendation_accuracy', 0):.1f}%")
        
    except Exception as e:
        print(f"❌ Error loading model summary: {e}")

def main():
    """Main function to run all displays"""
    display_recent_predictions_vs_results()
    check_enhanced_validation_results()
    check_model_feature_importance()
    
    print("\\n🎉 PREDICTION TRACKING COMPLETE!")
    print("📁 Check the following files for detailed analysis:")
    print("   - enhanced_ml_validation_graphs.png (visualization graphs)")
    print("   - enhanced_validation_results.csv (detailed results)")
    print("   - enhanced_validation_summary.json (summary statistics)")

if __name__ == "__main__":
    main()
