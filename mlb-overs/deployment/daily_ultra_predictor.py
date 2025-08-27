#!/usr/bin/env python3
"""
📅 DAILY ULTRA PREDICTION SYSTEM
===============================
🎯 Integrate ultra 80% system into daily workflow
⚡ Automated predictions for today's games
===============================
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, date
from pathlib import Path
import os
import sys

# Add the training directory to path
sys.path.append(str(Path(__file__).parent / 'mlb-overs' / 'training'))

def predict_today_games():
    """Generate ultra predictions for today's MLB games"""
    
    print("🎯 ULTRA DAILY PREDICTION SYSTEM")
    print("=" * 50)
    
    today = date.today()
    print(f"📅 Generating predictions for: {today}")
    
    try:
        # Import the ultra system
        from ultra_80_percent_system import (
            get_historical_season_data, create_ultra_features, 
            Ultra80PercentSystem, get_engine
        )
        
        # Load historical data for training
        print("📊 Loading training data...")
        df, dates = get_historical_season_data()
        
        # Initialize and train ultra system
        print("🚀 Initializing ultra system...")
        ultra_system = Ultra80PercentSystem()
        ultra_system.create_ultra_ensemble()
        
        # Prepare training data (last 200 games)
        recent_data = df.tail(200)
        train_X = []
        train_y = []
        
        print("🧠 Training ultra models...")
        for _, game in recent_data.iterrows():
            features, _ = create_ultra_features(df, game['date'], game, window_size=30)
            if features is not None:
                train_X.append(features)
                train_y.append(game['total_runs'])
        
        if len(train_X) < 50:
            print("❌ Insufficient training data")
            return []
        
        # Train the system
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        ultra_system.train_ultra_system(train_X, train_y)
        
        # Get today's games from API or database
        todays_games = get_todays_games()
        
        if not todays_games:
            print("📭 No games found for today")
            return []
        
        # Generate predictions
        predictions = []
        print(f"🎯 Making predictions for {len(todays_games)} games...")
        
        for game in todays_games:
            try:
                # Create features for today's game
                features, _ = create_ultra_features(df, today, game, window_size=30)
                
                if features is not None:
                    # Make prediction
                    prediction = ultra_system.predict_ultra(features)
                    
                    # Calculate confidence and recommendation
                    market_total = float(game.get('market_total', 9.0))
                    recommendation = "OVER" if prediction > market_total else "UNDER"
                    confidence = abs(prediction - market_total) / market_total
                    
                    # Edge calculation (how much above/below market)
                    edge = prediction - market_total
                    
                    pred_result = {
                        'date': today.isoformat(),
                        'game': f"{game['away_team']} @ {game['home_team']}",
                        'away_team': game['away_team'],
                        'home_team': game['home_team'],
                        'market_total': market_total,
                        'predicted_total': round(prediction, 1),
                        'recommendation': recommendation,
                        'edge': round(edge, 1),
                        'confidence': round(confidence, 3),
                        'model': 'ULTRA_80_PERCENT',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    predictions.append(pred_result)
                    
                    print(f"   {game['away_team']} @ {game['home_team']}: "
                          f"Pred {prediction:.1f} ({recommendation}) vs Market {market_total} "
                          f"(Edge: {edge:+.1f}, Conf: {confidence:.2f})")
                
            except Exception as e:
                print(f"   ⚠️ Error predicting {game.get('away_team', 'Unknown')} @ {game.get('home_team', 'Unknown')}: {e}")
        
        # Save predictions
        save_daily_predictions(predictions)
        
        return predictions
        
    except Exception as e:
        print(f"❌ Error in daily prediction system: {e}")
        return []

def get_todays_games():
    """Get today's games from database or API"""
    
    try:
        from ultra_80_percent_system import get_engine
        
        engine = get_engine()
        today = date.today()
        
        # Try to get from enhanced_games first
        todays_query = """
            SELECT DISTINCT away_team, home_team, market_total,
                   home_sp_name, away_sp_name, temperature, wind_speed,
                   ballpark_run_factor, ballpark_hr_factor, day_night
            FROM enhanced_games 
            WHERE date = %s
            AND market_total IS NOT NULL
        """
        
        df = pd.read_sql(todays_query, engine, params=[today])
        
        if len(df) > 0:
            return df.to_dict('records')
        
        # If no games in enhanced_games, check for scheduled games
        scheduled_query = """
            SELECT away_team, home_team, 
                   9.0 as market_total,  -- Default if not available
                   'TBD' as home_sp_name, 'TBD' as away_sp_name,
                   72.0 as temperature, 5.0 as wind_speed,
                   1.0 as ballpark_run_factor, 1.0 as ballpark_hr_factor,
                   'Night' as day_night
            FROM games 
            WHERE date = %s
        """
        
        df_scheduled = pd.read_sql(scheduled_query, engine, params=[today])
        return df_scheduled.to_dict('records')
        
    except Exception as e:
        print(f"⚠️ Error getting today's games: {e}")
        # Fallback: return sample data structure
        return []

def save_daily_predictions(predictions):
    """Save predictions to JSON file"""
    
    if not predictions:
        return
    
    # Create predictions directory if it doesn't exist
    pred_dir = Path(__file__).parent / 'daily_predictions'
    pred_dir.mkdir(exist_ok=True)
    
    # Save with timestamp
    today = date.today()
    filename = f"ultra_predictions_{today.isoformat()}.json"
    filepath = pred_dir / filename
    
    # Add summary statistics
    total_games = len(predictions)
    over_count = sum(1 for p in predictions if p['recommendation'] == 'OVER')
    under_count = total_games - over_count
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    output = {
        'date': today.isoformat(),
        'model': 'ULTRA_80_PERCENT_SYSTEM',
        'summary': {
            'total_games': total_games,
            'over_recommendations': over_count,
            'under_recommendations': under_count,
            'average_confidence': round(avg_confidence, 3),
            'generated_at': datetime.now().isoformat()
        },
        'predictions': predictions
    }
    
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"💾 Predictions saved to: {filepath}")
    
    # Also save to latest.json for easy access
    latest_path = pred_dir / 'latest.json'
    with open(latest_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"📊 SUMMARY: {total_games} games, {over_count} OVER, {under_count} UNDER")
    print(f"📈 Average confidence: {avg_confidence:.3f}")

def create_prediction_report(predictions):
    """Create formatted prediction report"""
    
    if not predictions:
        return "No predictions available for today."
    
    report = []
    report.append("🏆 ULTRA 80% DAILY PREDICTIONS")
    report.append("=" * 50)
    report.append(f"📅 Date: {predictions[0]['date']}")
    report.append(f"🎯 Total Games: {len(predictions)}")
    report.append("")
    
    # Sort by confidence (highest first)
    sorted_preds = sorted(predictions, key=lambda x: x['confidence'], reverse=True)
    
    for i, pred in enumerate(sorted_preds, 1):
        confidence_emoji = "🔥" if pred['confidence'] > 0.15 else "💪" if pred['confidence'] > 0.10 else "📊"
        edge_emoji = "🚀" if abs(pred['edge']) > 1.0 else "⚡" if abs(pred['edge']) > 0.5 else "📈"
        
        report.append(f"{i:2d}. {pred['game']}")
        report.append(f"     Market: {pred['market_total']} | Prediction: {pred['predicted_total']}")
        report.append(f"     {confidence_emoji} {pred['recommendation']} | Edge: {pred['edge']:+.1f} | Conf: {pred['confidence']:.3f} {edge_emoji}")
        report.append("")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Generate today's predictions
    predictions = predict_today_games()
    
    if predictions:
        # Print report
        report = create_prediction_report(predictions)
        print("\n" + report)
        
        # Send to Slack/Discord if configured
        # send_predictions_to_slack(predictions)
    else:
        print("❌ No predictions generated for today")
