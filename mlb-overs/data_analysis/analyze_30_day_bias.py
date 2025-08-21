#!/usr/bin/env python3
"""
Analyze bias corrections based on last 30 days of actual vs predicted performance
"""

import json
import psycopg2
import numpy as np
from datetime import datetime, timedelta

def analyze_30_day_bias():
    """Analyze bias corrections based on 30 days of actual vs predicted performance"""
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host='localhost',
            database='mlb',
            user='mlbuser', 
            password='mlbpass'
        )
        cursor = conn.cursor()
        
        # Get last 30 days of games with actual results
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=30)
        
        print(f"🔍 ANALYZING BIAS CORRECTIONS FOR LAST 30 DAYS")
        print(f"📅 Date Range: {start_date} to {end_date}")
        print("=" * 60)
        
        # Query for games with actual results (total_runs is not null)
        cursor.execute('''
        SELECT 
            date, home_team, away_team,
            predicted_total, market_total, total_runs,
            home_sp_season_era, away_sp_season_era,
            temperature, venue_name,
            confidence, recommendation
        FROM enhanced_games 
        WHERE date BETWEEN %s AND %s 
        AND total_runs IS NOT NULL
        AND predicted_total IS NOT NULL
        ORDER BY date DESC
        ''', (start_date, end_date))
        
        games = cursor.fetchall()
        
        if not games:
            print("❌ No completed games found in the last 30 days")
            return
            
        print(f"📊 Found {len(games)} completed games with predictions")
        print()
        
        # Analyze predictions vs actual results
        predictions = []
        actuals = []
        markets = []
        errors = []
        
        high_error_games = []
        
        for game in games:
            date, home, away, pred, market, actual, home_era, away_era, temp, venue, conf, rec = game
            
            pred_val = float(pred) if pred else None
            actual_val = float(actual) if actual else None
            market_val = float(market) if market else None
            
            if pred_val and actual_val:
                predictions.append(pred_val)
                actuals.append(actual_val)
                markets.append(market_val or 0)
                
                error = pred_val - actual_val
                errors.append(error)
                
                # Track high error games
                if abs(error) > 5.0:
                    high_error_games.append({
                        'date': date,
                        'matchup': f"{away} @ {home}",
                        'predicted': pred_val,
                        'actual': actual_val,
                        'error': error,
                        'market': market_val
                    })
        
        # Calculate bias statistics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        markets = np.array(markets)
        errors = np.array(errors)
        
        mean_error = np.mean(errors)  # Negative = under-predicting
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        mean_pred = np.mean(predictions)
        mean_actual = np.mean(actuals)
        mean_market = np.mean(markets)
        
        print(f"🎯 BIAS ANALYSIS RESULTS:")
        print(f"   Mean Prediction: {mean_pred:.2f} runs")
        print(f"   Mean Actual:     {mean_actual:.2f} runs")
        print(f"   Mean Market:     {mean_market:.2f} runs")
        print()
        print(f"📈 ERROR METRICS:")
        print(f"   Mean Error:      {mean_error:.2f} runs ({'UNDER' if mean_error < 0 else 'OVER'}-predicting)")
        print(f"   MAE:             {mae:.2f} runs")
        print(f"   RMSE:            {rmse:.2f} runs")
        print(f"   Bias Magnitude:  {abs(mean_error):.2f} runs")
        print()
        
        # Recommended bias correction
        recommended_correction = -mean_error  # Opposite of the bias
        
        print(f"💡 RECOMMENDED BIAS CORRECTION:")
        print(f"   Current Global Adjustment: +{recommended_correction:.2f} runs")
        print()
        
        # Show worst prediction errors
        if high_error_games:
            print(f"⚠️  GAMES WITH HIGH PREDICTION ERRORS (>5 runs):")
            for game in sorted(high_error_games, key=lambda x: abs(x['error']), reverse=True)[:10]:
                print(f"   {game['date']} | {game['matchup']:<30} | Pred: {game['predicted']:5.1f} | Actual: {game['actual']:5.1f} | Error: {game['error']:+5.1f}")
            print()
        
        # Analysis by scoring ranges
        low_scoring = [(p, a) for p, a in zip(predictions, actuals) if a <= 7]
        mid_scoring = [(p, a) for p, a in zip(predictions, actuals) if 7 < a <= 10]
        high_scoring = [(p, a) for p, a in zip(predictions, actuals) if a > 10]
        
        print(f"📊 BIAS BY SCORING RANGE:")
        for range_name, range_data in [("Low (≤7)", low_scoring), ("Mid (8-10)", mid_scoring), ("High (11+)", high_scoring)]:
            if range_data:
                range_preds = [p for p, a in range_data]
                range_actuals = [a for p, a in range_data]
                range_bias = np.mean(range_preds) - np.mean(range_actuals)
                print(f"   {range_name:<12}: {len(range_data):3d} games | Bias: {range_bias:+5.2f} runs")
        print()
        
        # Update bias corrections file
        corrections = {
            "global_adjustment": round(recommended_correction, 2),
            "scoring_range_adjustments": {
                "Low (≤7)": round(-np.mean([p - a for p, a in low_scoring]) if low_scoring else 0, 2),
                "Mid (8-10)": round(-np.mean([p - a for p, a in mid_scoring]) if mid_scoring else 0, 2),
                "High (11+)": round(-np.mean([p - a for p, a in high_scoring]) if high_scoring else 0, 2)
            },
            "confidence_adjustments": {},
            "temperature_adjustments": {
                "Hot (80+°F)": 0.2,
                "Mild (70-79°F)": 0.1
            },
            "venue_adjustments": {
                "COL": 0.3,
                "TEX": 0.2,
                "MIN": 0.15
            },
            "pitcher_quality_adjustments": {
                "Both ERA > 4.5": 0.3,
                "Both ERA < 3.5": -0.2
            },
            "day_of_week_adjustments": {
                "weekend": 0.1
            },
            "market_deviation_adjustments": {},
            "high_scoring_adjustments": {
                "high_scoring_teams": {
                    "teams": ["COL", "TEX", "MIN", "TOR", "LAA"],
                    "adjustment": 0.25
                }
            },
            "timestamp": datetime.now().isoformat(),
            "based_on_days": 30,
            "games_analyzed": len(games),
            "performance_metrics": {
                "mean_error": round(mean_error, 3),
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "mean_predicted": round(mean_pred, 2),
                "mean_actual": round(mean_actual, 2),
                "mean_market": round(mean_market, 2)
            },
            "note": f"30-day analysis: Model {('under' if mean_error < 0 else 'over')}-predicting by {abs(mean_error):.2f} runs on average"
        }
        
        # Save to both locations
        main_file = 's:\\Projects\\AI_Predictions\\model_bias_corrections.json'
        deployment_file = 's:\\Projects\\AI_Predictions\\mlb-overs\\deployment\\model_bias_corrections.json'
        
        for file_path in [main_file, deployment_file]:
            with open(file_path, 'w') as f:
                json.dump(corrections, f, indent=2)
        
        print(f"✅ BIAS CORRECTIONS UPDATED:")
        print(f"   📁 Main: {main_file}")
        print(f"   📁 Deployment: {deployment_file}")
        print(f"   🎯 New Global Adjustment: {corrections['global_adjustment']:+.2f} runs")
        print(f"   📈 Based on {len(games)} games over 30 days")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error analyzing bias: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_30_day_bias()
