#!/usr/bin/env python3
"""
Daily Learning Pipeline
Automated system to update models each morning with previous day's results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning_system import ContinuousLearningSystem
from datetime import datetime, timedelta
import joblib
import numpy as np
import pandas as pd
import psycopg2

class DailyLearningPipeline:
    def __init__(self):
        self.learner = ContinuousLearningSystem()
        self.production_model_file = 'models/production_model.joblib'
        
    def update_daily_model(self, target_date=None):
        """Update model with results from previous day"""
        
        if target_date is None:
            # Use yesterday as default
            target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        print(f"🔄 DAILY MODEL UPDATE FOR {target_date}")
        print("=" * 50)
        
        # Perform learning update
        success = self.learner.daily_learning_update(target_date)
        
        if success:
            # Load the new model
            model_file = os.path.join(self.learner.models_dir, f"models_{target_date}.joblib")
            
            if os.path.exists(model_file):
                # Update production model with best performing model
                self.update_production_model(model_file, target_date)
                return True
        
        return False
    
    def update_production_model(self, new_model_file, date):
        """Update production model with best performing model"""
        
        print(f"\\n🚀 UPDATING PRODUCTION MODEL...")
        
        # Load new model data
        model_data = joblib.load(new_model_file)
        
        # Find best performing model
        test_perf = model_data.get('test_performance', {})
        if not test_perf:
            print("⚠️  No test performance data available")
            return
        
        # Choose model with lowest MAE
        best_model_name = min(test_perf.keys(), key=lambda x: test_perf[x]['mae'])
        best_mae = test_perf[best_model_name]['mae']
        
        print(f"Best model: {best_model_name} (MAE: {best_mae:.2f})")
        
        # Prepare production model package
        production_package = {
            'model': model_data['models'][best_model_name],
            'model_type': best_model_name,
            'feature_names': model_data['feature_names'],
            'training_date': date,
            'performance': test_perf[best_model_name],
            'feature_importance': model_data['feature_importance'].get(best_model_name, {}),
            'version': f"daily_{date}_{best_model_name}"
        }
        
        # Save production model
        joblib.dump(production_package, self.production_model_file)
        
        print(f"✅ Production model updated: {best_model_name}")
        print(f"📊 Performance: MAE={best_mae:.2f}")
        print(f"🗂️  Saved to: {self.production_model_file}")
    
    def predict_with_production_model(self, game_features):
        """Make prediction using current production model"""
        
        if not os.path.exists(self.production_model_file):
            print("❌ No production model available")
            return None
        
        # Load production model
        model_package = joblib.load(self.production_model_file)
        model = model_package['model']
        model_type = model_package['model_type']
        feature_names = model_package['feature_names']
        
        # Prepare features in correct order
        if len(game_features) != len(feature_names):
            print(f"❌ Feature mismatch: expected {len(feature_names)}, got {len(game_features)}")
            return None
        
        # Make prediction
        if model_type == 'linear':
            # Linear model needs scaling
            scaler = model['scaler']
            features_scaled = scaler.transform([game_features])
            prediction = model['model'].predict(features_scaled)[0]
        else:
            # Tree-based models
            prediction = model.predict([game_features])[0]
        
        return {
            'predicted_total': round(prediction, 1),
            'model_type': model_type,
            'model_version': model_package['version'],
            'feature_names': feature_names
        }
    
    def get_model_status(self):
        """Get current production model status"""
        
        if not os.path.exists(self.production_model_file):
            return {"status": "No production model available"}
        
        model_package = joblib.load(self.production_model_file)
        
        return {
            "status": "Active",
            "model_type": model_package['model_type'],
            "version": model_package['version'],
            "training_date": model_package['training_date'],
            "performance": model_package['performance'],
            "top_features": sorted(
                model_package['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5] if model_package['feature_importance'] else []
        }
    
    def test_production_model_on_today(self):
        """Test production model on today's games"""
        
        print("\\n🧪 TESTING PRODUCTION MODEL ON TODAY'S GAMES")
        print("=" * 50)
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Get today's games
        conn = self.learner.get_db_connection()
        
        query = """
        SELECT 
            game_id, home_team, away_team, total_runs,
            temperature, humidity, wind_speed, air_pressure,
            home_sp_season_era, away_sp_season_era,
            home_sp_whip, away_sp_whip,
            home_sp_days_rest, away_sp_days_rest,
            home_team_avg, away_team_avg,
            ballpark_run_factor, ballpark_hr_factor,
            umpire_ou_tendency, market_total,
            day_night, getaway_day, doubleheader,
            home_sp_hand, away_sp_hand
        FROM enhanced_games 
        WHERE date = %s
        AND total_runs IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn, params=[today])
        conn.close()
        
        if len(df) == 0:
            print(f"No completed games found for {today}")
            return
        
        print(f"Found {len(df)} completed games for {today}")
        
        # Test each game
        predictions = []
        errors = []
        
        for _, game in df.iterrows():
            # Prepare features (same order as training)
            features = [
                game['temperature'] or 70,
                game['humidity'] or 50,
                game['wind_speed'] or 5,
                game['air_pressure'] or 30.0,
                game['home_sp_season_era'] or 4.5,
                game['away_sp_season_era'] or 4.5,
                game['home_sp_whip'] or 1.3,
                game['away_sp_whip'] or 1.3,
                game['home_sp_days_rest'] or 4,
                game['away_sp_days_rest'] or 4,
                game['home_team_avg'] or 0.250,
                game['away_team_avg'] or 0.250,
                game['ballpark_run_factor'] or 1.0,
                game['ballpark_hr_factor'] or 1.0,
                game['umpire_ou_tendency'] or 0.0,
                1 if game['day_night'] == 'N' else 0,
                1 if game['getaway_day'] else 0,
                1 if game['doubleheader'] else 0,
                1 if game['home_sp_hand'] == 'L' and game['away_sp_hand'] == 'L' else 0,
                1 if game['home_sp_hand'] == 'R' and game['away_sp_hand'] == 'R' else 0,
                game['market_total'] or 8.5
            ]
            
            # Make prediction
            result = self.predict_with_production_model(features)
            
            if result:
                predicted = result['predicted_total']
                actual = game['total_runs']
                error = abs(predicted - actual)
                
                predictions.append({
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'predicted': predicted,
                    'actual': actual,
                    'error': error,
                    'model_type': result['model_type']
                })
                
                errors.append(error)
        
        if predictions:
            avg_error = np.mean(errors)
            print(f"\\n📊 PRODUCTION MODEL PERFORMANCE:")
            print(f"Average MAE: {avg_error:.2f} runs")
            print(f"Model type: {predictions[0]['model_type']}")
            
            print("\\n🎯 INDIVIDUAL PREDICTIONS:")
            for pred in predictions:
                status = "✅" if pred['error'] <= 2 else "⚠️" if pred['error'] <= 3 else "❌"
                print(f"{status} {pred['game']}: {pred['predicted']} (actual: {pred['actual']}, error: {pred['error']:.1f})")

def main():
    """Main execution"""
    
    pipeline = DailyLearningPipeline()
    
    print("🤖 DAILY LEARNING PIPELINE")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        # Manual date specification
        target_date = sys.argv[1]
        pipeline.update_daily_model(target_date)
    else:
        # Default: update with yesterday's results
        pipeline.update_daily_model()
    
    # Show model status
    print("\\n📊 PRODUCTION MODEL STATUS:")
    status = pipeline.get_model_status()
    
    for key, value in status.items():
        if key == "top_features":
            print(f"{key}:")
            for feature, importance in value:
                print(f"  - {feature}: {importance:.3f}")
        else:
            print(f"{key}: {value}")
    
    # Test on today's games if available
    pipeline.test_production_model_on_today()

if __name__ == "__main__":
    main()
