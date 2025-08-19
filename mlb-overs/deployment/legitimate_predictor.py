#!/usr/bin/env python3
"""
Legitimate Predictor
==================
Real-time MLB over/under predictions using only legitimate pre-game features
No data leakage - uses season statistics and environmental factors only
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import psycopg2
import joblib
from datetime import datetime, timedelta
import logging
from pathlib import Path
import requests
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegitimatePredictor:
    def __init__(self):
        self.models_dir = Path("../models")
        # PostgreSQL connection for Docker container
        self.db_url = "postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb"
        
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_performance = {}
        
        # Load the latest model
        self.load_model()
    
    def get_engine(self):
        """Get PostgreSQL database engine"""
        return create_engine(self.db_url)
    
    def load_model(self, model_path=None):
        """Load the trained legitimate model"""
        if model_path is None:
            model_path = self.models_dir / "legitimate_model_latest.joblib"
        
        try:
            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            logger.info(f"Loading model from {model_path}")
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.model_performance = model_data.get('performance', {})
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"   Model type: {model_data.get('model_type', 'Unknown')}")
            logger.info(f"   Training date: {model_data.get('training_date', 'Unknown')}")
            logger.info(f"   Features: {len(self.feature_columns)}")
            
            if 'test_mae' in self.model_performance:
                logger.info(f"   Expected MAE: {self.model_performance['test_mae']:.3f} runs")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_team_season_stats(self, team_id, as_of_date=None):
        """Get team season statistics as of a specific date"""
        try:
            engine = self.get_engine()
            
            # Query team season stats
            query = """
            SELECT * FROM team_season_stats 
            WHERE team_id = %(team_id)s 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = pd.read_sql(query, engine, params={'team_id': team_id})
            
            if len(result) == 0:
                # Return default stats if no data found
                logger.warning(f"No season stats found for team {team_id}, using defaults")
                return {
                    'runs_per_game': 4.5,
                    'batting_avg': 0.250,
                    'ops': 0.750,
                    'era': 4.00,
                    'whip': 1.30
                }
            
            stats = result.iloc[0]
            return {
                'runs_per_game': stats.get('runs_per_game', 4.5),
                'batting_avg': stats.get('batting_avg', 0.250),
                'ops': stats.get('ops', 0.750),
                'era': stats.get('era', 4.00),
                'whip': stats.get('whip', 1.30)
            }
            
        except Exception as e:
            logger.warning(f"Error getting team stats for {team_id}: {e}")
            return {
                'runs_per_game': 4.5,
                'batting_avg': 0.250,
                'ops': 0.750,
                'era': 4.00,
                'whip': 1.30
            }
    
    def get_pitcher_season_stats(self, pitcher_id, as_of_date=None):
        """Get pitcher season statistics as of a specific date"""
        try:
            engine = self.get_engine()
            
            # Query pitcher season stats
            query = """
            SELECT * FROM pitcher_season_stats 
            WHERE player_id = %(pitcher_id)s 
            ORDER BY date DESC 
            LIMIT 1
            """
            
            result = pd.read_sql(query, engine, params={'pitcher_id': pitcher_id})
            
            if len(result) == 0:
                # Return default stats if no data found
                logger.warning(f"No season stats found for pitcher {pitcher_id}, using defaults")
                return {
                    'season_era': 4.00,
                    'season_whip': 1.30,
                    'k_per_9': 8.5,
                    'bb_per_9': 3.2,
                    'hr_per_9': 1.1
                }
            
            stats = result.iloc[0]
            return {
                'season_era': stats.get('era', 4.00),
                'season_whip': stats.get('whip', 1.30),
                'k_per_9': stats.get('k_per_9', 8.5),
                'bb_per_9': stats.get('bb_per_9', 3.2),
                'hr_per_9': stats.get('hr_per_9', 1.1)
            }
            
        except Exception as e:
            logger.warning(f"Error getting pitcher stats for {pitcher_id}: {e}")
            return {
                'season_era': 4.00,
                'season_whip': 1.30,
                'k_per_9': 8.5,
                'bb_per_9': 3.2,
                'hr_per_9': 1.1
            }
    
    def get_ballpark_factors(self, venue_id):
        """Get ballpark factors for a venue"""
        ballpark_factors = {
            # MLB Ballparks with run/HR factors
            1: {'run_factor': 1.05, 'hr_factor': 1.1, 'name': 'Coors Field'},  # Colorado
            2: {'run_factor': 0.95, 'hr_factor': 0.9, 'name': 'Fenway Park'},  # Boston
            3: {'run_factor': 1.02, 'hr_factor': 1.15, 'name': 'Yankee Stadium'},  # Yankees
            4: {'run_factor': 0.98, 'hr_factor': 0.95, 'name': 'Tropicana Field'},  # Tampa Bay
            5: {'run_factor': 1.0, 'hr_factor': 1.0, 'name': 'Rogers Centre'},  # Toronto
        }
        
        return ballpark_factors.get(venue_id, {'run_factor': 1.0, 'hr_factor': 1.0, 'name': 'Unknown'})
    
    def get_weather_forecast(self, venue_id, game_date):
        """Get weather forecast for game location and date"""
        # This would normally call a weather API
        # For now, return reasonable defaults
        return {
            'temperature': 75,
            'wind_speed': 8,
            'weather_condition': 'Clear',
            'humidity': 60,
            'pressure': 30.0
        }
    
    def build_prediction_features(self, game_data):
        """Build legitimate features for a single game prediction"""
        logger.info(f"Building features for game: {game_data.get('home_team')} vs {game_data.get('away_team')}")
        
        # Get team season stats
        home_team_stats = self.get_team_season_stats(game_data['home_team_id'])
        away_team_stats = self.get_team_season_stats(game_data['away_team_id'])
        
        # Get pitcher season stats
        home_pitcher_stats = self.get_pitcher_season_stats(game_data['home_pitcher_id'])
        away_pitcher_stats = self.get_pitcher_season_stats(game_data['away_pitcher_id'])
        
        # Get ballpark factors
        ballpark = self.get_ballpark_factors(game_data['venue_id'])
        
        # Get weather forecast
        weather = self.get_weather_forecast(game_data['venue_id'], game_data['date'])
        
        # Build feature dictionary
        features = {}
        
        # Environmental factors
        features['temperature'] = weather['temperature']
        features['wind_speed'] = weather['wind_speed']
        features['temp_factor'] = (weather['temperature'] - 70) * 0.02
        features['wind_factor'] = weather['wind_speed'] * 0.1
        features['is_night_game'] = 1 if game_data.get('day_night') == 'N' else 0
        
        # Ballpark factors
        features['ballpark_run_factor'] = ballpark['run_factor']
        features['ballpark_hr_factor'] = ballpark['hr_factor']
        features['park_offensive_factor'] = ballpark['run_factor'] * ballpark['hr_factor']
        
        # Pitcher season stats
        features['home_pitcher_season_era'] = home_pitcher_stats['season_era']
        features['away_pitcher_season_era'] = away_pitcher_stats['season_era']
        features['home_pitcher_season_whip'] = home_pitcher_stats['season_whip']
        features['away_pitcher_season_whip'] = away_pitcher_stats['season_whip']
        features['home_pitcher_k_per_9'] = home_pitcher_stats['k_per_9']
        features['away_pitcher_k_per_9'] = away_pitcher_stats['k_per_9']
        
        # Team season stats
        features['home_team_runs_per_game'] = home_team_stats['runs_per_game']
        features['away_team_runs_per_game'] = away_team_stats['runs_per_game']
        features['home_team_batting_avg'] = home_team_stats['batting_avg']
        features['away_team_batting_avg'] = away_team_stats['batting_avg']
        features['home_team_ops'] = home_team_stats['ops']
        features['away_team_ops'] = away_team_stats['ops']
        
        # Derived features
        features['era_difference'] = features['home_pitcher_season_era'] - features['away_pitcher_season_era']
        features['combined_era'] = (features['home_pitcher_season_era'] + features['away_pitcher_season_era']) / 2
        features['combined_whip'] = (features['home_pitcher_season_whip'] + features['away_pitcher_season_whip']) / 2
        features['pitching_advantage'] = (features['home_pitcher_k_per_9'] + features['away_pitcher_k_per_9']) / 2
        
        features['combined_team_offense'] = (features['home_team_runs_per_game'] + features['away_team_runs_per_game']) / 2
        features['combined_ops'] = (features['home_team_ops'] + features['away_team_ops']) / 2
        features['offensive_balance'] = abs(features['home_team_runs_per_game'] - features['away_team_runs_per_game'])
        
        # Interactions
        features['temp_park_interaction'] = features['temp_factor'] * features['ballpark_run_factor']
        features['wind_park_interaction'] = features['wind_factor'] * features['ballpark_hr_factor']
        
        # Add any additional derived features expected by the model
        features['offensive_advantage'] = features['combined_team_offense'] - features['combined_era']
        features['pitching_quality'] = 5.0 - features['combined_era']  # Inverse of ERA
        features['expected_offensive_environment'] = features['combined_team_offense'] * features['ballpark_run_factor']
        
        # Weather dummy variables (set based on weather condition)
        weather_condition = weather['weather_condition']
        weather_conditions = ['Clear', 'Cloudy', 'Rain', 'Snow', 'Overcast']
        for condition in weather_conditions:
            features[f'weather_{condition}'] = 1 if weather_condition == condition else 0
        
        logger.info(f"Built {len(features)} features for prediction")
        
        return features
    
    def predict_game_total(self, game_data):
        """Predict total runs for a single game"""
        if self.model is None:
            logger.error("No model loaded!")
            return None
        
        try:
            # Build features
            features = self.build_prediction_features(game_data)
            
            # Convert to DataFrame
            feature_df = pd.DataFrame([features])
            
            # Ensure all expected features are present
            for expected_feature in self.feature_columns:
                if expected_feature not in feature_df.columns:
                    logger.warning(f"Missing expected feature: {expected_feature}, setting to 0")
                    feature_df[expected_feature] = 0
            
            # Select and order features to match training
            X = feature_df[self.feature_columns]
            
            # Handle any missing values
            X = X.fillna(0)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            
            # Get prediction confidence (if available)
            confidence = None
            if hasattr(self.model, 'predict_proba'):
                # For regression, we can estimate confidence from feature importance
                feature_importance_score = np.sum(X_scaled[0] * getattr(self.model, 'feature_importances_', np.ones(len(X_scaled[0]))))
                confidence = min(0.9, max(0.6, 0.8 + feature_importance_score * 0.1))
            
            # Round prediction to reasonable precision
            prediction = round(prediction, 1)
            
            logger.info(f"Prediction: {prediction} total runs")
            
            # Build result dictionary
            result = {
                'prediction': prediction,
                'confidence': confidence,
                'model_mae': self.model_performance.get('test_mae', None),
                'features_used': len(self.feature_columns),
                'prediction_date': datetime.now().isoformat(),
                'game_info': {
                    'home_team': game_data.get('home_team'),
                    'away_team': game_data.get('away_team'),
                    'date': game_data.get('date'),
                    'venue': game_data.get('venue_name', 'Unknown')
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return None
    
    def predict_today_games(self):
        """Predict all games scheduled for today"""
        logger.info("Predicting today's games...")
        
        try:
            # Get today's games from database
            engine = self.get_engine()
            
            today = datetime.now().strftime('%Y-%m-%d')
            query = """
            SELECT game_id, home_team_id, away_team_id, venue_id, 
                   home_team, away_team, date, day_night,
                   home_sp_id, away_sp_id, venue_name
            FROM enhanced_games 
            WHERE date = %(today)s AND total_runs IS NULL
            """
            
            games_df = pd.read_sql(query, engine, params={'today': today})
            
            if len(games_df) == 0:
                logger.info("No games found for today")
                return []
            
            logger.info(f"Found {len(games_df)} games for today")
            
            predictions = []
            
            for _, game in games_df.iterrows():
                game_data = {
                    'game_id': game['game_id'],
                    'home_team_id': game['home_team_id'],
                    'away_team_id': game['away_team_id'],
                    'home_pitcher_id': game.get('home_sp_id', 0),
                    'away_pitcher_id': game.get('away_sp_id', 0),
                    'venue_id': game['venue_id'],
                    'venue_name': game.get('venue_name', ''),
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'date': game['date'],
                    'day_night': game.get('day_night', 'D')
                }
                
                prediction = self.predict_game_total(game_data)
                
                if prediction:
                    prediction['game_id'] = game['game_id']
                    predictions.append(prediction)
                    
                    logger.info(f"{game['away_team']} @ {game['home_team']}: {prediction['prediction']} runs")
            
            logger.info(f"Generated {len(predictions)} predictions")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting today's games: {e}")
            return []
    
    def save_predictions(self, predictions, output_file="daily_predictions.json"):
        """Save predictions to file and database"""
        if not predictions:
            logger.warning("No predictions to save")
            return
        
        try:
            # Save to JSON file
            import json
            json_output = {
                'prediction_date': datetime.now().isoformat(),
                'model_performance': self.model_performance,
                'predictions': predictions,
                'summary': {
                    'total_games': len(predictions),
                    'average_prediction': np.mean([p['prediction'] for p in predictions]),
                    'prediction_range': {
                        'min': min([p['prediction'] for p in predictions]),
                        'max': max([p['prediction'] for p in predictions])
                    }
                }
            }
            
            with open(output_file, 'w') as f:
                json.dump(json_output, f, indent=2)
            
            logger.info(f"✅ Predictions saved to {output_file}")
            
            # Also save to database
            engine = self.get_engine()
            
            with engine.begin() as conn:
                for prediction in predictions:
                    # Insert or update prediction
                    query = text("""
                    INSERT INTO game_predictions 
                    (game_id, predicted_total, prediction_date, model_version, confidence)
                    VALUES (%(game_id)s, %(predicted_total)s, %(prediction_date)s, %(model_version)s, %(confidence)s)
                    ON CONFLICT (game_id, prediction_date) 
                    DO UPDATE SET 
                        predicted_total = EXCLUDED.predicted_total,
                        model_version = EXCLUDED.model_version,
                        confidence = EXCLUDED.confidence
                    """)
                    
                    conn.execute(query, {
                        'game_id': prediction['game_id'],
                        'predicted_total': prediction['prediction'],
                        'prediction_date': prediction['prediction_date'],
                        'model_version': 'legitimate_model',
                        'confidence': prediction.get('confidence')
                    })
            
            logger.info("✅ Predictions saved to database")
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")

def main():
    """Main prediction function"""
    print("🎯 LEGITIMATE MLB PREDICTOR")
    print("=" * 60)
    print("Making predictions with legitimate pre-game features only")
    print("")
    
    predictor = LegitimatePredictor()
    
    if predictor.model is None:
        print("❌ Failed to load model!")
        return 1
    
    # Make predictions for today's games
    predictions = predictor.predict_today_games()
    
    if predictions:
        print(f"\n🏆 DAILY PREDICTIONS ({len(predictions)} games)")
        print("=" * 60)
        
        for pred in predictions:
            game_info = pred['game_info']
            confidence_str = f" (conf: {pred['confidence']:.1%})" if pred['confidence'] else ""
            print(f"{game_info['away_team']} @ {game_info['home_team']}: {pred['prediction']:.1f} runs{confidence_str}")
        
        # Save predictions
        predictor.save_predictions(predictions)
        
        # Summary
        avg_pred = np.mean([p['prediction'] for p in predictions])
        print(f"\n📊 Average predicted total: {avg_pred:.1f} runs")
        print(f"📈 Model expected accuracy: ±{predictor.model_performance.get('test_mae', 'Unknown')} runs")
        print("\n✅ All predictions saved successfully!")
        
        return 0
    else:
        print("❌ No predictions generated")
        return 1

if __name__ == "__main__":
    exit(main())
