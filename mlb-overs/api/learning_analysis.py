"""
Learning Model Analysis API
Apply current learning model to historical games to measure improvement
"""

import pandas as pd
import psycopg2
from datetime import datetime, timedelta
import joblib
import numpy as np
from typing import Dict, List, Any, Optional
import os

class LearningModelAnalyzer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
    def get_historical_games(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical games with all features needed for prediction"""
        
        conn = psycopg2.connect(**self.db_config)
        
        query = """
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            total_runs as actual_total,
            predicted_total as original_prediction,
            market_total,
            confidence as original_confidence,
            edge as original_edge,
            recommendation as original_recommendation,
            -- Weather features
            temperature,
            humidity,
            wind_speed,
            wind_direction_deg,
            air_pressure,
            -- Pitching features
            home_sp_season_era,
            away_sp_season_era,
            home_sp_whip,
            away_sp_whip,
            home_sp_days_rest,
            away_sp_days_rest,
            home_sp_hand,
            away_sp_hand,
            -- Team features  
            home_team_avg,
            away_team_avg,
            -- Ballpark features
            ballpark_run_factor,
            ballpark_hr_factor,
            -- Umpire features
            umpire_ou_tendency,
            -- Game context
            series_game,
            getaway_day,
            doubleheader,
            day_after_night,
            day_night
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND total_runs IS NOT NULL
        AND predicted_total IS NOT NULL
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        return df
        
    def load_current_learning_model(self) -> Optional[Any]:
        """Load the most recent learning model"""
        
        # Look for learning models in the models directory
        model_dirs = [
            'models',
            '../models', 
            'mlb-overs/models',
            'models_backup'
        ]
        
        learning_model = None
        learning_model_path = None
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                # Look for learning model files
                for filename in os.listdir(model_dir):
                    if 'learning' in filename.lower() and filename.endswith('.joblib'):
                        try:
                            model_path = os.path.join(model_dir, filename)
                            loaded_obj = joblib.load(model_path)
                            
                            # Handle different model formats
                            if hasattr(loaded_obj, 'predict'):
                                learning_model = loaded_obj
                                learning_model_path = model_path
                                print(f"Loaded learning model from: {model_path}")
                                break
                            elif isinstance(loaded_obj, (list, tuple)) and len(loaded_obj) > 0:
                                # Sometimes models are saved as lists/tuples
                                for item in loaded_obj:
                                    if hasattr(item, 'predict'):
                                        learning_model = item
                                        learning_model_path = model_path
                                        print(f"Loaded learning model (from list) from: {model_path}")
                                        break
                                if learning_model:
                                    break
                        except Exception as e:
                            print(f"Failed to load {model_path}: {e}")
                            continue
                            
        if learning_model is None:
            print("No learning model found, will use enhanced model as baseline")
            # Try to load the enhanced model as a fallback
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for filename in os.listdir(model_dir):
                        if 'enhanced' in filename.lower() and filename.endswith('.joblib'):
                            try:
                                model_path = os.path.join(model_dir, filename)
                                loaded_obj = joblib.load(model_path)
                                
                                # Handle different formats for enhanced model too
                                if hasattr(loaded_obj, 'predict'):
                                    learning_model = loaded_obj
                                    learning_model_path = model_path
                                    print(f"Loaded enhanced model as baseline: {model_path}")
                                    break
                                elif isinstance(loaded_obj, (list, tuple)) and len(loaded_obj) > 0:
                                    for item in loaded_obj:
                                        if hasattr(item, 'predict'):
                                            learning_model = item
                                            learning_model_path = model_path
                                            print(f"Loaded enhanced model (from list) as baseline: {model_path}")
                                            break
                                    if learning_model:
                                        break
                            except Exception as e:
                                continue
                                
        return learning_model, learning_model_path
        
    def prepare_features_for_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features in the format expected by the learning model"""
        
        # Create feature dataframe with the same structure as training
        feature_columns = [
            # Weather features
            'temperature', 'humidity', 'wind_speed', 'wind_direction_deg', 'air_pressure',
            # Pitching features
            'home_sp_season_era', 'away_sp_season_era', 'home_sp_whip', 'away_sp_whip',
            'home_sp_days_rest', 'away_sp_days_rest',
            # Team features
            'home_team_avg', 'away_team_avg',
            # Ballpark features
            'ballpark_run_factor', 'ballpark_hr_factor',
            # Umpire features
            'umpire_ou_tendency',
            # Game context features
            'series_game', 'getaway_day', 'doubleheader', 'day_after_night'
        ]
        
        # Handle categorical features
        df_features = df.copy()
        
        # Convert boolean columns to numeric
        bool_columns = ['getaway_day', 'doubleheader', 'day_after_night']
        for col in bool_columns:
            if col in df_features.columns:
                df_features[col] = df_features[col].astype(int)
                
        # Convert hand (L/R) to numeric
        if 'home_sp_hand' in df_features.columns:
            df_features['home_sp_hand_R'] = (df_features['home_sp_hand'] == 'R').astype(int)
        if 'away_sp_hand' in df_features.columns:
            df_features['away_sp_hand_R'] = (df_features['away_sp_hand'] == 'R').astype(int)
            
        # Convert day_night to numeric
        if 'day_night' in df_features.columns:
            df_features['day_night_D'] = (df_features['day_night'] == 'D').astype(int)
            
        # Select and order features
        final_features = []
        for col in feature_columns:
            if col in df_features.columns:
                final_features.append(col)
                
        # Add encoded categorical features
        if 'home_sp_hand_R' in df_features.columns:
            final_features.append('home_sp_hand_R')
        if 'away_sp_hand_R' in df_features.columns:
            final_features.append('away_sp_hand_R')
        if 'day_night_D' in df_features.columns:
            final_features.append('day_night_D')
            
        # Fill missing values with reasonable defaults
        feature_df = df_features[final_features].fillna({
            'temperature': 70,
            'humidity': 50,
            'wind_speed': 5,
            'wind_direction_deg': 180,
            'air_pressure': 30.0,
            'home_sp_season_era': 4.0,
            'away_sp_season_era': 4.0,
            'home_sp_whip': 1.3,
            'away_sp_whip': 1.3,
            'home_sp_days_rest': 4,
            'away_sp_days_rest': 4,
            'home_team_avg': 0.250,
            'away_team_avg': 0.250,
            'ballpark_run_factor': 1.0,
            'ballpark_hr_factor': 1.0,
            'umpire_ou_tendency': 0.0,
            'series_game': 1,
            'getaway_day': 0,
            'doubleheader': 0,
            'day_after_night': 0,
            'home_sp_hand_R': 1,
            'away_sp_hand_R': 1,
            'day_night_D': 1
        })
        
        return feature_df
        
    def apply_learning_model_to_history(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Apply current learning model to historical games and compare performance"""
        
        print(f"Analyzing learning model performance on historical data: {start_date} to {end_date}")
        
        # Get historical games
        historical_games = self.get_historical_games(start_date, end_date)
        print(f"Retrieved {len(historical_games)} historical games")
        
        if len(historical_games) == 0:
            return {"error": "No historical games found for the specified date range"}
            
        # Load learning model
        learning_model, model_path = self.load_current_learning_model()
        if learning_model is None:
            return {"error": "No learning model available for analysis"}
            
        # Prepare features
        try:
            features = self.prepare_features_for_prediction(historical_games)
            print(f"Prepared features with shape: {features.shape}")
            print(f"Feature columns: {list(features.columns)}")
            
            # Make learning model predictions
            learning_predictions = learning_model.predict(features)
            
            # Calculate learning model confidence (if available)
            learning_confidence = None
            try:
                if hasattr(learning_model, 'predict_proba'):
                    # For classification models, use max probability as confidence
                    probabilities = learning_model.predict_proba(features)
                    learning_confidence = np.max(probabilities, axis=1) * 100
                elif hasattr(learning_model, 'score'):
                    # For regression models, use a different confidence measure
                    # This is a simplified approach - could be improved
                    residuals = np.abs(learning_predictions - historical_games['actual_total'])
                    learning_confidence = np.maximum(0, 100 - residuals * 10)
            except Exception as e:
                print(f"Could not calculate learning model confidence: {e}")
                learning_confidence = np.full(len(learning_predictions), 50.0)  # Default confidence
                
        except Exception as e:
            return {"error": f"Failed to apply learning model: {str(e)}"}
            
        # Compare performance
        results = []
        for i, row in historical_games.iterrows():
            original_pred = row['original_prediction']
            learning_pred = learning_predictions[i]
            actual = row['actual_total']
            market = row['market_total']
            
            # Calculate errors
            original_error = abs(original_pred - actual)
            learning_error = abs(learning_pred - actual)
            
            # Calculate which prediction was better
            original_better = original_error < learning_error
            learning_better = learning_error < original_error
            tie = abs(original_error - learning_error) < 0.1
            
            # Calculate betting performance
            original_rec = row['original_recommendation']
            learning_rec = 'OVER' if learning_pred > market else 'UNDER'
            
            original_bet_won = False
            learning_bet_won = False
            
            if original_rec == 'OVER':
                original_bet_won = actual > market
            elif original_rec == 'UNDER':
                original_bet_won = actual < market
                
            if learning_rec == 'OVER':
                learning_bet_won = actual > market
            elif learning_rec == 'UNDER':
                learning_bet_won = actual < market
                
            result = {
                'game_id': row['game_id'],
                'date': row['date'],
                'game': f"{row['away_team']} @ {row['home_team']}",
                'actual_total': actual,
                'market_total': market,
                'original_prediction': original_pred,
                'learning_prediction': learning_pred,
                'original_error': original_error,
                'learning_error': learning_error,
                'original_confidence': row['original_confidence'],
                'learning_confidence': learning_confidence[i] if learning_confidence is not None else 50.0,
                'prediction_improvement': learning_error < original_error,
                'original_recommendation': original_rec,
                'learning_recommendation': learning_rec,
                'original_bet_won': original_bet_won,
                'learning_bet_won': learning_bet_won,
                'betting_improvement': learning_bet_won and not original_bet_won
            }
            results.append(result)
            
        # Calculate summary statistics
        total_games = len(results)
        original_avg_error = np.mean([r['original_error'] for r in results])
        learning_avg_error = np.mean([r['learning_error'] for r in results])
        
        original_bet_wins = sum([r['original_bet_won'] for r in results])
        learning_bet_wins = sum([r['learning_bet_won'] for r in results])
        
        improved_predictions = sum([r['prediction_improvement'] for r in results])
        improved_betting = sum([r['betting_improvement'] for r in results])
        
        summary = {
            'model_path': model_path,
            'analysis_period': f"{start_date} to {end_date}",
            'total_games': total_games,
            'original_model': {
                'avg_error': original_avg_error,
                'bet_win_rate': original_bet_wins / total_games,
                'bet_wins': original_bet_wins
            },
            'learning_model': {
                'avg_error': learning_avg_error,
                'bet_win_rate': learning_bet_wins / total_games,
                'bet_wins': learning_bet_wins
            },
            'improvement': {
                'error_reduction': original_avg_error - learning_avg_error,
                'error_improvement_rate': improved_predictions / total_games,
                'betting_improvement_rate': improved_betting / total_games,
                'additional_bet_wins': learning_bet_wins - original_bet_wins
            }
        }
        
        return {
            'summary': summary,
            'game_by_game': results
        }

# Example usage function
def analyze_learning_improvement(days_back: int = 30) -> Dict[str, Any]:
    """Analyze learning model improvement over the last N days"""
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    analyzer = LearningModelAnalyzer()
    return analyzer.apply_learning_model_to_history(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

if __name__ == "__main__":
    # Test the analyzer
    result = analyze_learning_improvement(14)  # Last 2 weeks
    print("Learning Model Analysis Results:")
    print("=" * 50)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        summary = result['summary']
        print(f"Analysis Period: {summary['analysis_period']}")
        print(f"Total Games: {summary['total_games']}")
        print()
        print("Original Model:")
        print(f"  Average Error: {summary['original_model']['avg_error']:.2f}")
        print(f"  Bet Win Rate: {summary['original_model']['bet_win_rate']:.1%}")
        print()
        print("Learning Model:")
        print(f"  Average Error: {summary['learning_model']['avg_error']:.2f}")
        print(f"  Bet Win Rate: {summary['learning_model']['bet_win_rate']:.1%}")
        print()
        print("Improvement:")
        print(f"  Error Reduction: {summary['improvement']['error_reduction']:.2f}")
        print(f"  Predictions Improved: {summary['improvement']['error_improvement_rate']:.1%}")
        print(f"  Additional Bet Wins: {summary['improvement']['additional_bet_wins']}")
