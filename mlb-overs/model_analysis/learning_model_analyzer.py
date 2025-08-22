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
import sys

# Add deployment path for predictor access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'deployment'))
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

class LearningModelAnalyzer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Initialize the predictor to use its feature engineering
        self.predictor = EnhancedBullpenPredictor()
        
        # Load the learning model
        self.model = None
        self._load_model()
    def _load_model(self):
        """Load the learning model"""
        try:
            # First try to load the latest retrained model
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            retrained_models = []
            
            if os.path.exists(models_dir):
                for filename in os.listdir(models_dir):
                    if filename.startswith('learning_model_retrained_') and filename.endswith('.joblib'):
                        retrained_models.append(filename)
            
            if retrained_models:
                # Use the most recent retrained model
                latest_model = sorted(retrained_models)[-1]
                model_path = os.path.join(models_dir, latest_model)
                print(f"🔄 Loading latest retrained model: {latest_model}")
                
                model_package = joblib.load(model_path)
                
                # Handle both old and new model formats
                if isinstance(model_package, dict) and 'model' in model_package:
                    self.model = model_package['model']
                    print(f"✅ Loaded retrained model with {len(model_package.get('feature_names', []))} features")
                    print(f"   Training date: {model_package.get('created_date', 'Unknown')}")
                    metrics = model_package.get('training_metrics', {})
                    if 'test_mae' in metrics:
                        print(f"   Test MAE: {metrics['test_mae']:.3f}")
                else:
                    self.model = model_package
                    print(f"✅ Loaded retrained model (legacy format)")
                    
                self._model_path = model_path
                return
                
            # Fallback to original enhanced model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'enhanced_leak_free_model.joblib')
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                self._model_path = model_path
                print(f"⚠️ Using fallback enhanced model: {model_path}")
            else:
                print(f"❌ No models found")
                self.model = None
                self._model_path = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            self._model_path = None
        
    def get_historical_games(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical games with ALL raw data needed for feature engineering"""
        
        conn = psycopg2.connect(**self.db_config)
        
        # Get ALL columns from enhanced_games to ensure we have everything needed
        query = """
        SELECT *
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND total_runs IS NOT NULL
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        print(f"✅ Retrieved {len(df)} historical games from {start_date} to {end_date}")
        print(f"   Columns available: {len(df.columns)}")
        
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
        """Prepare features using the same feature engineering as the prediction system"""
        
        if self.model is None:
            print("❌ No model loaded for feature preparation")
            return pd.DataFrame()
            
        try:
            # Use the predictor's feature engineering pipeline
            print(f"🔧 Applying feature engineering to {len(df)} games...")
            
            # The predictor's engineer_features method will transform raw data into 
            # the engineered features that the model expects
            featured_df = self.predictor.engineer_features(df)
            
            print(f"✅ Feature engineering complete: {len(featured_df.columns)} features generated")
            
            # Get the feature names that the model expects
            if hasattr(self.model, 'feature_names_in_'):
                expected_features = self.model.feature_names_in_
                print(f"   Model expects {len(expected_features)} features")
                
                # Check which expected features are missing
                missing_features = set(expected_features) - set(featured_df.columns)
                if missing_features:
                    print(f"⚠️  Missing features: {len(missing_features)}")
                    for feat in list(missing_features)[:10]:  # Show first 10
                        print(f"     - {feat}")
                    if len(missing_features) > 10:
                        print(f"     ... and {len(missing_features) - 10} more")
                        
                # Filter to only the features the model expects
                available_features = [f for f in expected_features if f in featured_df.columns]
                featured_df = featured_df[available_features]
                
                print(f"✅ Using {len(available_features)} out of {len(expected_features)} expected features")
            else:
                print("⚠️  Model doesn't specify expected features, using all available")
                
            return featured_df
            
        except Exception as e:
            print(f"❌ Error in feature preparation: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
            
    def apply_learning_model_to_historical_games(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """Apply current learning model to historical games and compare performance"""
        
        print(f"🔍 Analyzing learning model performance on historical data: {start_date} to {end_date}")
        
        # Get historical games with all raw data
        historical_games = self.get_historical_games(start_date, end_date)
        print(f"✅ Retrieved {len(historical_games)} historical games")
        
        if len(historical_games) == 0:
            return {"error": "No historical games found for the specified date range"}
            
        if self.model is None:
            return {"error": "No learning model available for analysis"}
            
        # Prepare features using the same pipeline as predictions
        features_df = self.prepare_features_for_prediction(historical_games)
        
        if len(features_df) == 0:
            return {"error": "Feature preparation failed"}
            
        try:
            # Apply learning model to get new predictions
            learning_predictions = self.model.predict(features_df)
            
            # Prepare results comparison
            results = {
                'analysis_period': f"{start_date} to {end_date}",
                'total_games': len(historical_games),
                'model_path': getattr(self, '_model_path', 'Retrained learning model'),
                'features_used': len(features_df.columns),
                'original_vs_learning': self._compare_predictions(
                    historical_games, learning_predictions
                ),
                'performance_metrics': self._calculate_performance_metrics(
                    historical_games, learning_predictions
                )
            }
            
            return results
            
        except Exception as e:
            print(f"❌ Error applying learning model: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Model application failed: {str(e)}"}
        
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
    def _compare_predictions(self, historical_games: pd.DataFrame, learning_predictions: np.ndarray) -> Dict[str, Any]:
        """Compare original predictions vs learning model predictions"""
        
        comparisons = []
        
        for i, row in historical_games.iterrows():
            original_pred = row.get('predicted_total', 0)  # Use get() in case column missing
            learning_pred = learning_predictions[i]
            actual = row['total_runs']
            
            # Calculate errors
            original_error = abs(original_pred - actual) if original_pred > 0 else 999
            learning_error = abs(learning_pred - actual)
            
            comparisons.append({
                'game_id': row['game_id'],
                'date': str(row['date']),
                'teams': f"{row['away_team']} @ {row['home_team']}",
                'actual_total': actual,
                'original_prediction': original_pred,
                'learning_prediction': learning_pred,
                'original_error': original_error,
                'learning_error': learning_error,
                'learning_better': learning_error < original_error,
                'improvement': original_error - learning_error
            })
            
        return comparisons
        
    def _calculate_performance_metrics(self, historical_games: pd.DataFrame, learning_predictions: np.ndarray) -> Dict[str, Any]:
        """Calculate performance metrics for both models"""
        
        # Calculate metrics for original model
        original_preds = historical_games['predicted_total'].fillna(0)
        original_errors = np.abs(original_preds - historical_games['total_runs'])
        original_valid = original_preds > 0
        
        # Calculate metrics for learning model
        learning_errors = np.abs(learning_predictions - historical_games['total_runs'])
        
        # Overall metrics
        metrics = {
            'original_model': {
                'avg_error': float(original_errors[original_valid].mean()) if original_valid.any() else 999,
                'rmse': float(np.sqrt(np.mean(original_errors[original_valid]**2))) if original_valid.any() else 999,
                'valid_predictions': int(original_valid.sum())
            },
            'learning_model': {
                'avg_error': float(learning_errors.mean()),
                'rmse': float(np.sqrt(np.mean(learning_errors**2))),
                'valid_predictions': len(learning_predictions)
            }
        }
        
        # Calculate improvement metrics
        if original_valid.any():
            # Only compare where both models have valid predictions
            valid_original_errors = original_errors[original_valid]
            valid_learning_errors = learning_errors[original_valid] 
            
            improved_count = (valid_learning_errors < valid_original_errors).sum()
            total_valid = len(valid_original_errors)
            
            metrics['improvement'] = {
                'error_reduction': float(valid_original_errors.mean() - valid_learning_errors.mean()),
                'improvement_rate': float(improved_count / total_valid) if total_valid > 0 else 0,
                'games_improved': int(improved_count),
                'total_compared': int(total_valid)
            }
        else:
            metrics['improvement'] = {
                'error_reduction': 999,  # No valid original predictions to compare
                'improvement_rate': 0,
                'games_improved': 0,
                'total_compared': 0
            }
            
        return metrics


# Usage function for API
def analyze_learning_improvement(days_back: int = 14) -> Dict[str, Any]:
    """Analyze learning model improvement over the last N days"""
    
    analyzer = LearningModelAnalyzer()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    return analyzer.apply_learning_model_to_historical_games(
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
        print(f"Analysis Period: {result['analysis_period']}")
        print(f"Total Games: {result['total_games']}")
        print(f"Features Used: {result['features_used']}")
        print(f"Model: {result['model_path']}")
        print()
        
        metrics = result['performance_metrics']
        print("PERFORMANCE COMPARISON:")
        print(f"Original Model Avg Error: {metrics['original_model']['avg_error']:.2f}")
        print(f"Learning Model Avg Error: {metrics['learning_model']['avg_error']:.2f}")
        print()
        
        if 'improvement' in metrics:
            imp = metrics['improvement']
            print("IMPROVEMENT ANALYSIS:")
            print(f"Error Reduction: {imp['error_reduction']:.2f}")
            print(f"Games Improved: {imp['games_improved']} / {imp['total_compared']}")
            print(f"Improvement Rate: {imp['improvement_rate']:.1%}")