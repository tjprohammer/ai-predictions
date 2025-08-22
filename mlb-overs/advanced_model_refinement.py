#!/usr/bin/env python3
"""
Advanced Model Refinement System
================================
A sophisticated system to refine the learning model and push picking accuracy to 90%+
by analyzing patterns, feature importance, and prediction errors.
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from pathlib import Path
from sqlalchemy import create_engine, text
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedModelRefinement:
    """
    Advanced refinement system to push model accuracy to 90%+
    """
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or 'postgresql://mlbuser:mlbpass@localhost/mlb'
        self.engine = create_engine(self.db_url)
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        # Advanced refinement strategies
        self.refinement_strategies = {
            'ensemble_stacking': True,
            'feature_engineering_v2': True,
            'temporal_patterns': True,
            'situational_modeling': True,
            'calibration_optimization': True,
            'error_pattern_analysis': True
        }
        
        # Load current model and backtesting results
        self.current_model = None
        self.backtest_results = None
        self.load_current_assets()
        
    def load_current_assets(self):
        """Load current model and backtesting results"""
        try:
            # Load current learning model
            model_path = self.models_dir / "adaptive_learning_model.joblib"
            if model_path.exists():
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict) and 'model' in model_data:
                    self.current_model = model_data['model']
                    self.feature_names = model_data.get('feature_columns', [])
                    logger.info("✅ Current learning model loaded successfully")
                    
            # Load backtesting results
            backtest_path = Path(__file__).parent / "backtest_results_20250822_145615.json"
            if backtest_path.exists():
                with open(backtest_path, 'r') as f:
                    self.backtest_results = json.load(f)
                logger.info("✅ Backtesting results loaded successfully")
                
        except Exception as e:
            logger.error(f"❌ Error loading current assets: {e}")
    
    def analyze_prediction_errors(self) -> Dict:
        """Analyze patterns in prediction errors to identify improvement opportunities"""
        logger.info("🔍 Analyzing prediction error patterns...")
        
        if not self.backtest_results:
            logger.error("No backtesting results available")
            return {}
            
        games = self.backtest_results['individual_games']
        df = pd.DataFrame(games)
        
        # Convert numeric columns
        numeric_cols = ['actual_total', 'market_total', 'original_prediction', 
                       'learning_prediction', 'original_error', 'learning_error']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        analysis = {
            'error_distribution': {
                'mean_error': float(df['learning_error'].mean()),
                'median_error': float(df['learning_error'].median()),
                'std_error': float(df['learning_error'].std()),
                'percentiles': {
                    '25th': float(df['learning_error'].quantile(0.25)),
                    '75th': float(df['learning_error'].quantile(0.75)),
                    '90th': float(df['learning_error'].quantile(0.90))
                }
            },
            'error_by_total_range': {},
            'systematic_biases': {
                'high_total_bias': float(df[df['actual_total'] > 10]['learning_error'].mean()),
                'low_total_bias': float(df[df['actual_total'] < 8]['learning_error'].mean()),
                'under_prediction_rate': float((df['learning_prediction'] < df['actual_total']).mean()),
                'over_prediction_rate': float((df['learning_prediction'] > df['actual_total']).mean())
            },
            'worst_predictions': []
        }
        
        # Analyze errors by total ranges
        total_ranges = [(0, 7), (7, 9), (9, 11), (11, 15), (15, 100)]
        for low, high in total_ranges:
            mask = (df['actual_total'] >= low) & (df['actual_total'] < high)
            if mask.sum() > 0:
                range_data = df[mask]
                analysis['error_by_total_range'][f'{low}-{high}'] = {
                    'count': int(mask.sum()),
                    'mean_error': float(range_data['learning_error'].mean()),
                    'picking_accuracy': float((range_data['learning_correct_pick']).mean())
                }
        
        # Find worst predictions
        worst_errors = df.nlargest(10, 'learning_error')
        for _, game in worst_errors.iterrows():
            analysis['worst_predictions'].append({
                'game_id': game['game_id'],
                'matchup': game['matchup'],
                'actual': float(game['actual_total']),
                'predicted': float(game['learning_prediction']),
                'error': float(game['learning_error'])
            })
            
        logger.info(f"📊 Error analysis complete: Mean error = {analysis['error_distribution']['mean_error']:.3f}")
        return analysis
    
    def create_advanced_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction accuracy"""
        logger.info("🔧 Creating advanced features...")
        
        X_enhanced = X.copy()
        
        # Feature engineering strategies
        features_created = 0
        
        # 1. Interaction features for key combinations
        if 'market_total' in X.columns and 'predicted_total' in X.columns:
            X_enhanced['market_vs_prediction_diff'] = X['market_total'] - X['predicted_total']
            X_enhanced['market_prediction_ratio'] = X['market_total'] / (X['predicted_total'] + 0.1)
            features_created += 2
            
        # 2. Weather impact combinations
        weather_cols = [col for col in X.columns if any(term in col.lower() 
                       for term in ['temp', 'wind', 'humidity', 'pressure'])]
        if len(weather_cols) >= 2:
            # Weather severity index
            weather_data = X[weather_cols].fillna(0)
            if len(weather_data.columns) > 0:
                X_enhanced['weather_severity'] = weather_data.std(axis=1)
                features_created += 1
        
        # 3. Team performance ratios and trends
        team_cols = [col for col in X.columns if any(term in col.lower() 
                    for term in ['era', 'whip', 'ops', 'avg', 'obp', 'slg'])]
        home_team_cols = [col for col in team_cols if 'home' in col.lower()]
        away_team_cols = [col for col in team_cols if 'away' in col.lower()]
        
        # Create team strength ratios
        for home_col in home_team_cols:
            away_equivalent = home_col.replace('home', 'away')
            if away_equivalent in X.columns:
                ratio_name = f"{home_col}_vs_away_ratio"
                X_enhanced[ratio_name] = (X[home_col] + 0.001) / (X[away_equivalent] + 0.001)
                features_created += 1
        
        # 4. Bullpen strength combinations
        bullpen_cols = [col for col in X.columns if 'bullpen' in col.lower()]
        if len(bullpen_cols) >= 2:
            bullpen_data = X[bullpen_cols].fillna(X[bullpen_cols].median())
            X_enhanced['combined_bullpen_strength'] = bullpen_data.mean(axis=1)
            X_enhanced['bullpen_differential'] = bullpen_data.max(axis=1) - bullpen_data.min(axis=1)
            features_created += 2
        
        # 5. Pitcher fatigue and usage patterns
        pitcher_cols = [col for col in X.columns if any(term in col.lower() 
                       for term in ['pitcher', 'sp_', 'innings', 'pitch_count'])]
        if len(pitcher_cols) > 0:
            pitcher_data = X[pitcher_cols].fillna(X[pitcher_cols].median())
            X_enhanced['pitcher_fatigue_index'] = pitcher_data.std(axis=1)
            features_created += 1
        
        # 6. Market efficiency indicators
        if 'over_odds' in X.columns and 'under_odds' in X.columns:
            X_enhanced['odds_spread'] = abs(X['over_odds'] - X['under_odds'])
            X_enhanced['market_confidence'] = 1 / (X_enhanced['odds_spread'] + 0.01)
            features_created += 2
        
        # 7. Temporal features (if date information available)
        if 'date' in X.columns:
            try:
                dates = pd.to_datetime(X['date'])
                X_enhanced['day_of_week'] = dates.dt.dayofweek
                X_enhanced['month'] = dates.dt.month
                X_enhanced['is_weekend'] = (dates.dt.dayofweek >= 5).astype(int)
                features_created += 3
            except:
                pass
        
        # 8. Venue-specific adjustments
        if 'venue_name' in X.columns:
            venue_stats = X.groupby('venue_name')['total_runs'].mean() if 'total_runs' in X.columns else None
            if venue_stats is not None:
                X_enhanced['venue_avg_runs'] = X['venue_name'].map(venue_stats)
                features_created += 1
        
        logger.info(f"✅ Created {features_created} advanced features")
        return X_enhanced
    
    def optimize_model_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Optimize hyperparameters using advanced techniques"""
        logger.info("⚙️ Optimizing model hyperparameters...")
        
        # Prepare data
        X_processed = self.create_advanced_features(X)
        X_processed = self.handle_categorical_features(X_processed)
        
        # Scale features
        scaler = RobustScaler()  # More robust to outliers than StandardScaler
        X_scaled = scaler.fit_transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
        
        models_to_test = {
            'random_forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [200, 300, 500],
                    'max_depth': [15, 20, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2', 0.8]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [6, 8, 10],
                    'subsample': [0.8, 0.9, 1.0]
                }
            },
            'xgboost': {
                'model': xgb.XGBRegressor(random_state=42, eval_metric='mae'),
                'params': {
                    'n_estimators': [200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [6, 8, 10],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            }
        }
        
        best_models = {}
        
        for model_name, config in models_to_test.items():
            logger.info(f"🔍 Optimizing {model_name}...")
            
            try:
                grid_search = GridSearchCV(
                    config['model'],
                    config['params'],
                    cv=5,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_scaled, y)
                
                best_models[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'best_score': -grid_search.best_score_,
                    'cv_scores': cross_val_score(grid_search.best_estimator_, X_scaled, y, 
                                               cv=5, scoring='neg_mean_absolute_error')
                }
                
                logger.info(f"✅ {model_name} optimized - MAE: {best_models[model_name]['best_score']:.3f}")
                
            except Exception as e:
                logger.error(f"❌ Error optimizing {model_name}: {e}")
        
        return best_models, scaler, X_processed.columns.tolist()
    
    def handle_categorical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Advanced categorical feature handling"""
        X_processed = X.copy()
        
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                # Use frequency encoding for high cardinality
                if X_processed[col].nunique() > 10:
                    freq_map = X_processed[col].value_counts().to_dict()
                    X_processed[col] = X_processed[col].map(freq_map).fillna(0)
                else:
                    # Use label encoding for low cardinality
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].fillna('Unknown'))
        
        return X_processed
    
    def create_ensemble_model(self, best_models: Dict, X: pd.DataFrame, y: pd.Series) -> object:
        """Create an ensemble model from the best performing models"""
        logger.info("🔗 Creating ensemble model...")
        
        # Use the top 3 performing models for ensemble
        sorted_models = sorted(best_models.items(), key=lambda x: x[1]['best_score'])
        top_models = dict(sorted_models[:3])
        
        class EnsembleRegressor:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights or [1/len(models)] * len(models)
                
            def fit(self, X, y):
                for model in self.models:
                    model.fit(X, y)
                return self
                
            def predict(self, X):
                predictions = np.array([model.predict(X) for model in self.models])
                return np.average(predictions, axis=0, weights=self.weights)
        
        # Extract models and create ensemble
        models = [config['model'] for config in top_models.values()]
        
        # Calculate weights based on inverse of MAE (better models get higher weight)
        scores = [config['best_score'] for config in top_models.values()]
        weights = [1/(score + 0.001) for score in scores]
        weights = [w/sum(weights) for w in weights]  # Normalize
        
        ensemble = EnsembleRegressor(models, weights)
        
        logger.info(f"✅ Ensemble created with {len(models)} models")
        logger.info(f"   Weights: {[f'{w:.3f}' for w in weights]}")
        
        return ensemble
    
    def apply_calibration_corrections(self, model, X: pd.DataFrame, y: pd.Series):
        """Apply advanced calibration corrections based on error patterns"""
        logger.info("🎯 Applying calibration corrections...")
        
        # Make predictions with current model
        predictions = model.predict(X)
        errors = predictions - y
        
        # Identify systematic biases
        bias_corrections = {}
        
        # Total range biases
        for low, high in [(0, 7), (7, 9), (9, 11), (11, 15), (15, 100)]:
            mask = (y >= low) & (y < high)
            if mask.sum() > 5:  # Need enough samples
                range_bias = errors[mask].mean()
                bias_corrections[f'range_{low}_{high}'] = range_bias
                
        # Create calibrated model wrapper
        class CalibratedModel:
            def __init__(self, base_model, corrections):
                self.base_model = base_model
                self.corrections = corrections
                
            def predict(self, X):
                base_predictions = self.base_model.predict(X)
                calibrated = base_predictions.copy()
                
                # Apply range-based corrections
                for range_key, correction in self.corrections.items():
                    if range_key.startswith('range_'):
                        _, low, high = range_key.split('_')
                        low, high = float(low), float(high)
                        mask = (base_predictions >= low) & (base_predictions < high)
                        calibrated[mask] -= correction * 0.5  # Apply 50% of correction
                        
                return calibrated
                
            def fit(self, X, y):
                return self.base_model.fit(X, y)
        
        calibrated_model = CalibratedModel(model, bias_corrections)
        
        logger.info(f"✅ Calibration applied with {len(bias_corrections)} corrections")
        return calibrated_model
    
    def train_refined_model(self) -> Dict:
        """Train the refined model with all optimizations"""
        logger.info("🚀 Training refined model...")
        
        # Get training data
        training_data = self.get_comprehensive_training_data()
        if training_data.empty:
            logger.error("No training data available")
            return {}
        
        # Prepare features and target
        target_col = 'total_runs'
        if target_col not in training_data.columns:
            logger.error(f"Target column '{target_col}' not found")
            return {}
        
        X = training_data.drop(columns=[target_col, 'game_id', 'date'], errors='ignore')
        y = training_data[target_col]
        
        # Remove any remaining non-numeric or problematic columns
        X = X.select_dtypes(include=[np.number])
        
        logger.info(f"📊 Training with {len(X)} samples, {len(X.columns)} features")
        
        # Optimize models
        best_models, scaler, feature_columns = self.optimize_model_hyperparameters(X, y)
        
        if not best_models:
            logger.error("No models were successfully optimized")
            return {}
        
        # Create enhanced features for final training
        X_enhanced = self.create_advanced_features(X)
        X_processed = self.handle_categorical_features(X_enhanced)
        X_scaled = scaler.transform(X_processed)
        X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)
        
        # Create ensemble model
        ensemble_model = self.create_ensemble_model(best_models, X_scaled, y)
        ensemble_model.fit(X_scaled, y)
        
        # Apply calibration
        final_model = self.apply_calibration_corrections(ensemble_model, X_scaled, y)
        
        # Evaluate final model
        final_predictions = final_model.predict(X_scaled)
        final_mae = mean_absolute_error(y, final_predictions)
        
        # Calculate picking accuracy
        picking_accuracy = self.calculate_picking_accuracy(final_predictions, y, training_data.get('market_total', y))
        
        # Save the refined model
        model_data = {
            'model': final_model,
            'scaler': scaler,
            'feature_columns': X_processed.columns.tolist(),
            'refinement_config': self.refinement_strategies,
            'performance': {
                'mae': final_mae,
                'picking_accuracy': picking_accuracy,
                'training_samples': len(X)
            },
            'best_individual_models': {name: config['best_params'] for name, config in best_models.items()}
        }
        
        # Save model
        model_path = self.models_dir / "refined_learning_model.joblib"
        joblib.dump(model_data, model_path)
        
        logger.info(f"🎉 Refined model trained successfully!")
        logger.info(f"   Final MAE: {final_mae:.3f}")
        logger.info(f"   Picking Accuracy: {picking_accuracy:.1f}%")
        logger.info(f"   Model saved to: {model_path}")
        
        return {
            'model_path': str(model_path),
            'performance': model_data['performance'],
            'feature_count': len(X_processed.columns),
            'refinement_strategies': self.refinement_strategies
        }
    
    def calculate_picking_accuracy(self, predictions: np.ndarray, actuals: np.ndarray, market_totals: pd.Series) -> float:
        """Calculate over/under picking accuracy"""
        if len(market_totals) != len(predictions):
            market_totals = pd.Series([actuals.mean()] * len(predictions))
        
        correct_picks = 0
        total_picks = len(predictions)
        
        for i in range(len(predictions)):
            pred = predictions[i]
            actual = actuals.iloc[i] if hasattr(actuals, 'iloc') else actuals[i]
            market = market_totals.iloc[i] if hasattr(market_totals, 'iloc') else market_totals[i] if hasattr(market_totals, '__getitem__') else actuals.mean()
            
            # Determine if prediction was correct
            if pred > market and actual > market:  # Predicted over, actual over
                correct_picks += 1
            elif pred < market and actual < market:  # Predicted under, actual under
                correct_picks += 1
        
        return (correct_picks / total_picks) * 100
    
    def get_comprehensive_training_data(self) -> pd.DataFrame:
        """Get comprehensive training data for model refinement"""
        try:
            query = text("""
                SELECT * FROM enhanced_games 
                WHERE total_runs IS NOT NULL 
                AND date >= :start_date
                ORDER BY date DESC
                LIMIT 2000
            """)
            
            start_date = (datetime.now() - timedelta(days=120)).date()
            
            with self.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={'start_date': start_date})
            
            logger.info(f"📊 Retrieved {len(df)} training samples")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error getting training data: {e}")
            return pd.DataFrame()
    
    def run_complete_refinement(self) -> Dict:
        """Run the complete model refinement process"""
        logger.info("🔥 Starting complete model refinement process...")
        
        # Step 1: Analyze current errors
        error_analysis = self.analyze_prediction_errors()
        
        # Step 2: Train refined model
        training_results = self.train_refined_model()
        
        # Step 3: Compile final report
        refinement_report = {
            'timestamp': datetime.now().isoformat(),
            'error_analysis': error_analysis,
            'training_results': training_results,
            'improvement_strategies_used': self.refinement_strategies,
            'target_accuracy': '90%+',
            'current_baseline': '84.5%'
        }
        
        # Save report
        report_path = self.models_dir / f"refinement_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(refinement_report, f, indent=2, default=str)
        
        logger.info(f"📊 Refinement report saved to: {report_path}")
        return refinement_report

def main():
    """Main execution function"""
    refinement_system = AdvancedModelRefinement()
    
    print("🎯 Advanced Model Refinement System")
    print("=" * 50)
    print("Goal: Push picking accuracy from 84.5% to 90%+")
    print()
    
    # Run complete refinement
    results = refinement_system.run_complete_refinement()
    
    print("\n🎉 Refinement Process Complete!")
    print(f"📈 Performance Summary:")
    if 'training_results' in results and 'performance' in results['training_results']:
        perf = results['training_results']['performance']
        print(f"   Final MAE: {perf.get('mae', 'N/A'):.3f}")
        print(f"   Picking Accuracy: {perf.get('picking_accuracy', 'N/A'):.1f}%")
        print(f"   Training Samples: {perf.get('training_samples', 'N/A')}")
    
    return results

if __name__ == "__main__":
    main()
