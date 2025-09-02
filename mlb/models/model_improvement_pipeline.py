#!/usr/bin/env python3
"""
Model Improvement Pipeline for MLB Predictions
Implement ensemble methods, feature selection, and bias correction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelImprovementPipeline:
    """Advanced modeling pipeline for better MLB predictions"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.ensemble_weights = {}
        
    def prepare_training_data(self, df):
        """Prepare data for training with proper feature selection"""
        logger.info("🔧 Preparing training data...")
        
        # Remove non-predictive columns
        exclude_cols = [
            'game_id', 'date', 'home_team', 'away_team', 'venue_name',
            'home_sp_name', 'away_sp_name', 'plate_umpire',
            'weather_condition', 'weather_description',
            'total_runs', 'predicted_total', 'predicted_total_learning',
            'home_score', 'away_score', 'id', 'created_at'
        ]
        
        # Get numeric features only
        numeric_features = []
        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
                
        logger.info(f"Found {len(numeric_features)} numeric features for training")
        
        X = df[numeric_features].copy()
        y = df['total_runs'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove constant features
        constant_features = X.columns[X.std() == 0]
        if len(constant_features) > 0:
            logger.info(f"Removing {len(constant_features)} constant features")
            X = X.drop(columns=constant_features)
            
        # Remove highly correlated features (correlation > 0.95)
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        high_corr_features = [
            column for column in upper_triangle.columns 
            if any(upper_triangle[column] > 0.95)
        ]
        
        if len(high_corr_features) > 0:
            logger.info(f"Removing {len(high_corr_features)} highly correlated features")
            X = X.drop(columns=high_corr_features)
            
        logger.info(f"Final feature count: {X.shape[1]}")
        return X, y
        
    def select_best_features(self, X, y, max_features=50):
        """Select the most predictive features"""
        logger.info(f"🎯 Selecting top {max_features} features...")
        
        # Use multiple feature selection methods
        
        # 1. Statistical F-test
        f_selector = SelectKBest(score_func=f_regression, k=min(max_features, X.shape[1]))
        X_f_selected = f_selector.fit_transform(X, y)
        f_features = X.columns[f_selector.get_support()].tolist()
        
        # 2. Recursive Feature Elimination with Random Forest
        rf_estimator = RandomForestRegressor(n_estimators=50, random_state=42)
        rfe_selector = RFE(estimator=rf_estimator, n_features_to_select=min(max_features, X.shape[1]))
        X_rfe_selected = rfe_selector.fit_transform(X, y)
        rfe_features = X.columns[rfe_selector.get_support()].tolist()
        
        # 3. Feature importance from Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_rf_features = importance_df.head(max_features)['feature'].tolist()
        
        # Combine feature selections (features that appear in at least 2 methods)
        all_features = f_features + rfe_features + top_rf_features
        feature_counts = pd.Series(all_features).value_counts()
        selected_features = feature_counts[feature_counts >= 2].index.tolist()
        
        # If not enough features, take top RF features
        if len(selected_features) < 20:
            selected_features = top_rf_features[:max_features]
            
        logger.info(f"Selected {len(selected_features)} features using ensemble selection")
        
        # Show top features
        logger.info("Top 10 selected features:")
        for i, feature in enumerate(selected_features[:10], 1):
            importance = importance_df[importance_df['feature'] == feature]['importance'].iloc[0]
            logger.info(f"  {i:2}. {feature:<30} (importance: {importance:.4f})")
            
        self.selected_features = selected_features
        return X[selected_features]
        
    def train_ensemble_models(self, X, y):
        """Train multiple models for ensemble prediction"""
        logger.info("🚀 Training ensemble models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features for models that need it
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['standard'] = scaler
        
        # Define models
        models_config = {
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'use_scaling': False
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=150,
                    max_depth=8,
                    learning_rate=0.1,
                    random_state=42
                ),
                'use_scaling': False
            },
            'ridge_regression': {
                'model': Ridge(alpha=1.0, random_state=42),
                'use_scaling': True
            },
            'linear_regression': {
                'model': LinearRegression(),
                'use_scaling': True
            }
        }
        
        model_scores = {}
        
        # Train each model
        for name, config in models_config.items():
            logger.info(f"Training {name}...")
            
            model = config['model']
            
            if config['use_scaling']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            model_scores[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'model': model
            }
            
            logger.info(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
        # Store models
        self.models = {name: scores['model'] for name, scores in model_scores.items()}
        
        # Calculate ensemble weights based on inverse MAE
        total_inverse_mae = sum(1/scores['mae'] for scores in model_scores.values())
        self.ensemble_weights = {
            name: (1/scores['mae']) / total_inverse_mae 
            for name, scores in model_scores.items()
        }
        
        logger.info("Ensemble weights:")
        for name, weight in self.ensemble_weights.items():
            logger.info(f"  {name}: {weight:.3f}")
            
        return model_scores
        
    def predict_ensemble(self, X):
        """Make ensemble predictions"""
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if name in ['ridge_regression', 'linear_regression']:
                # Use scaled features
                X_scaled = self.scalers['standard'].transform(X)
                predictions[name] = model.predict(X_scaled)
            else:
                predictions[name] = model.predict(X)
                
        # Weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for name, weight in self.ensemble_weights.items():
            ensemble_pred += weight * predictions[name]
            
        return ensemble_pred
        
    def add_bias_correction(self, predictions, actual=None):
        """Apply bias correction to predictions"""
        if actual is not None:
            # Calculate current bias
            bias = np.mean(predictions - actual)
            logger.info(f"Detected bias: {bias:+.3f} runs")
            
            # Store bias for future corrections
            self.bias_correction = bias
        else:
            bias = getattr(self, 'bias_correction', 0)
            
        # Apply correction
        corrected_predictions = predictions - bias
        
        return corrected_predictions
        
    def evaluate_predictions(self, predictions, actual):
        """Comprehensive prediction evaluation"""
        logger.info("📊 Evaluating predictions...")
        
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        r2 = r2_score(actual, predictions)
        bias = np.mean(predictions - actual)
        
        # Accuracy within different thresholds
        within_1 = np.mean(np.abs(predictions - actual) <= 1.0)
        within_2 = np.mean(np.abs(predictions - actual) <= 2.0)
        within_3 = np.mean(np.abs(predictions - actual) <= 3.0)
        
        # Over/Under accuracy (compare to market if available)
        # For now, use median as proxy
        median_total = np.median(actual)
        over_under_acc = np.mean(
            (predictions > median_total) == (actual > median_total)
        )
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'bias': bias,
            'within_1_run': within_1,
            'within_2_runs': within_2,
            'within_3_runs': within_3,
            'over_under_accuracy': over_under_acc
        }
        
        logger.info("Performance Metrics:")
        logger.info(f"  MAE: {mae:.3f}")
        logger.info(f"  RMSE: {rmse:.3f}")
        logger.info(f"  R²: {r2:.3f}")
        logger.info(f"  Bias: {bias:+.3f}")
        logger.info(f"  Within 1 run: {within_1:.1%}")
        logger.info(f"  Within 2 runs: {within_2:.1%}")
        logger.info(f"  O/U Accuracy: {over_under_acc:.1%}")
        
        return metrics
        
    def save_model_pipeline(self, filepath_prefix):
        """Save the trained pipeline"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save ensemble models
        model_path = f"{filepath_prefix}_ensemble_{timestamp}.joblib"
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'ensemble_weights': self.ensemble_weights,
            'selected_features': self.selected_features,
            'bias_correction': getattr(self, 'bias_correction', 0)
        }, model_path)
        
        logger.info(f"💾 Model pipeline saved to: {model_path}")
        return model_path
        
    def load_model_pipeline(self, filepath):
        """Load a trained pipeline"""
        pipeline_data = joblib.load(filepath)
        
        self.models = pipeline_data['models']
        self.scalers = pipeline_data['scalers']
        self.ensemble_weights = pipeline_data['ensemble_weights']
        self.selected_features = pipeline_data['selected_features']
        self.bias_correction = pipeline_data.get('bias_correction', 0)
        
        logger.info(f"📂 Model pipeline loaded from: {filepath}")

def main():
    """Test the model improvement pipeline"""
    from sqlalchemy import create_engine, text
    
    # Load training data
    engine = create_engine("postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb")
    
    query = text("""
    SELECT * FROM enhanced_games 
    WHERE date >= '2025-07-01' 
    AND total_runs IS NOT NULL 
    AND (predicted_total IS NOT NULL OR predicted_total_learning IS NOT NULL)
    ORDER BY date DESC 
    LIMIT 500
    """)
    
    df = pd.read_sql(query, engine)
    logger.info(f"Loaded {len(df)} games for testing")
    
    # Initialize pipeline
    pipeline = ModelImprovementPipeline()
    
    # Prepare data
    X, y = pipeline.prepare_training_data(df)
    
    # Select features
    X_selected = pipeline.select_best_features(X, y, max_features=40)
    
    # Train ensemble
    model_scores = pipeline.train_ensemble_models(X_selected, y)
    
    # Test ensemble prediction
    predictions = pipeline.predict_ensemble(X_selected)
    
    # Apply bias correction
    corrected_predictions = pipeline.add_bias_correction(predictions, y)
    
    # Evaluate
    metrics = pipeline.evaluate_predictions(corrected_predictions, y)
    
    # Save pipeline
    model_path = pipeline.save_model_pipeline("models/improved_mlb_model")
    
    print(f"\n✅ Model improvement pipeline complete!")
    print(f"📈 Performance: MAE {metrics['mae']:.3f}, Within 1 run: {metrics['within_1_run']:.1%}")

if __name__ == "__main__":
    main()
