#!/usr/bin/env python3
"""
Retrain Adaptive Learning Model
==============================
Retrains the AdaptiveLearningPipeline with current feature schema from the enhanced pipeline.
This fixes the feature mismatch issue where the old model expects 201 features
but the current pipeline generates different features.

Result: Updated adaptive_learning_model.joblib that works with current features
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add deployment path for predictor access
sys.path.append(str(Path(__file__).parent.parent / "deployment"))
from enhanced_bullpen_predictor import EnhancedBullpenPredictor

class AdaptiveLearningModelRetrainer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Initialize the predictor to use its feature engineering
        print("🔧 Initializing Enhanced Bullpen Predictor for feature engineering...")
        self.predictor = EnhancedBullpenPredictor()
        print("✅ Feature engineering pipeline loaded")
        
        # Define data leak features to EXCLUDE (these are only available after games start/finish)
        self.data_leak_features = [
            # In-game pitcher stats
            'home_sp_er', 'away_sp_er', 'home_sp_ip', 'away_sp_ip',
            'home_sp_k', 'away_sp_k', 'home_sp_bb', 'away_sp_bb', 'home_sp_h', 'away_sp_h',
            
            # In-game team stats  
            'home_team_runs', 'away_team_runs', 'home_team_hits', 'away_team_hits',
            'home_team_rbi', 'away_team_rbi', 'home_team_lob', 'away_team_lob',
            
            # Post-game results
            'total_runs', 'actual_total', 'outcome', 'edge', 'profit',
            'predicted_total', 'predicted_total_original', 'predicted_total_learning',
            
            # Market closing data (use opening data only)
            'confidence', 'recommendation', 'prediction_timestamp', 'prediction_comparison',
            
            # ID columns that can cause overfitting
            'game_id', 'venue_id', 'home_sp_id', 'away_sp_id', 'home_team_id', 'away_team_id'
        ]
        
    def load_training_data(self, start_date='2025-03-20', end_date='2025-08-27'):
        """Load training data with current feature engineering"""
        print(f"📊 Loading training data from {start_date} to {end_date}...")
        
        # Use the enhanced predictor to get games with proper feature engineering
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        all_featured_data = []
        
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            try:
                # Use predict_today_games to get feature-engineered data
                predictions_df, featured_df, X = self.predictor.predict_today_games(date_str)
                
                if featured_df is not None and len(featured_df) > 0:
                    # Add date for reference
                    featured_df = featured_df.copy()
                    featured_df['date'] = date_str
                    all_featured_data.append(featured_df)
                    
                if len(all_featured_data) % 10 == 0:
                    print(f"  📅 Processed {len(all_featured_data)} dates...")
                    
            except Exception as e:
                print(f"  ⚠️ Skipped {date_str}: {e}")
                continue
        
        if not all_featured_data:
            raise ValueError("No training data loaded!")
            
        # Combine all data
        full_data = pd.concat(all_featured_data, ignore_index=True)
        print(f"✅ Loaded {len(full_data)} games from {len(all_featured_data)} dates")
        
        return full_data
    
    def prepare_features_and_targets(self, data):
        """Prepare feature matrix and target variable"""
        print("🔧 Preparing features and targets...")
        
        # Remove data leak features
        feature_columns = [col for col in data.columns if col not in self.data_leak_features]
        
        # Must have target variable
        if 'total_runs' not in data.columns:
            # Try to load actual results from database
            data = self.add_actual_results(data)
        
        # Filter to games with actual results
        data_with_results = data[data['total_runs'].notna()].copy()
        print(f"📊 {len(data_with_results)} games have actual results for training")
        
        if len(data_with_results) < 100:
            raise ValueError("Insufficient training data with results!")
        
        # Prepare feature matrix
        X = data_with_results[feature_columns].copy()
        y = data_with_results['total_runs'].values
        
        # Handle missing values
        print(f"🔧 Feature matrix shape: {X.shape}")
        print(f"🔧 Features: {len(feature_columns)}")
        
        # Fill missing values (same approach as enhanced predictor)
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
        
        # Handle categorical columns
        categorical_columns = X.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            X[col] = X[col].fillna('Unknown')
            # Simple label encoding for categorical features
            X[col] = pd.Categorical(X[col]).codes
        
        print(f"✅ Final feature matrix: {X.shape}")
        print(f"✅ Target range: {y.min():.1f} - {y.max():.1f} (mean: {y.mean():.1f})")
        
        return X, y, feature_columns
    
    def add_actual_results(self, data):
        """Add actual game results from database"""
        print("🔍 Loading actual game results from database...")
        
        conn = psycopg2.connect(**self.db_config)
        
        # Get actual results for games in our dataset
        game_ids = data['game_id'].unique()
        game_ids_str = "', '".join([str(gid) for gid in game_ids])
        
        query = f"""
        SELECT game_id, total_runs
        FROM enhanced_games 
        WHERE game_id IN ('{game_ids_str}')
        AND total_runs IS NOT NULL
        """
        
        results_df = pd.read_sql(query, conn)
        conn.close()
        
        print(f"📊 Found actual results for {len(results_df)} games")
        
        # Merge results back to data
        data = data.merge(results_df, on='game_id', how='left')
        
        return data
        
    def train_model(self, X, y, feature_columns):
        """Train the adaptive learning model"""
        print("🚀 Training Adaptive Learning Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"📊 Train: {len(X_train)} games, Test: {len(X_test)} games")
        
        # Train Random Forest (similar to original AdaptiveLearningPipeline)
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        
        print("🏋️ Training model...")
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("📊 MODEL PERFORMANCE:")
        print(f"   Train MAE: {train_mae:.3f}")
        print(f"   Test MAE:  {test_mae:.3f}")
        print(f"   Train R²:  {train_r2:.3f}")
        print(f"   Test R²:   {test_r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("🔝 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
        
        return model, feature_importance, test_mae
    
    def save_model(self, model, feature_columns, test_mae):
        """Save the retrained model in the same format as the original"""
        models_dir = Path(__file__).parent.parent / "models"
        model_path = models_dir / "adaptive_learning_model.joblib"
        
        # Create model bundle (same format as original)
        model_bundle = {
            'model': model,
            'feature_columns': feature_columns,
            'learning_config': {
                'model_type': 'RandomForestRegressor',
                'retrained_date': datetime.now().isoformat(),
                'feature_count': len(feature_columns),
                'test_mae': test_mae
            },
            'performance': {
                'test_mae': test_mae,
                'training_date': datetime.now().isoformat()
            },
            'preprocessing_info': {
                'missing_value_strategy': 'median_fill',
                'categorical_encoding': 'label_encoding'
            }
        }
        
        # Save the model
        joblib.dump(model_bundle, model_path)
        print(f"💾 Saved retrained model to {model_path}")
        print(f"📊 Model expects {len(feature_columns)} features")
        
        return str(model_path)

def main():
    """Main training pipeline"""
    print("🚀 ADAPTIVE LEARNING MODEL RETRAINING")
    print("=" * 50)
    
    trainer = AdaptiveLearningModelRetrainer()
    
    try:
        # Load training data with current feature engineering
        data = trainer.load_training_data()
        
        # Prepare features and targets
        X, y, feature_columns = trainer.prepare_features_and_targets(data)
        
        # Train model
        model, feature_importance, test_mae = trainer.train_model(X, y, feature_columns)
        
        # Save model
        model_path = trainer.save_model(model, feature_columns, test_mae)
        
        print("=" * 50)
        print("✅ RETRAINING COMPLETED SUCCESSFULLY!")
        print(f"💾 Model saved to: {model_path}")
        print(f"📊 Test MAE: {test_mae:.3f}")
        print(f"🔧 Features: {len(feature_columns)}")
        print("\n🎯 The retrained model is now compatible with the current feature pipeline!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
