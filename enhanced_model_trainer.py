#!/usr/bin/env python3
"""
Production Pipeline: Integrate Enhanced Features and Retrain Models
This script will update the main prediction system with enhanced features
"""

import pandas as pd
from sqlalchemy import create_engine, text
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from mlb.features.enhanced_feature_engine import EnhancedFeatureEngine
import os
from datetime import datetime

class EnhancedModelTrainer:
    def __init__(self):
        self.engine = create_engine('postgresql+psycopg2://mlbuser:mlbpass@localhost:5432/mlb')
        self.feature_engine = EnhancedFeatureEngine()
        
    def load_training_data(self, start_date='2025-07-01', min_games=500):
        """Load comprehensive training data"""
        print(f"🔄 Loading training data from {start_date}...")
        
        with self.engine.connect() as conn:
            query = text(f'''
                SELECT * FROM enhanced_games 
                WHERE date >= '{start_date}'
                AND predicted_total IS NOT NULL 
                AND total_runs IS NOT NULL
                AND home_score IS NOT NULL
                AND away_score IS NOT NULL
                ORDER BY date DESC
                LIMIT {min_games}
            ''')
            df = pd.read_sql(query, conn)
        
        print(f"📊 Loaded {len(df)} games for training")
        return df
    
    def prepare_features(self, df):
        """Apply enhanced feature engineering and prepare for modeling"""
        print("🛠️ Applying enhanced feature engineering...")
        
        # Apply enhanced features
        enhanced_df = self.feature_engine.process_enhanced_features(df.copy())
        print(f"✨ Enhanced from {len(df.columns)} to {len(enhanced_df.columns)} features")
        
        # Prepare feature matrix
        target = 'total_runs'
        exclude_cols = {
            'home_score', 'away_score', 'home_team_runs', 'away_team_runs', 
            'id', 'game_id', target, 'date', 'created_at', 'prediction_timestamp',
            'home_team', 'away_team', 'venue_name', 'ballpark', 'venue',
            'home_sp_name', 'away_sp_name', 'plate_umpire', 'home_catcher', 'away_catcher'
        }
        
        # Select numeric features
        feature_cols = []
        for col in enhanced_df.columns:
            if col not in exclude_cols and enhanced_df[col].dtype in ['int64', 'float64']:
                # Only include features with reasonable variance and not too many missing values
                missing_pct = enhanced_df[col].isnull().mean()
                variance = enhanced_df[col].var()
                if missing_pct < 0.3 and variance > 0.001:
                    feature_cols.append(col)
        
        print(f"🎯 Selected {len(feature_cols)} features for modeling")
        
        # Prepare final dataset
        X = enhanced_df[feature_cols].fillna(0)
        y = enhanced_df[target]
        
        return X, y, feature_cols, enhanced_df
    
    def train_models(self, X, y, feature_cols):
        """Train baseline and enhanced models"""
        print("🚀 Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        print(f"📈 Training set: {len(X_train)} games")
        print(f"📉 Test set: {len(X_test)} games")
        
        # Train enhanced model
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\\n📊 ENHANCED MODEL PERFORMANCE:")
        print(f"📈 Training MAE: {train_mae:.3f} runs")
        print(f"📉 Test MAE:     {test_mae:.3f} runs")
        print(f"📈 Training R²:  {train_r2:.3f}")
        print(f"📉 Test R²:      {test_r2:.3f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\n🔥 TOP 15 MOST IMPORTANT FEATURES:")
        for i, (_, row) in enumerate(importance_df.head(15).iterrows()):
            print(f"{i+1:2d}. {row['feature']}: {row['importance']:.4f}")
        
        return model, importance_df, {
            'train_mae': train_mae, 'test_mae': test_mae,
            'train_r2': train_r2, 'test_r2': test_r2
        }
    
    def save_model(self, model, feature_cols, importance_df, metrics):
        """Save the enhanced model for production use"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f'models_enhanced_{timestamp}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'enhanced_total_runs_model.joblib')
        joblib.dump(model, model_path)
        print(f"💾 Saved model to {model_path}")
        
        # Save feature list
        feature_path = os.path.join(model_dir, 'feature_columns.joblib')
        joblib.dump(feature_cols, feature_path)
        print(f"💾 Saved features to {feature_path}")
        
        # Save importance and metrics
        importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(model_dir, 'model_metrics.csv'), index=False)
        
        print(f"✅ Model package saved to {model_dir}/")
        return model_dir
    
    def run_full_pipeline(self):
        """Execute the complete enhanced model training pipeline"""
        print("🚀 STARTING ENHANCED MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Load data
        df = self.load_training_data()
        
        # Prepare features
        X, y, feature_cols, enhanced_df = self.prepare_features(df)
        
        # Train model
        model, importance_df, metrics = self.train_models(X, y, feature_cols)
        
        # Save model
        model_dir = self.save_model(model, feature_cols, importance_df, metrics)
        
        print("\\n" + "=" * 60)
        print("✅ ENHANCED MODEL TRAINING COMPLETE!")
        print(f"📁 Model saved to: {model_dir}")
        print(f"📊 Test MAE: {metrics['test_mae']:.3f} runs")
        print(f"📊 Test R²:  {metrics['test_r2']:.3f}")
        print("=" * 60)
        
        return model_dir, metrics

def main():
    trainer = EnhancedModelTrainer()
    model_dir, metrics = trainer.run_full_pipeline()
    
    print(f"\\n🎯 Next steps:")
    print(f"1. Review feature importance in {model_dir}/feature_importance.csv")
    print(f"2. Test the model on recent games")
    print(f"3. Deploy to production pipeline")
    print(f"4. Monitor performance improvements")

if __name__ == "__main__":
    main()
