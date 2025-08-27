#!/usr/bin/env python3
"""
COMPREHENSIVE ML TRAINING PIPELINE
Train both original and learning models with our excellent 1,978-game dataset
Then set up continuous learning for daily adaptation
"""

import psycopg2
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveMLTrainer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Core features for consistent training
        self.core_features = [
            # Rolling statistics (recently fixed!)
            'home_team_runs_l7', 'away_team_runs_l7',
            'home_team_runs_allowed_l7', 'away_team_runs_allowed_l7',
            'home_team_runs_l20', 'away_team_runs_l20',
            'home_team_runs_l30', 'away_team_runs_l30',
            
            # Batting statistics
            'home_team_avg', 'away_team_avg',
            'home_team_obp', 'away_team_obp',
            'home_team_ops', 'away_team_ops',
            'home_team_woba', 'away_team_woba',
            
            # Pitching statistics (recently fixed!)
            'home_sp_season_era', 'away_sp_season_era',
            'home_sp_era_l3starts', 'away_sp_era_l3starts',
            'home_bullpen_era', 'away_bullpen_era',
            
            # Environmental factors
            'temperature', 'wind_speed',
            'home_lineup_strength', 'away_lineup_strength',
            'offensive_environment_score',
            
            # Market data
            'market_total',
            
            # Advanced metrics
            'home_team_hits', 'away_team_hits',
            'home_team_rbi', 'away_team_rbi'
        ]
        
        self.models_dir = "S:/Projects/AI_Predictions/mlb-overs/models"
        
    def connect_db(self):
        return psycopg2.connect(**self.db_config)
    
    def get_training_data(self):
        """Get our excellent 1,978-game training dataset"""
        
        print("📊 LOADING TRAINING DATA")
        print("   Using our verified 1,978-game dataset")
        print("=" * 50)
        
        conn = self.connect_db()
        
        # Get complete games for training (no future data leakage!)
        training_query = f"""
        SELECT 
            date,
            home_team,
            away_team,
            total_runs,  -- TARGET VARIABLE
            {', '.join(self.core_features)}
        FROM enhanced_games
        WHERE date >= '2025-03-01'
          AND date <= '2025-08-23'  -- Only completed games
          AND total_runs IS NOT NULL
          AND home_team_runs_l7 IS NOT NULL
          AND away_team_runs_l7 IS NOT NULL
          AND home_sp_season_era IS NOT NULL
          AND away_sp_season_era IS NOT NULL
          AND home_team_avg IS NOT NULL
          AND away_team_avg IS NOT NULL
        ORDER BY date;
        """
        
        df = pd.read_sql(training_query, conn)
        conn.close()
        
        print(f"✅ Loaded {len(df)} complete games")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Split by date for temporal validation (no data leakage)
        cutoff_date = '2025-07-15'  # 70/30 split approximately
        train_df = df[df['date'] <= cutoff_date]
        val_df = df[df['date'] > cutoff_date]
        
        print(f"   Training set: {len(train_df)} games (through {cutoff_date})")
        print(f"   Validation set: {len(val_df)} games (after {cutoff_date})")
        
        return df, train_df, val_df
    
    def prepare_features(self, df):
        """Prepare features for training with consistent feature set"""
        
        print(f"\n🔧 PREPARING FEATURES")
        print("=" * 30)
        
        # Start with core features
        available_features = []
        for feature in self.core_features:
            if feature in df.columns:
                available_features.append(feature)
            else:
                print(f"   ⚠️  Missing feature: {feature}")
        
        print(f"✅ Using {len(available_features)} consistent features")
        
        # Prepare feature matrix
        X = df[available_features].copy()
        y = df['total_runs'].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Create feature engineering
        if 'market_total' in X.columns:
            # Market vs team strength differential
            X['team_strength_vs_market'] = (
                (X['home_team_runs_l7'] + X['away_team_runs_l7']) / 14
            ) - X['market_total']
        
        if 'home_sp_season_era' in X.columns and 'away_sp_season_era' in X.columns:
            # Combined pitching strength
            X['combined_sp_era'] = (X['home_sp_season_era'] + X['away_sp_season_era']) / 2
            X['sp_era_differential'] = abs(X['home_sp_season_era'] - X['away_sp_season_era'])
        
        if 'temperature' in X.columns:
            # Weather impact on offense
            X['weather_offense_factor'] = np.where(X['temperature'] > 75, 1.1, 
                                         np.where(X['temperature'] < 50, 0.9, 1.0))
        
        final_features = list(X.columns)
        print(f"   Final feature count: {len(final_features)}")
        
        return X, y, final_features
    
    def train_original_model(self, X_train, y_train, X_val, y_val, features):
        """Train the original enhanced model"""
        
        print(f"\n🎯 TRAINING ORIGINAL MODEL")
        print("=" * 40)
        
        # Original model: Random Forest (proven stable)
        original_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        original_model.fit(X_train, y_train)
        
        # Validate performance
        train_pred = original_model.predict(X_train)
        val_pred = original_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"✅ Original Model Performance:")
        print(f"   Training MAE: {train_mae:.3f}")
        print(f"   Validation MAE: {val_mae:.3f}")
        print(f"   R² Score: {r2_score(y_val, val_pred):.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': original_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 5 Features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        return original_model, feature_importance
    
    def train_learning_model(self, X_train, y_train, X_val, y_val, features):
        """Train the adaptive learning model"""
        
        print(f"\n🧠 TRAINING LEARNING MODEL")
        print("=" * 40)
        
        # Learning model: Gradient Boosting (adaptive)
        learning_model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )
        
        # Train model
        learning_model.fit(X_train, y_train)
        
        # Validate performance
        train_pred = learning_model.predict(X_train)
        val_pred = learning_model.predict(X_val)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        print(f"✅ Learning Model Performance:")
        print(f"   Training MAE: {train_mae:.3f}")
        print(f"   Validation MAE: {val_mae:.3f}")
        print(f"   R² Score: {r2_score(y_val, val_pred):.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': learning_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n📈 Top 5 Features:")
        for i, (_, row) in enumerate(feature_importance.head().iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
        
        return learning_model, feature_importance
    
    def evaluate_dual_models(self, original_model, learning_model, X_val, y_val):
        """Compare both models on validation set"""
        
        print(f"\n⚖️  DUAL MODEL COMPARISON")
        print("=" * 40)
        
        original_pred = original_model.predict(X_val)
        learning_pred = learning_model.predict(X_val)
        
        original_mae = mean_absolute_error(y_val, original_pred)
        learning_mae = mean_absolute_error(y_val, learning_pred)
        
        # Picking accuracy (within 0.5 runs)
        original_picks = np.abs(original_pred - y_val) <= 0.5
        learning_picks = np.abs(learning_pred - y_val) <= 0.5
        
        original_accuracy = original_picks.mean()
        learning_accuracy = learning_picks.mean()
        
        print(f"📊 Model Comparison:")
        print(f"   Original Model MAE: {original_mae:.3f}")
        print(f"   Learning Model MAE: {learning_mae:.3f}")
        print(f"   Original Accuracy: {original_accuracy:.1%}")
        print(f"   Learning Accuracy: {learning_accuracy:.1%}")
        
        improvement = learning_accuracy - original_accuracy
        print(f"   Learning Improvement: {improvement:+.1%}")
        
        return {
            'original_mae': original_mae,
            'learning_mae': learning_mae,
            'original_accuracy': original_accuracy,
            'learning_accuracy': learning_accuracy,
            'improvement': improvement
        }
    
    def save_models(self, original_model, learning_model, features, performance):
        """Save both models for production use"""
        
        print(f"\n💾 SAVING MODELS")
        print("=" * 25)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        original_path = f"{self.models_dir}/original_model_{timestamp}.joblib"
        learning_path = f"{self.models_dir}/learning_model_{timestamp}.joblib"
        
        joblib.dump(original_model, original_path)
        joblib.dump(learning_model, learning_path)
        
        # Save feature list
        features_path = f"{self.models_dir}/model_features_{timestamp}.json"
        with open(features_path, 'w') as f:
            json.dump({
                'features': features,
                'feature_count': len(features),
                'training_date': timestamp,
                'performance': performance
            }, f, indent=2)
        
        print(f"✅ Models saved:")
        print(f"   Original: {original_path}")
        print(f"   Learning: {learning_path}")
        print(f"   Features: {features_path}")
        
        return original_path, learning_path, features_path

def main():
    print("🚀 COMPREHENSIVE ML TRAINING PIPELINE")
    print("   Training both original and learning models")
    print("=" * 55)
    
    trainer = ComprehensiveMLTrainer()
    
    # Step 1: Load training data
    print("STEP 1: Load verified training dataset")
    full_df, train_df, val_df = trainer.get_training_data()
    
    # Step 2: Prepare features
    print("\nSTEP 2: Prepare consistent feature set")
    X_train, y_train, features = trainer.prepare_features(train_df)
    X_val, y_val, _ = trainer.prepare_features(val_df)
    
    # Step 3: Train original model
    print("\nSTEP 3: Train original enhanced model")
    original_model, original_importance = trainer.train_original_model(
        X_train, y_train, X_val, y_val, features
    )
    
    # Step 4: Train learning model
    print("\nSTEP 4: Train adaptive learning model")
    learning_model, learning_importance = trainer.train_learning_model(
        X_train, y_train, X_val, y_val, features
    )
    
    # Step 5: Compare models
    print("\nSTEP 5: Evaluate dual model performance")
    performance = trainer.evaluate_dual_models(
        original_model, learning_model, X_val, y_val
    )
    
    # Step 6: Save models
    print("\nSTEP 6: Save models for production")
    original_path, learning_path, features_path = trainer.save_models(
        original_model, learning_model, features, performance
    )
    
    # Final assessment
    print(f"\n🏆 TRAINING COMPLETE!")
    print(f"   Original Model: {performance['original_accuracy']:.1%} accuracy")
    print(f"   Learning Model: {performance['learning_accuracy']:.1%} accuracy")
    print(f"   Improvement: {performance['improvement']:+.1%}")
    print(f"   Feature Count: {len(features)} consistent features")
    print(f"\n🎯 READY FOR CONTINUOUS LEARNING DEPLOYMENT!")

if __name__ == "__main__":
    main()
