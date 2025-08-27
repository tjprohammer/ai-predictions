#!/usr/bin/env python3
"""
Create a clean learning model without data leaks
This model will only use pre-game features available at prediction time
"""

import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime, timedelta
import joblib
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CleanLearningModelTrainer:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        
        # Define PRE-GAME ONLY features (no data leaks)
        self.clean_features = [
            # Weather
            'temperature', 'humidity', 'wind_speed', 'wind_direction_deg', 'air_pressure',
            
            # Pitcher season stats (available pre-game)
            'home_sp_season_era', 'away_sp_season_era',
            'home_sp_whip', 'away_sp_whip',
            'home_sp_days_rest', 'away_sp_days_rest',
            
            # Team season averages (available pre-game)
            'home_team_avg', 'away_team_avg',
            'home_team_obp', 'away_team_obp',
            'home_team_slg', 'away_team_slg',
            'home_team_ops', 'away_team_ops',
            
            # Ballpark factors
            'ballpark_run_factor', 'ballpark_hr_factor',
            
            # Umpire tendencies
            'umpire_ou_tendency',
            
            # Game context
            'series_game', 'getaway_day', 'doubleheader', 'day_after_night',
            
            # Market info
            'market_total'
        ]
        
    def get_clean_training_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical games with only pre-game features"""
        
        conn = psycopg2.connect(**self.db_config)
        
        query = """
        SELECT *
        FROM enhanced_games
        WHERE date >= %s AND date <= %s
        AND total_runs IS NOT NULL
        AND total_runs > 3 AND total_runs < 20
        AND market_total IS NOT NULL
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        conn.close()
        
        print(f"✅ Retrieved {len(df)} clean training games")
        return df
        
    def prepare_clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare only pre-game features without data leaks"""
        
        print("🔧 Preparing clean pre-game features...")
        
        # Start with base features
        feature_df = df.copy()
        
        # Add derived features that don't leak
        feature_df['combined_era'] = (feature_df['home_sp_season_era'] + feature_df['away_sp_season_era']) / 2
        feature_df['era_differential'] = feature_df['home_sp_season_era'] - feature_df['away_sp_season_era']
        feature_df['combined_whip'] = (feature_df['home_sp_whip'] + feature_df['away_sp_whip']) / 2
        feature_df['combined_team_ops'] = (feature_df['home_team_ops'] + feature_df['away_team_ops']) / 2
        feature_df['offensive_environment'] = feature_df['combined_team_ops'] * feature_df['ballpark_run_factor']
        
        # Weather factors
        feature_df['temp_factor'] = np.where(feature_df['temperature'] > 80, 1.05, 
                                   np.where(feature_df['temperature'] < 60, 0.95, 1.0))
        feature_df['wind_factor'] = np.where(feature_df['wind_speed'] > 15, 1.03, 1.0)
        
        # Categorical encoding
        feature_df['day_night_encoded'] = feature_df['day_night'].map({'D': 1, 'N': 0}).fillna(0)
        feature_df['home_sp_hand_R'] = (feature_df['home_sp_hand'] == 'R').astype(int)
        feature_df['away_sp_hand_R'] = (feature_df['away_sp_hand'] == 'R').astype(int)
        
        # Update clean features list with derived features
        self.clean_features.extend([
            'combined_era', 'era_differential', 'combined_whip', 'combined_team_ops',
            'offensive_environment', 'temp_factor', 'wind_factor', 
            'day_night_encoded', 'home_sp_hand_R', 'away_sp_hand_R'
        ])
        
        # Fill missing values with reasonable defaults
        fill_values = {
            'temperature': 72,
            'humidity': 50,
            'wind_speed': 8,
            'wind_direction_deg': 180,
            'air_pressure': 30.0,
            'home_sp_season_era': 4.5,
            'away_sp_season_era': 4.5,
            'home_sp_whip': 1.3,
            'away_sp_whip': 1.3,
            'home_sp_days_rest': 4,
            'away_sp_days_rest': 4,
            'home_team_avg': 0.250,
            'away_team_avg': 0.250,
            'home_team_obp': 0.320,
            'away_team_obp': 0.320,
            'home_team_slg': 0.400,
            'away_team_slg': 0.400,
            'home_team_ops': 0.720,
            'away_team_ops': 0.720,
            'ballpark_run_factor': 1.0,
            'ballpark_hr_factor': 1.0,
            'umpire_ou_tendency': 0.0,
            'series_game': 1,
            'getaway_day': 0,
            'doubleheader': 0,
            'day_after_night': 0,
            'market_total': 8.5
        }
        
        for feature in self.clean_features:
            if feature in feature_df.columns:
                feature_df[feature] = feature_df[feature].fillna(fill_values.get(feature, 0))
        
        # Filter to only available features
        available_features = [f for f in self.clean_features if f in feature_df.columns]
        print(f"📊 Using {len(available_features)} clean features")
        
        return feature_df[available_features], available_features
        
    def train_clean_model(self, start_date: str = None, end_date: str = None):
        """Train a clean learning model without data leaks"""
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
        print(f"🚀 CLEAN LEARNING MODEL TRAINING")
        print("=" * 60)
        print(f"📅 Training period: {start_date} to {end_date}")
        
        # Get training data
        df = self.get_clean_training_data(start_date, end_date)
        
        # Prepare features
        X, feature_names = self.prepare_clean_features(df)
        y = df['total_runs'].values
        
        print(f"📊 Training data shape: {X.shape}")
        print(f"📊 Target range: {y.min():.1f} - {y.max():.1f} runs")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        print(f"   Training set: {len(X_train)} games")
        print(f"   Test set: {len(X_test)} games")
        
        # Train model
        print("🎯 Training clean learning model...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
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
        
        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
        cv_mae = -cv_scores.mean()
        
        print(f"📈 MODEL PERFORMANCE:")
        print(f"   Training MAE: {train_mae:.3f}")
        print(f"   Test MAE: {test_mae:.3f}")
        print(f"   Training R²: {train_r2:.3f}")
        print(f"   Test R²: {test_r2:.3f}")
        print(f"   CV MAE: {cv_mae:.3f} ± {cv_scores.std():.3f}")
        
        # Feature importance
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        for i, (feature, importance) in enumerate(feature_importance[:10]):
            print(f"   {i+1:2d}. {feature:25s} ({importance:.3f})")
        
        # Save model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"clean_learning_model_{timestamp}.joblib"
        model_path = os.path.join('mlb-overs', 'models', model_name)
        
        # Create model package
        model_package = {
            'model': model,
            'feature_columns': feature_names,
            'feature_fill_values': {f: 0 for f in feature_names},  # Simple defaults
            'model_type': 'clean_random_forest',
            'training_metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_mae': cv_mae,
                'feature_names': feature_names
            },
            'created_date': datetime.now().isoformat(),
            'training_period': {'start': start_date, 'end': end_date},
            'note': 'Clean model without data leaks - only pre-game features'
        }
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(model_package, model_path)
        
        print(f"\\n💾 Model saved to: {model_path}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Test MAE: {test_mae:.3f}")
        
        print(f"\\n✅ CLEAN MODEL TRAINING COMPLETE!")
        print(f"   Model path: {model_path}")
        print(f"   Training games: {len(df)}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Test MAE: {test_mae:.3f}")
        
        return model_path

if __name__ == "__main__":
    trainer = CleanLearningModelTrainer()
    model_path = trainer.train_clean_model()
    
    print(f"\\n🎉 SUCCESS! Clean model available at:")
    print(f"   {model_path}")
    print(f"\\nTo deploy this model, run:")
    print(f"   cp {model_path} mlb-overs/models/legitimate_model_latest.joblib")
