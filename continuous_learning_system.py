#!/usr/bin/env python3
"""
Continuous Learning System for MLB Predictions
Implements daily model retraining based on game results
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import warnings
warnings.filterwarnings('ignore')

class ContinuousLearningSystem:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser', 
            'password': 'mlbpass'
        }
        self.models_dir = 'models/daily_learning'
        self.learning_log_file = 'daily_learning_log.json'
        
        # Create models directory
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Load or initialize learning log
        self.learning_log = self.load_learning_log()
        
    def load_learning_log(self):
        """Load or create learning log"""
        if os.path.exists(self.learning_log_file):
            with open(self.learning_log_file, 'r') as f:
                return json.load(f)
        return {
            'daily_performance': {},
            'model_updates': {},
            'feature_importance_evolution': {},
            'learning_metrics': {}
        }
    
    def save_learning_log(self):
        """Save learning log"""
        with open(self.learning_log_file, 'w') as f:
            json.dump(self.learning_log, f, indent=2, default=str)
    
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def get_training_data(self, start_date, end_date, min_games=50):
        """Get training data for a date range"""
        
        conn = self.get_db_connection()
        
        # Feature selection query - comprehensive features for learning
        query = """
        SELECT 
            date,
            game_id,
            home_team,
            away_team,
            total_runs,
            market_total,
            
            -- Weather features
            temperature,
            humidity,
            wind_speed,
            wind_direction_deg,
            air_pressure,
            
            -- Venue features  
            venue_name,
            ballpark_run_factor,
            ballpark_hr_factor,
            roof_type,
            
            -- Pitcher features
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
            
            -- Game context
            day_night,
            series_game,
            getaway_day,
            doubleheader,
            
            -- Umpire
            umpire_ou_tendency
            
        FROM enhanced_games 
        WHERE date >= %s 
        AND date <= %s
        AND total_runs IS NOT NULL
        AND predicted_total IS NOT NULL
        ORDER BY date
        """
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        if len(df) < min_games:
            print(f"⚠️  Not enough games ({len(df)}) for training (minimum: {min_games})")
            return None
            
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        
        # Create feature matrix
        features = []
        feature_names = []
        
        # Weather features
        features.append(df['temperature'].fillna(70))
        feature_names.append('temperature')
        
        features.append(df['humidity'].fillna(50))
        feature_names.append('humidity')
        
        features.append(df['wind_speed'].fillna(5))
        feature_names.append('wind_speed')
        
        features.append(df['air_pressure'].fillna(30.0))
        feature_names.append('air_pressure')
        
        # Pitcher features
        features.append(df['home_sp_season_era'].fillna(4.5))
        feature_names.append('home_sp_era')
        
        features.append(df['away_sp_season_era'].fillna(4.5))
        feature_names.append('away_sp_era')
        
        features.append(df['home_sp_whip'].fillna(1.3))
        feature_names.append('home_sp_whip')
        
        features.append(df['away_sp_whip'].fillna(1.3))
        feature_names.append('away_sp_whip')
        
        features.append(df['home_sp_days_rest'].fillna(4))
        feature_names.append('home_sp_rest')
        
        features.append(df['away_sp_days_rest'].fillna(4))
        feature_names.append('away_sp_rest')
        
        # Team features
        features.append(df['home_team_avg'].fillna(0.250))
        feature_names.append('home_team_avg')
        
        features.append(df['away_team_avg'].fillna(0.250))
        feature_names.append('away_team_avg')
        
        # Ballpark features
        features.append(df['ballpark_run_factor'].fillna(1.0))
        feature_names.append('ballpark_run_factor')
        
        features.append(df['ballpark_hr_factor'].fillna(1.0))
        feature_names.append('ballpark_hr_factor')
        
        # Umpire
        features.append(df['umpire_ou_tendency'].fillna(0.0))
        feature_names.append('umpire_tendency')
        
        # Binary features
        features.append((df['day_night'] == 'N').astype(int))
        feature_names.append('is_night_game')
        
        features.append(df['getaway_day'].fillna(False).astype(int))
        feature_names.append('is_getaway_day')
        
        features.append(df['doubleheader'].fillna(False).astype(int))
        feature_names.append('is_doubleheader')
        
        # Pitching handedness combinations
        left_vs_left = ((df['home_sp_hand'] == 'L') & (df['away_sp_hand'] == 'L')).astype(int)
        right_vs_right = ((df['home_sp_hand'] == 'R') & (df['away_sp_hand'] == 'R')).astype(int)
        
        features.append(left_vs_left)
        feature_names.append('both_lefties')
        
        features.append(right_vs_right)
        feature_names.append('both_righties')
        
        # Market baseline
        features.append(df['market_total'].fillna(8.5))
        feature_names.append('market_total')
        
        # Convert to numpy array
        X = np.column_stack(features)
        y = df['total_runs'].values
        
        return X, y, feature_names
    
    def train_ensemble_models(self, X, y, feature_names):
        """Train ensemble of models"""
        
        models = {}
        
        # Random Forest - good for feature interactions
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            random_state=42
        )
        rf.fit(X, y)
        models['random_forest'] = rf
        
        # Gradient Boosting - good for sequential learning
        gb = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb.fit(X, y)
        models['gradient_boosting'] = gb
        
        # Linear model - fast and interpretable
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LinearRegression()
        lr.fit(X_scaled, y)
        models['linear'] = {'model': lr, 'scaler': scaler}
        
        return models
    
    def evaluate_models(self, models, X, y):
        """Evaluate model performance"""
        
        results = {}
        
        for name, model in models.items():
            if name == 'linear':
                X_eval = model['scaler'].transform(X)
                y_pred = model['model'].predict(X_eval)
            else:
                y_pred = model.predict(X)
            
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'predictions': y_pred
            }
        
        return results
    
    def get_feature_importance(self, models, feature_names):
        """Get feature importance from models"""
        
        importance = {}
        
        # Random Forest importance
        if 'random_forest' in models:
            rf_importance = models['random_forest'].feature_importances_
            importance['random_forest'] = dict(zip(feature_names, rf_importance))
        
        # Gradient Boosting importance
        if 'gradient_boosting' in models:
            gb_importance = models['gradient_boosting'].feature_importances_
            importance['gradient_boosting'] = dict(zip(feature_names, gb_importance))
        
        return importance
    
    def daily_learning_update(self, target_date):
        """Perform daily learning update for a specific date"""
        
        print(f"🎯 DAILY LEARNING UPDATE: {target_date}")
        print("=" * 60)
        
        # Calculate training window (last 60 days before target date)
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        training_start = target_dt - timedelta(days=60)
        training_end = target_dt - timedelta(days=1)
        
        print(f"Training window: {training_start.date()} to {training_end.date()}")
        
        # Get training data
        df_train = self.get_training_data(training_start.date(), training_end.date())
        if df_train is None:
            return False
        
        print(f"Training games: {len(df_train)}")
        
        # Prepare features
        X_train, y_train, feature_names = self.prepare_features(df_train)
        
        # Train models
        print("Training ensemble models...")
        models = self.train_ensemble_models(X_train, y_train, feature_names)
        
        # Evaluate on training data
        train_results = self.evaluate_models(models, X_train, y_train)
        
        # Get feature importance
        importance = self.get_feature_importance(models, feature_names)
        
        # Test on target date if data available
        df_test = self.get_training_data(target_date, target_date, min_games=1)
        test_results = None
        
        if df_test is not None and len(df_test) > 0:
            X_test, y_test, _ = self.prepare_features(df_test)
            test_results = self.evaluate_models(models, X_test, y_test)
            
            print(f"\\n📊 PERFORMANCE ON {target_date}:")
            for model_name, result in test_results.items():
                print(f"  {model_name}: MAE={result['mae']:.2f}, RMSE={result['rmse']:.2f}")
        
        # Save models
        model_file = os.path.join(self.models_dir, f"models_{target_date}.joblib")
        joblib.dump({
            'models': models,
            'feature_names': feature_names,
            'training_period': f"{training_start.date()}_to_{training_end.date()}",
            'train_performance': train_results,
            'test_performance': test_results,
            'feature_importance': importance
        }, model_file)
        
        # Update learning log
        self.learning_log['daily_performance'][target_date] = {
            'training_games': len(df_train),
            'test_games': len(df_test) if df_test is not None else 0,
            'train_performance': {k: {'mae': v['mae'], 'rmse': v['rmse']} for k, v in train_results.items()},
            'test_performance': {k: {'mae': v['mae'], 'rmse': v['rmse']} for k, v in test_results.items()} if test_results else None
        }
        
        self.learning_log['feature_importance_evolution'][target_date] = importance
        
        self.save_learning_log()
        
        print(f"✅ Models saved to: {model_file}")
        return True
    
    def batch_learning_update(self, start_date, end_date):
        """Perform learning updates for a date range"""
        
        print(f"🚀 BATCH LEARNING UPDATE: {start_date} to {end_date}")
        print("=" * 70)
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_dt = start_dt
        success_count = 0
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            try:
                if self.daily_learning_update(date_str):
                    success_count += 1
                print()
                
            except Exception as e:
                print(f"❌ Error updating {date_str}: {e}")
            
            current_dt += timedelta(days=1)
        
        print(f"✅ BATCH COMPLETE: {success_count} successful updates")
        self.analyze_learning_progress()
    
    def analyze_learning_progress(self):
        """Analyze learning progress over time"""
        
        print("\\n📈 LEARNING PROGRESS ANALYSIS")
        print("=" * 50)
        
        daily_perf = self.learning_log['daily_performance']
        dates = sorted(daily_perf.keys())
        
        if len(dates) < 2:
            print("Not enough data for progress analysis")
            return
        
        # Analyze performance trends
        rf_maes = []
        gb_maes = []
        linear_maes = []
        
        for date in dates:
            perf = daily_perf[date]
            if perf['test_performance']:
                if 'random_forest' in perf['test_performance']:
                    rf_maes.append(perf['test_performance']['random_forest']['mae'])
                if 'gradient_boosting' in perf['test_performance']:
                    gb_maes.append(perf['test_performance']['gradient_boosting']['mae'])
                if 'linear' in perf['test_performance']:
                    linear_maes.append(perf['test_performance']['linear']['mae'])
        
        if rf_maes:
            print(f"Random Forest MAE trend: {rf_maes[0]:.2f} → {rf_maes[-1]:.2f} (Δ{rf_maes[-1]-rf_maes[0]:+.2f})")
        if gb_maes:
            print(f"Gradient Boosting MAE trend: {gb_maes[0]:.2f} → {gb_maes[-1]:.2f} (Δ{gb_maes[-1]-gb_maes[0]:+.2f})")
        if linear_maes:
            print(f"Linear Model MAE trend: {linear_maes[0]:.2f} → {linear_maes[-1]:.2f} (Δ{linear_maes[-1]-linear_maes[0]:+.2f})")
        
        # Feature importance evolution
        print("\\n🎯 TOP FEATURE TRENDS:")
        if len(dates) >= 2:
            early_importance = self.learning_log['feature_importance_evolution'].get(dates[0], {})
            late_importance = self.learning_log['feature_importance_evolution'].get(dates[-1], {})
            
            if 'random_forest' in early_importance and 'random_forest' in late_importance:
                early_rf = early_importance['random_forest']
                late_rf = late_importance['random_forest']
                
                # Show top features and their evolution
                top_features = sorted(late_rf.items(), key=lambda x: x[1], reverse=True)[:5]
                
                for feature, late_imp in top_features:
                    early_imp = early_rf.get(feature, 0)
                    change = late_imp - early_imp
                    print(f"  {feature}: {early_imp:.3f} → {late_imp:.3f} (Δ{change:+.3f})")

def main():
    """Main execution"""
    
    learner = ContinuousLearningSystem()
    
    print("🤖 MLB CONTINUOUS LEARNING SYSTEM")
    print("=" * 50)
    print("This system implements daily model learning from game results")
    print()
    
    # Example: Learn from recent games
    print("🎯 LEARNING FROM RECENT HISTORICAL DATA...")
    
    # Start learning from when we have good data
    learner.batch_learning_update('2025-08-01', '2025-08-20')

if __name__ == "__main__":
    main()
