#!/usr/bin/env python3
"""
10-Session Continuous Learning System for MLB Predictions
Implements iterative learning with 120-day training windows,
tracks prediction outcomes, and shows improvement over 10 sessions
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings('ignore')

class TenSessionLearningSystem:
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser', 
            'password': 'mlbpass'
        }
        
        # Create directories for learning models
        self.learning_models_dir = 'models/learning_sessions'
        self.session_logs_dir = 'session_logs'
        os.makedirs(self.learning_models_dir, exist_ok=True)
        os.makedirs(self.session_logs_dir, exist_ok=True)
        
        # Initialize learning session tracking
        self.session_results = {
            'sessions': [],
            'performance_evolution': [],
            'feature_importance_evolution': [],
            'prediction_accuracy_by_session': []
        }
        
        print("🎯 10-SESSION CONTINUOUS LEARNING SYSTEM INITIALIZED")
        print("=" * 70)
        
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(**self.db_config)
    
    def get_training_data(self, end_date, days=120):
        """Get 120 days of training data ending on specified date"""
        
        start_date = (datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days)).strftime('%Y-%m-%d')
        
        conn = self.get_db_connection()
        
        # Get comprehensive training data with outcomes
        query = """
        SELECT 
            date, game_id, home_team, away_team,
            home_score, away_score, total_runs,
            
            -- Weather and venue
            temperature, humidity, wind_speed, air_pressure,
            ballpark_run_factor, ballpark_hr_factor,
            
            -- Pitching
            home_sp_season_era, away_sp_season_era,
            home_sp_whip, away_sp_whip,
            home_sp_days_rest, away_sp_days_rest,
            home_sp_k, away_sp_k, home_sp_bb, away_sp_bb,
            
            -- Team batting
            home_team_avg, away_team_avg,
            home_team_obp, away_team_obp,
            home_team_ops, away_team_ops,
            
            -- Umpire
            umpire_ou_tendency, plate_umpire,
            
            -- Game context
            day_night, getaway_day, doubleheader,
            home_sp_hand, away_sp_hand,
            
            -- Market data
            market_total, over_odds, under_odds,
            predicted_total, confidence, recommendation, edge,
            
            -- Enhanced sophisticated features
            home_team_defensive_efficiency, away_team_defensive_efficiency,
            home_team_bullpen_fatigue_score, away_team_bullpen_fatigue_score,
            home_team_weighted_runs_scored, away_team_weighted_runs_scored,
            home_team_xrv_differential, away_team_xrv_differential,
            home_team_clutch_factor, away_team_clutch_factor,
            home_team_offensive_efficiency, away_team_offensive_efficiency
            
        FROM enhanced_games 
        WHERE date BETWEEN %s AND %s
        AND home_score IS NOT NULL 
        AND away_score IS NOT NULL
        AND total_runs IS NOT NULL
        ORDER BY date, game_id
        """
        
        df = pd.read_sql_query(query, conn, params=[start_date, end_date])
        conn.close()
        
        print(f"📊 Loaded {len(df)} games from {start_date} to {end_date}")
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        
        # Select and prepare numeric features
        feature_cols = [
            'temperature', 'humidity', 'wind_speed', 'air_pressure',
            'ballpark_run_factor', 'ballpark_hr_factor',
            'home_sp_season_era', 'away_sp_season_era',
            'home_sp_whip', 'away_sp_whip',
            'home_sp_days_rest', 'away_sp_days_rest',
            'home_sp_k', 'away_sp_k', 'home_sp_bb', 'away_sp_bb',
            'home_team_avg', 'away_team_avg',
            'home_team_obp', 'away_team_obp',
            'home_team_ops', 'away_team_ops',
            'umpire_ou_tendency', 'market_total',
            'home_team_defensive_efficiency', 'away_team_defensive_efficiency',
            'home_team_bullpen_fatigue_score', 'away_team_bullpen_fatigue_score',
            'home_team_weighted_runs_scored', 'away_team_weighted_runs_scored',
            'home_team_xrv_differential', 'away_team_xrv_differential',
            'home_team_clutch_factor', 'away_team_clutch_factor',
            'home_team_offensive_efficiency', 'away_team_offensive_efficiency'
        ]
        
        # Create categorical features
        df['is_night'] = (df['day_night'] == 'N').astype(int)
        df['is_getaway'] = df['getaway_day'].astype(int)
        df['is_doubleheader'] = df['doubleheader'].astype(int)
        df['both_lefties'] = ((df['home_sp_hand'] == 'L') & (df['away_sp_hand'] == 'L')).astype(int)
        df['both_righties'] = ((df['home_sp_hand'] == 'R') & (df['away_sp_hand'] == 'R')).astype(int)
        
        feature_cols.extend(['is_night', 'is_getaway', 'is_doubleheader', 'both_lefties', 'both_righties'])
        
        # Fill missing values with defaults for each feature type
        X = df[feature_cols].copy()
        
        # Weather defaults
        X['temperature'].fillna(72, inplace=True)
        X['humidity'].fillna(50, inplace=True)
        X['wind_speed'].fillna(5, inplace=True)
        X['air_pressure'].fillna(30.0, inplace=True)
        
        # Ballpark defaults
        X['ballpark_run_factor'].fillna(1.0, inplace=True)
        X['ballpark_hr_factor'].fillna(1.0, inplace=True)
        
        # Pitching defaults
        for col in ['home_sp_season_era', 'away_sp_season_era']:
            X[col].fillna(4.50, inplace=True)
        for col in ['home_sp_whip', 'away_sp_whip']:
            X[col].fillna(1.30, inplace=True)
        for col in ['home_sp_days_rest', 'away_sp_days_rest']:
            X[col].fillna(4, inplace=True)
        for col in ['home_sp_k', 'away_sp_k', 'home_sp_bb', 'away_sp_bb']:
            X[col].fillna(X[col].median(), inplace=True)
        
        # Team batting defaults
        for col in ['home_team_avg', 'away_team_avg']:
            X[col].fillna(0.250, inplace=True)
        for col in ['home_team_obp', 'away_team_obp']:
            X[col].fillna(0.320, inplace=True)
        for col in ['home_team_ops', 'away_team_ops']:
            X[col].fillna(0.750, inplace=True)
        
        # Umpire defaults
        X['umpire_ou_tendency'].fillna(0.0, inplace=True)
        X['market_total'].fillna(8.5, inplace=True)
        
        # Advanced features - use median or defaults
        advanced_features = [col for col in feature_cols if 'team_' in col and ('efficiency' in col or 'fatigue' in col or 'weighted' in col or 'xrv' in col or 'clutch' in col)]
        for col in advanced_features:
            if col in X.columns:
                X[col].fillna(X[col].median(), inplace=True)
        
        # Final check - fill any remaining NaNs with column median
        for col in X.columns:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        
        y = df['total_runs']
        
        # Verify no NaNs remain
        assert not X.isnull().any().any(), "NaN values still present after preprocessing"
        
        return X, y, feature_cols
    
    def train_learning_model(self, X_train, y_train, session_num):
        """Train a learning model for this session"""
        
        print(f"🧠 Training Session {session_num} Model...")
        
        # Use Gradient Boosting for learning (adapts well to new data)
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            random_state=42 + session_num,  # Vary randomness per session
            subsample=0.8
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Save the session model
        model_path = os.path.join(self.learning_models_dir, f'session_{session_num}_model.joblib')
        joblib.dump(model, model_path)
        
        return model
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'predictions': predictions
        }
    
    def run_single_learning_session(self, session_num, end_date):
        """Run a single learning session"""
        
        print(f"\n🎯 LEARNING SESSION {session_num}")
        print("=" * 50)
        print(f"📅 Training End Date: {end_date}")
        
        # Get 120 days of training data
        df = self.get_training_data(end_date, days=120)
        
        if len(df) < 100:
            print(f"❌ Insufficient data ({len(df)} games)")
            return None
        
        # Prepare features and targets
        X, y, feature_cols = self.prepare_features(df)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"📊 Training: {len(X_train)} games, Testing: {len(X_test)} games")
        
        # Train learning model
        model = self.train_learning_model(X_train, y_train, session_num)
        
        # Evaluate performance
        train_metrics = self.evaluate_model(model, X_train, y_train)
        test_metrics = self.evaluate_model(model, X_test, y_test)
        
        # Get feature importance
        feature_importance = dict(zip(feature_cols, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Store session results
        session_result = {
            'session': session_num,
            'end_date': end_date,
            'training_games': len(X_train),
            'test_games': len(X_test),
            'train_mae': train_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'test_r2': test_metrics['r2'],
            'top_features': top_features[:5],
            'all_feature_importance': feature_importance
        }
        
        # Display results
        print(f"📈 PERFORMANCE METRICS:")
        print(f"   Training MAE: {train_metrics['mae']:.3f} runs")
        print(f"   Test MAE: {test_metrics['mae']:.3f} runs")
        print(f"   Training R²: {train_metrics['r2']:.3f}")
        print(f"   Test R²: {test_metrics['r2']:.3f}")
        
        print(f"\n🔍 TOP 5 FEATURES:")
        for i, (feature, importance) in enumerate(top_features[:5], 1):
            print(f"   {i}. {feature}: {importance:.4f}")
        
        return session_result
    
    def analyze_learning_progression(self):
        """Analyze how the model improved across sessions"""
        
        print(f"\n📊 LEARNING PROGRESSION ANALYSIS")
        print("=" * 60)
        
        if len(self.session_results['sessions']) < 2:
            print("Need at least 2 sessions for progression analysis")
            return
        
        sessions = self.session_results['sessions']
        
        # Track improvement metrics
        print(f"Session | Train MAE | Test MAE  | Train R²  | Test R²   | Improvement")
        print("-" * 70)
        
        for i, session in enumerate(sessions):
            improvement = ""
            if i > 0:
                prev_test_mae = sessions[i-1]['test_mae']
                curr_test_mae = session['test_mae']
                mae_change = prev_test_mae - curr_test_mae
                
                if mae_change > 0:
                    improvement = f"↑ -{mae_change:.3f}"
                elif mae_change < 0:
                    improvement = f"↓ +{abs(mae_change):.3f}"
                else:
                    improvement = "→ same"
            
            print(f"   {session['session']:2}   | {session['train_mae']:8.3f} | {session['test_mae']:8.3f} | {session['train_r2']:8.3f} | {session['test_r2']:8.3f} | {improvement}")
        
        # Best session
        best_session = min(sessions, key=lambda x: x['test_mae'])
        print(f"\n🏆 BEST SESSION: #{best_session['session']}")
        print(f"   Test MAE: {best_session['test_mae']:.3f} runs")
        print(f"   Test R²: {best_session['test_r2']:.3f}")
        
        # Feature evolution analysis
        print(f"\n🔍 FEATURE IMPORTANCE EVOLUTION:")
        print("-" * 50)
        
        # Track how top features changed across sessions
        all_features = set()
        for session in sessions:
            all_features.update([f[0] for f in session['top_features']])
        
        for feature in list(all_features)[:8]:  # Show top 8 features
            importances = []
            for session in sessions:
                importance = session['all_feature_importance'].get(feature, 0)
                importances.append(importance)
            
            avg_importance = np.mean(importances)
            trend = "📈" if importances[-1] > importances[0] else "📉" if importances[-1] < importances[0] else "➡️"
            
            print(f"   {feature[:30]:30} | Avg: {avg_importance:.4f} | {trend}")
    
    def run_ten_session_learning(self):
        """Run the complete 10-session learning system"""
        
        print("🚀 STARTING 10-SESSION CONTINUOUS LEARNING SYSTEM")
        print("=" * 70)
        
        # Define session end dates (working backwards from 2025-08-21)
        base_date = datetime.strptime('2025-08-21', '%Y-%m-%d')
        session_dates = []
        
        for i in range(10):
            # Each session uses data ending 5 days before the previous
            session_end = base_date - timedelta(days=i * 5)
            session_dates.append(session_end.strftime('%Y-%m-%d'))
        
        session_dates.reverse()  # Start with earliest, progress to latest
        
        print(f"📅 Session dates: {session_dates[0]} → {session_dates[-1]}")
        print()
        
        # Run each learning session
        for i, end_date in enumerate(session_dates, 1):
            session_result = self.run_single_learning_session(i, end_date)
            
            if session_result:
                self.session_results['sessions'].append(session_result)
            
            # Save progress after each session
            self.save_session_results()
        
        # Analyze overall learning progression
        self.analyze_learning_progression()
        
        # Save final results
        self.save_session_results()
        
        print(f"\n🎉 10-SESSION LEARNING COMPLETE!")
        print(f"📁 Results saved to: {self.session_logs_dir}/ten_session_results.json")
        
        return self.session_results
    
    def save_session_results(self):
        """Save session results to file"""
        
        results_file = os.path.join(self.session_logs_dir, 'ten_session_results.json')
        
        # Convert numpy types to native Python for JSON serialization
        serializable_results = {}
        for key, value in self.session_results.items():
            if isinstance(value, list):
                serializable_results[key] = []
                for item in value:
                    if isinstance(item, dict):
                        serialized_item = {}
                        for k, v in item.items():
                            if isinstance(v, np.ndarray):
                                serialized_item[k] = v.tolist()
                            elif isinstance(v, (np.float64, np.float32)):
                                serialized_item[k] = float(v)
                            elif isinstance(v, (np.int64, np.int32)):
                                serialized_item[k] = int(v)
                            else:
                                serialized_item[k] = v
                        serializable_results[key].append(serialized_item)
                    else:
                        serializable_results[key].append(item)
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def predict_with_best_model(self, game_features):
        """Make predictions using the best learning model"""
        
        if not self.session_results['sessions']:
            print("No trained sessions available")
            return None
        
        # Find best session (lowest test MAE)
        best_session = min(self.session_results['sessions'], key=lambda x: x['test_mae'])
        best_session_num = best_session['session']
        
        # Load the best model
        model_path = os.path.join(self.learning_models_dir, f'session_{best_session_num}_model.joblib')
        
        if not os.path.exists(model_path):
            print(f"Best model file not found: {model_path}")
            return None
        
        model = joblib.load(model_path)
        prediction = model.predict([game_features])[0]
        
        return {
            'predicted_total': prediction,
            'model_session': best_session_num,
            'model_test_mae': best_session['test_mae'],
            'model_test_r2': best_session['test_r2']
        }

def main():
    """Run the 10-session learning system"""
    
    learning_system = TenSessionLearningSystem()
    results = learning_system.run_ten_session_learning()
    
    return results

if __name__ == "__main__":
    results = main()
