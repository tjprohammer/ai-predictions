#!/usr/bin/env python3
"""
Model Deep Dive Analysis
========================
Detailed analysis of your currently trained model
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class ModelDeepDive:
    def __init__(self):
        self.model_path = "S:/Projects/AI_Predictions/mlb-overs/training/daily_model.joblib"
        self.data_path = "S:/Projects/AI_Predictions/mlb-overs/data/enhanced_historical_games_2025.parquet"
        
    def load_and_analyze_model(self):
        """Load and thoroughly analyze the model"""
        print("🤖 DEEP MODEL ANALYSIS")
        print("=" * 60)
        
        # Load model
        print("📂 Loading model...")
        try:
            self.model_data = joblib.load(self.model_path)
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return
        
        # Analyze model structure
        print(f"\n🏗️ Model Structure:")
        print(f"   Type: {type(self.model_data)}")
        
        if isinstance(self.model_data, dict):
            print(f"   Components:")
            for key, value in self.model_data.items():
                print(f"     {key}: {type(value).__name__}")
        
        # Get the actual model
        model = self.model_data.get('model')
        feature_columns = self.model_data.get('feature_columns', [])
        performance = self.model_data.get('performance', {})
        
        print(f"\n📊 Model Details:")
        print(f"   Algorithm: {model.__class__.__name__}")
        print(f"   Features: {len(feature_columns)}")
        if performance:
            print(f"   Training Performance:")
            for metric, value in performance.items():
                if isinstance(value, (int, float)):
                    print(f"     {metric}: {value:.4f}")
                else:
                    print(f"     {metric}: {value}")
        
        return True
    
    def analyze_feature_importance(self):
        """Analyze which features are most important"""
        print(f"\n🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)
        
        model = self.model_data.get('model')
        feature_columns = self.model_data.get('feature_columns', [])
        
        if model and hasattr(model, 'feature_importances_') and feature_columns:
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"📈 TOP 20 MOST IMPORTANT FEATURES:")
            print(f"{'Rank':<4} {'Feature':<30} {'Importance':<12} {'Percentage'}")
            print("-" * 60)
            
            total_importance = importance_df['importance'].sum()
            
            for i, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
                percentage = (row['importance'] / total_importance) * 100
                print(f"{i:2d}.  {row['feature']:<30} {row['importance']:<12.6f} {percentage:6.2f}%")
            
            # Analyze feature categories
            print(f"\n📊 FEATURE CATEGORY ANALYSIS:")
            categories = {
                'Pitcher Stats': ['era', 'whip', 'k_rate', 'bb_rate', 'pitcher'],
                'Team Offense': ['rbi', 'hits', 'runs', 'ba', 'ops', 'offensive'],
                'Weather': ['temperature', 'wind', 'weather'],
                'Ballpark': ['venue', 'park', 'day_night'],
                'Market Data': ['market', 'total', 'line'],
                'Game Context': ['rest', 'series', 'streak', 'lob']
            }
            
            for category, keywords in categories.items():
                category_features = importance_df[
                    importance_df['feature'].str.contains('|'.join(keywords), case=False, na=False)
                ]
                total_weight = category_features['importance'].sum()
                percentage = (total_weight / total_importance) * 100
                count = len(category_features)
                
                print(f"   {category:<15}: {percentage:6.2f}% ({count} features)")
            
            # Check for concerning patterns
            print(f"\n⚠️  MODEL VALIDATION CHECKS:")
            
            # Check if one feature dominates
            top_feature_weight = importance_df.iloc[0]['importance'] / total_importance
            if top_feature_weight > 0.5:
                print(f"   🚨 WARNING: Top feature '{importance_df.iloc[0]['feature']}' has {top_feature_weight*100:.1f}% importance!")
                print(f"      This suggests potential overfitting or data leakage.")
            else:
                print(f"   ✅ Good feature distribution (top feature: {top_feature_weight*100:.1f}%)")
            
            # Check for zero-importance features
            zero_importance = (importance_df['importance'] == 0).sum()
            if zero_importance > 0:
                print(f"   ⚠️  {zero_importance} features have zero importance (can be removed)")
            else:
                print(f"   ✅ All features contribute to the model")
            
            return importance_df
        else:
            print("❌ Feature importance not available")
            return None
    
    def analyze_training_data(self):
        """Analyze the training data quality"""
        print(f"\n📊 TRAINING DATA ANALYSIS")
        print("=" * 60)
        
        try:
            df = pd.read_parquet(self.data_path)
            print(f"✅ Training data loaded: {len(df):,} games")
            
            # Basic info
            print(f"\n📈 Dataset Overview:")
            print(f"   Shape: {df.shape}")
            print(f"   Date Range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
            
            # Check target variable
            if 'total_runs' in df.columns:
                total_runs = df['total_runs'].dropna()
                
                print(f"\n🎯 Target Variable (total_runs):")
                print(f"   Count: {len(total_runs):,} games")
                print(f"   Mean: {total_runs.mean():.2f} runs")
                print(f"   Median: {total_runs.median():.1f} runs")
                print(f"   Std Dev: {total_runs.std():.2f} runs")
                print(f"   Range: {total_runs.min()} - {total_runs.max()} runs")
                
                # Check if data looks realistic
                print(f"\n🔍 Data Quality Assessment:")
                
                realistic_mean = 7 <= total_runs.mean() <= 12
                realistic_range = total_runs.min() >= 0 and total_runs.max() <= 25
                reasonable_variance = 2 <= total_runs.std() <= 5
                
                print(f"   Realistic mean (7-12): {'✅' if realistic_mean else '❌'} ({total_runs.mean():.1f})")
                print(f"   Realistic range (0-25): {'✅' if realistic_range else '❌'} ({total_runs.min()}-{total_runs.max()})")
                print(f"   Reasonable variance (2-5): {'✅' if reasonable_variance else '❌'} ({total_runs.std():.1f})")
                
                # Distribution analysis
                print(f"\n📊 Score Distribution:")
                bins = [0, 5, 7, 9, 11, 13, 100]
                labels = ['Very Low (0-5)', 'Low (6-7)', 'Medium (8-9)', 'High (10-11)', 'Very High (12-13)', 'Extreme (14+)']
                distribution = pd.cut(total_runs, bins=bins, labels=labels, include_lowest=True).value_counts()
                
                for category, count in distribution.items():
                    percentage = (count / len(total_runs)) * 100
                    print(f"   {category}: {count:,} games ({percentage:.1f}%)")
            
            # Check key features
            print(f"\n🔑 Key Features Analysis:")
            key_features = ['home_team', 'away_team', 'temperature', 'wind_speed', 'weather_condition']
            
            for feature in key_features:
                if feature in df.columns:
                    missing_pct = (df[feature].isna().sum() / len(df)) * 100
                    unique_count = df[feature].nunique()
                    
                    print(f"   {feature}: {missing_pct:.1f}% missing, {unique_count} unique values")
                    
                    if missing_pct > 20:
                        print(f"     ⚠️  High missing data rate!")
                else:
                    print(f"   {feature}: ❌ MISSING from dataset!")
            
            # Check for recent data
            df['date'] = pd.to_datetime(df['date'])
            recent_cutoff = datetime.now() - timedelta(days=30)
            recent_data = df[df['date'] >= recent_cutoff]
            
            print(f"\n📅 Data Recency:")
            print(f"   Last 30 days: {len(recent_data):,} games ({len(recent_data)/len(df)*100:.1f}%)")
            print(f"   Most recent: {df['date'].max().strftime('%Y-%m-%d')}")
            
            days_since_last = (datetime.now() - df['date'].max()).days
            if days_since_last <= 2:
                print(f"   ✅ Data is current ({days_since_last} days old)")
            elif days_since_last <= 7:
                print(f"   ⚠️  Data is {days_since_last} days old")
            else:
                print(f"   ❌ Data is stale ({days_since_last} days old)")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
            return None
    
    def simulate_recent_predictions(self, df):
        """Simulate model predictions on recent data to check accuracy"""
        print(f"\n🎮 SIMULATED PREDICTION TESTING")
        print("=" * 60)
        
        if df is None:
            print("❌ No training data available")
            return
        
        # Get recent games for testing
        df['date'] = pd.to_datetime(df['date'])
        test_cutoff = df['date'].max() - timedelta(days=7)
        test_games = df[df['date'] > test_cutoff].copy()
        
        if len(test_games) == 0:
            print("❌ No recent games available for testing")
            return
        
        print(f"📊 Testing on {len(test_games)} recent games")
        
        # Simple prediction simulation using model features
        model = self.model_data.get('model')
        feature_columns = self.model_data.get('feature_columns', [])
        
        if model and feature_columns:
            try:
                # Create feature matrix (simplified)
                available_features = [col for col in feature_columns if col in test_games.columns]
                missing_features = [col for col in feature_columns if col not in test_games.columns]
                
                print(f"   Available features: {len(available_features)}/{len(feature_columns)}")
                if missing_features:
                    print(f"   Missing features: {len(missing_features)} (will use defaults)")
                
                # For missing features, create dummy data
                X_test = pd.DataFrame()
                for feature in feature_columns:
                    if feature in test_games.columns:
                        X_test[feature] = test_games[feature].fillna(0)
                    else:
                        # Create reasonable defaults
                        if 'temperature' in feature:
                            X_test[feature] = 70
                        elif 'wind' in feature:
                            X_test[feature] = 5
                        elif 'era' in feature:
                            X_test[feature] = 4.0
                        else:
                            X_test[feature] = 0
                
                # Make predictions
                predictions = model.predict(X_test)
                actual = test_games['total_runs'].values
                
                # Calculate accuracy
                errors = np.abs(predictions - actual)
                mae = np.mean(errors)
                
                print(f"\n🎯 SIMULATED ACCURACY RESULTS:")
                print(f"   Mean Absolute Error: {mae:.2f} runs")
                print(f"   Perfect predictions (≤0.5 error): {(errors <= 0.5).sum()}/{len(errors)} ({(errors <= 0.5).mean()*100:.1f}%)")
                print(f"   Excellent predictions (≤1.0 error): {(errors <= 1.0).sum()}/{len(errors)} ({(errors <= 1.0).mean()*100:.1f}%)")
                print(f"   Good predictions (≤1.5 error): {(errors <= 1.5).sum()}/{len(errors)} ({(errors <= 1.5).mean()*100:.1f}%)")
                
                # Show best and worst
                if len(errors) > 0:
                    best_idx = np.argmin(errors)
                    worst_idx = np.argmax(errors)
                    
                    print(f"\n🏆 Best Prediction:")
                    best_game = test_games.iloc[best_idx]
                    print(f"   {best_game.get('away_team', 'Team A')} @ {best_game.get('home_team', 'Team B')} ({best_game['date'].strftime('%Y-%m-%d')})")
                    print(f"   Predicted: {predictions[best_idx]:.1f} | Actual: {actual[best_idx]} | Error: {errors[best_idx]:.1f}")
                    
                    print(f"\n💥 Worst Prediction:")
                    worst_game = test_games.iloc[worst_idx]
                    print(f"   {worst_game.get('away_team', 'Team A')} @ {worst_game.get('home_team', 'Team B')} ({worst_game['date'].strftime('%Y-%m-%d')})")
                    print(f"   Predicted: {predictions[worst_idx]:.1f} | Actual: {actual[worst_idx]} | Error: {errors[worst_idx]:.1f}")
                
                return {
                    'predictions': predictions,
                    'actual': actual,
                    'errors': errors,
                    'mae': mae
                }
                
            except Exception as e:
                print(f"❌ Error making predictions: {e}")
                return None
        else:
            print("❌ Model or features not available")
            return None
    
    def run_complete_analysis(self):
        """Run the complete deep dive analysis"""
        print("🔬 COMPREHENSIVE MODEL DEEP DIVE")
        print("=" * 80)
        print(f"Analyzing your MLB prediction model in detail...")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Load and analyze model
        if not self.load_and_analyze_model():
            return
        
        # Analyze feature importance
        importance_df = self.analyze_feature_importance()
        
        # Analyze training data
        training_data = self.analyze_training_data()
        
        # Test predictions
        prediction_results = self.simulate_recent_predictions(training_data)
        
        # Final summary
        print(f"\n🏆 FINAL ASSESSMENT")
        print("=" * 60)
        
        print(f"✅ Model Type: {self.model_data.get('model', 'Unknown').__class__.__name__}")
        print(f"📊 Features: {len(self.model_data.get('feature_columns', []))}")
        
        if training_data is not None:
            print(f"📈 Training Data: {len(training_data):,} games")
            print(f"📅 Data Range: {training_data['date'].min()} to {training_data['date'].max()}")
        
        if prediction_results:
            mae = prediction_results['mae']
            if mae < 1.2:
                assessment = "EXCELLENT"
            elif mae < 1.8:
                assessment = "GOOD"
            elif mae < 2.5:
                assessment = "FAIR"
            else:
                assessment = "POOR"
            
            print(f"🎯 Prediction Accuracy: {assessment} (MAE: {mae:.2f})")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if importance_df is not None:
            top_feature_dominance = importance_df.iloc[0]['importance'] / importance_df['importance'].sum()
            if top_feature_dominance > 0.5:
                print(f"⚠️  Top feature dominates - check for data leakage")
            else:
                print(f"✅ Feature importance looks balanced")
        
        if training_data is not None:
            days_old = (datetime.now() - pd.to_datetime(training_data['date'].max())).days
            if days_old > 7:
                print(f"⚠️  Training data is {days_old} days old - consider updating")
            else:
                print(f"✅ Training data is current")
        
        if prediction_results and prediction_results['mae'] > 2.0:
            print(f"⚠️  Model accuracy could be improved - consider retraining")
        
        print(f"\n✅ Deep dive analysis complete!")

def main():
    analyzer = ModelDeepDive()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()
