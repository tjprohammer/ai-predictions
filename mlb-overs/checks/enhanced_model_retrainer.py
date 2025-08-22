#!/usr/bin/env python3
"""
Enhanced Learning Model Retrainer
================================

This script retrains learning models with the enhanced pitcher statistics data
for improved prediction accuracy.

Features:
- Retrains models with complete pitcher game statistics
- Compares performance before/after enhancement
- Validates feature engineering with bullpen data
- Exports retrained models for production use

Usage:
    python enhanced_model_retrainer.py [--retrain-all] [--compare-performance]
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
import os
from pathlib import Path
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Machine learning imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

class EnhancedModelRetrainer:
    """Retrain learning models with enhanced pitcher statistics"""
    
    def __init__(self, db_config: Optional[Dict] = None, verbose: bool = False):
        self.verbose = verbose
        self.db_config = db_config or {
            'host': 'localhost',
            'database': 'mlb',
            'user': 'mlbuser',
            'password': 'mlbpass'
        }
        self.conn = None
        self.models = {}
        self.performance_metrics = {}
        self.feature_importance = {}
        
    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            if self.verbose:
                print("✅ Database connection established")
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            sys.exit(1)
    
    def close_db(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "WARNING"]:
            print(f"[{timestamp}] {level}: {message}")
    
    def load_enhanced_training_data(self, start_date: str = "2025-03-20", 
                                  end_date: str = "2025-08-15") -> pd.DataFrame:
        """Load training data with enhanced pitcher statistics"""
        self.log("📊 Loading enhanced training data...")
        
        query = """
        SELECT 
            game_id,
            date,
            home_team,
            away_team,
            home_final_score,
            away_final_score,
            (home_final_score + away_final_score) as total_runs,
            
            -- Starting pitcher stats
            home_sp_ip, home_sp_er, home_sp_k, home_sp_bb, home_sp_h,
            away_sp_ip, away_sp_er, away_sp_k, away_sp_bb, away_sp_h,
            
            -- Bullpen stats  
            home_bp_ip, home_bp_er, home_bp_k, home_bp_bb, home_bp_h,
            away_bp_ip, away_bp_er, away_bp_k, away_bp_bb, away_bp_h,
            
            -- Additional features
            weather_temp, weather_wind_speed, weather_condition,
            ballpark_factor,
            home_runs_last_10, away_runs_last_10,
            home_starter, away_starter,
            
            -- Market data
            market_total, over_odds, under_odds
            
        FROM enhanced_games 
        WHERE date >= %s AND date <= %s
            AND game_status = 'Final'
            AND home_final_score IS NOT NULL
            AND away_final_score IS NOT NULL
            AND home_sp_ip IS NOT NULL AND away_sp_ip IS NOT NULL
            AND home_bp_ip IS NOT NULL AND away_bp_ip IS NOT NULL
        ORDER BY date
        """
        
        df = pd.read_sql(query, self.conn, params=(start_date, end_date))
        self.log(f"✅ Loaded {len(df)} games with complete pitcher statistics")
        
        return df
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features including pitcher and bullpen statistics"""
        self.log("🔧 Creating enhanced features...")
        
        # Create a copy to avoid modifying original
        features_df = df.copy()
        
        # Starting pitcher features
        features_df['home_sp_era'] = (features_df['home_sp_er'] * 9 / features_df['home_sp_ip']).fillna(0)
        features_df['away_sp_era'] = (features_df['away_sp_er'] * 9 / features_df['away_sp_ip']).fillna(0)
        
        features_df['home_sp_whip'] = ((features_df['home_sp_bb'] + features_df['home_sp_h']) / features_df['home_sp_ip']).fillna(0)
        features_df['away_sp_whip'] = ((features_df['away_sp_bb'] + features_df['away_sp_h']) / features_df['away_sp_ip']).fillna(0)
        
        features_df['home_sp_k_per_9'] = (features_df['home_sp_k'] * 9 / features_df['home_sp_ip']).fillna(0)
        features_df['away_sp_k_per_9'] = (features_df['away_sp_k'] * 9 / features_df['away_sp_ip']).fillna(0)
        
        # Bullpen features
        features_df['home_bp_era'] = (features_df['home_bp_er'] * 9 / features_df['home_bp_ip']).fillna(0)
        features_df['away_bp_era'] = (features_df['away_bp_er'] * 9 / features_df['away_bp_ip']).fillna(0)
        
        features_df['home_bp_whip'] = ((features_df['home_bp_bb'] + features_df['home_bp_h']) / features_df['home_bp_ip']).fillna(0)
        features_df['away_bp_whip'] = ((features_df['away_bp_bb'] + features_df['away_bp_h']) / features_df['away_bp_ip']).fillna(0)
        
        features_df['home_bp_k_per_9'] = (features_df['home_bp_k'] * 9 / features_df['home_bp_ip']).fillna(0)
        features_df['away_bp_k_per_9'] = (features_df['away_bp_k'] * 9 / features_df['away_bp_ip']).fillna(0)
        
        # Combined pitching staff metrics
        features_df['home_total_ip'] = features_df['home_sp_ip'] + features_df['home_bp_ip']
        features_df['away_total_ip'] = features_df['away_sp_ip'] + features_df['away_bp_ip']
        
        features_df['home_total_er'] = features_df['home_sp_er'] + features_df['home_bp_er']
        features_df['away_total_er'] = features_df['away_sp_er'] + features_df['away_bp_er']
        
        features_df['home_combined_era'] = (features_df['home_total_er'] * 9 / features_df['home_total_ip']).fillna(0)
        features_df['away_combined_era'] = (features_df['away_total_er'] * 9 / features_df['away_total_ip']).fillna(0)
        
        # Pitching workload features
        features_df['home_bullpen_usage'] = features_df['home_bp_ip'] / features_df['home_total_ip']
        features_df['away_bullpen_usage'] = features_df['away_bp_ip'] / features_df['away_total_ip']
        
        # Game context features
        features_df['total_innings'] = (features_df['home_total_ip'] + features_df['away_total_ip']) / 2
        features_df['pitching_quality_diff'] = features_df['away_combined_era'] - features_df['home_combined_era']
        
        # Encode categorical variables
        le_home = LabelEncoder()
        le_away = LabelEncoder()
        le_weather = LabelEncoder()
        
        features_df['home_team_encoded'] = le_home.fit_transform(features_df['home_team'].fillna('Unknown'))
        features_df['away_team_encoded'] = le_away.fit_transform(features_df['away_team'].fillna('Unknown'))
        features_df['weather_encoded'] = le_weather.fit_transform(features_df['weather_condition'].fillna('Unknown'))
        
        # Fill missing values
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_columns] = features_df[numeric_columns].fillna(features_df[numeric_columns].median())
        
        self.log(f"✅ Created {len(features_df.columns)} enhanced features")
        return features_df
    
    def select_model_features(self, df: pd.DataFrame) -> List[str]:
        """Select optimal features for model training"""
        
        # Core pitcher features - starting pitchers
        pitcher_features = [
            'home_sp_era', 'away_sp_era',
            'home_sp_whip', 'away_sp_whip', 
            'home_sp_k_per_9', 'away_sp_k_per_9',
            'home_sp_ip', 'away_sp_ip'
        ]
        
        # Bullpen features - NEW with enhanced data
        bullpen_features = [
            'home_bp_era', 'away_bp_era',
            'home_bp_whip', 'away_bp_whip',
            'home_bp_k_per_9', 'away_bp_k_per_9',
            'home_bullpen_usage', 'away_bullpen_usage'
        ]
        
        # Combined pitching metrics
        combined_features = [
            'home_combined_era', 'away_combined_era',
            'pitching_quality_diff', 'total_innings'
        ]
        
        # Game context features
        context_features = [
            'home_team_encoded', 'away_team_encoded',
            'weather_temp', 'weather_wind_speed', 'weather_encoded',
            'ballpark_factor'
        ]
        
        # Recent form features
        form_features = [
            'home_runs_last_10', 'away_runs_last_10'
        ]
        
        # Market features
        market_features = [
            'market_total', 'over_odds', 'under_odds'
        ]
        
        all_features = (pitcher_features + bullpen_features + combined_features + 
                       context_features + form_features + market_features)
        
        # Filter to only include features that exist in the dataframe
        available_features = [f for f in all_features if f in df.columns]
        
        self.log(f"✅ Selected {len(available_features)} features for modeling")
        return available_features
    
    def train_enhanced_model(self, df: pd.DataFrame, model_name: str = "enhanced_totals_predictor") -> Dict:
        """Train enhanced model with pitcher statistics"""
        self.log(f"🤖 Training enhanced model: {model_name}")
        
        # Prepare features and target
        feature_columns = self.select_model_features(df)
        X = df[feature_columns]
        y = df['total_runs']
        
        # Split data chronologically (80/20 split)
        split_index = int(len(df) * 0.8)
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest model (optimized for pitcher stats)
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)
        
        # Calculate performance metrics
        performance = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Feature importance analysis
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        feature_importance_sorted = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
        
        # Store model artifacts
        self.models[model_name] = {
            'model': model,
            'scaler': scaler,
            'features': feature_columns,
            'performance': performance,
            'feature_importance': feature_importance_sorted
        }
        
        self.log(f"✅ Model trained - Test MAE: {performance['test_mae']:.3f}, R²: {performance['test_r2']:.3f}")
        return performance
    
    def compare_with_baseline(self, df: pd.DataFrame) -> Dict:
        """Compare enhanced model with baseline (market predictions)"""
        self.log("📊 Comparing with baseline performance...")
        
        # Calculate baseline performance (using market totals)
        actual_totals = df['total_runs']
        market_totals = df['market_total'].fillna(actual_totals.mean())
        
        baseline_mae = mean_absolute_error(actual_totals, market_totals)
        baseline_rmse = np.sqrt(mean_squared_error(actual_totals, market_totals))
        baseline_r2 = r2_score(actual_totals, market_totals)
        
        # Get enhanced model performance
        enhanced_performance = self.models['enhanced_totals_predictor']['performance']
        
        comparison = {
            'baseline': {
                'mae': baseline_mae,
                'rmse': baseline_rmse,
                'r2': baseline_r2,
                'method': 'Market Totals'
            },
            'enhanced_model': {
                'mae': enhanced_performance['test_mae'],
                'rmse': enhanced_performance['test_rmse'],
                'r2': enhanced_performance['test_r2'],
                'method': 'Enhanced ML with Pitcher Stats'
            },
            'improvement': {
                'mae_reduction': baseline_mae - enhanced_performance['test_mae'],
                'mae_improvement_pct': ((baseline_mae - enhanced_performance['test_mae']) / baseline_mae) * 100,
                'rmse_reduction': baseline_rmse - enhanced_performance['test_rmse'],
                'r2_improvement': enhanced_performance['test_r2'] - baseline_r2
            }
        }
        
        self.log(f"✅ Performance comparison complete - {comparison['improvement']['mae_improvement_pct']:.1f}% MAE improvement")
        return comparison
    
    def analyze_feature_importance(self, model_name: str = "enhanced_totals_predictor") -> Dict:
        """Analyze feature importance in the enhanced model"""
        if model_name not in self.models:
            return {}
        
        importance = self.models[model_name]['feature_importance']
        
        # Categorize features
        categories = {
            'pitcher_stats': [],
            'bullpen_stats': [],
            'combined_pitching': [],
            'team_context': [],
            'game_context': [],
            'market_data': []
        }
        
        for feature, importance_score in importance.items():
            if any(x in feature for x in ['sp_', '_sp']):
                categories['pitcher_stats'].append((feature, importance_score))
            elif any(x in feature for x in ['bp_', '_bp', 'bullpen']):
                categories['bullpen_stats'].append((feature, importance_score))
            elif any(x in feature for x in ['combined', 'total_', 'pitching']):
                categories['combined_pitching'].append((feature, importance_score))
            elif any(x in feature for x in ['team', 'home_', 'away_']):
                categories['team_context'].append((feature, importance_score))
            elif any(x in feature for x in ['weather', 'ballpark']):
                categories['game_context'].append((feature, importance_score))
            elif any(x in feature for x in ['market', 'odds']):
                categories['market_data'].append((feature, importance_score))
        
        # Calculate category importance sums
        category_importance = {}
        for category, features in categories.items():
            if features:
                category_importance[category] = sum(score for _, score in features)
        
        analysis = {
            'top_features': list(importance.items())[:15],
            'category_importance': category_importance,
            'bullpen_contribution': category_importance.get('bullpen_stats', 0),
            'pitcher_vs_bullpen': {
                'starting_pitcher': category_importance.get('pitcher_stats', 0),
                'bullpen': category_importance.get('bullpen_stats', 0)
            }
        }
        
        return analysis
    
    def save_enhanced_models(self, output_dir: str = "models") -> Dict:
        """Save enhanced models and artifacts"""
        self.log("💾 Saving enhanced models...")
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for model_name, artifacts in self.models.items():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save model
            model_path = f"{output_dir}/{model_name}_{timestamp}.joblib"
            joblib.dump(artifacts['model'], model_path)
            saved_files[f"{model_name}_model"] = model_path
            
            # Save scaler
            scaler_path = f"{output_dir}/{model_name}_scaler_{timestamp}.joblib"
            joblib.dump(artifacts['scaler'], scaler_path)
            saved_files[f"{model_name}_scaler"] = scaler_path
            
            # Save metadata
            metadata = {
                'features': artifacts['features'],
                'performance': artifacts['performance'],
                'feature_importance': artifacts['feature_importance'],
                'training_date': datetime.now().isoformat(),
                'model_type': 'RandomForestRegressor',
                'enhancement_level': 'pitcher_and_bullpen_stats'
            }
            
            metadata_path = f"{output_dir}/{model_name}_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            saved_files[f"{model_name}_metadata"] = metadata_path
        
        self.log(f"✅ Saved {len(saved_files)} model artifacts")
        return saved_files
    
    def run_complete_retraining(self, start_date: str = "2025-03-20", 
                               end_date: str = "2025-08-15") -> Dict:
        """Run complete model retraining with enhanced data"""
        print("\n" + "="*80)
        print("🚀 ENHANCED MODEL RETRAINING")
        print("="*80)
        
        self.connect_db()
        
        try:
            # Load and prepare data
            df = self.load_enhanced_training_data(start_date, end_date)
            if len(df) == 0:
                raise ValueError("No training data available")
            
            df_features = self.create_enhanced_features(df)
            
            # Train enhanced model
            performance = self.train_enhanced_model(df_features)
            
            # Compare with baseline
            comparison = self.compare_with_baseline(df_features)
            
            # Analyze feature importance
            importance_analysis = self.analyze_feature_importance()
            
            # Save models
            saved_files = self.save_enhanced_models()
            
            # Compile results
            results = {
                'training_summary': {
                    'data_period': f"{start_date} to {end_date}",
                    'training_games': len(df),
                    'features_created': len(df_features.columns),
                    'model_features': len(self.models['enhanced_totals_predictor']['features'])
                },
                'performance': performance,
                'baseline_comparison': comparison,
                'feature_importance': importance_analysis,
                'saved_models': saved_files,
                'training_date': datetime.now().isoformat()
            }
            
            return results
            
        finally:
            self.close_db()
    
    def print_retraining_summary(self, results: Dict):
        """Print formatted retraining summary"""
        print("\n" + "="*80)
        print("📋 MODEL RETRAINING SUMMARY")
        print("="*80)
        
        summary = results['training_summary']
        print(f"\n📊 TRAINING DATA:")
        print(f"   Period: {summary['data_period']}")
        print(f"   Games: {summary['training_games']:,}")
        print(f"   Features: {summary['model_features']}")
        
        performance = results['performance']
        print(f"\n🤖 MODEL PERFORMANCE:")
        print(f"   Training MAE: {performance['train_mae']:.3f}")
        print(f"   Test MAE: {performance['test_mae']:.3f}")
        print(f"   Test R²: {performance['test_r2']:.3f}")
        
        comparison = results['baseline_comparison']
        print(f"\n📈 IMPROVEMENT OVER BASELINE:")
        print(f"   MAE Improvement: {comparison['improvement']['mae_improvement_pct']:.1f}%")
        print(f"   Baseline MAE: {comparison['baseline']['mae']:.3f}")
        print(f"   Enhanced MAE: {comparison['enhanced_model']['mae']:.3f}")
        
        importance = results['feature_importance']
        print(f"\n🎯 FEATURE IMPORTANCE:")
        print(f"   Bullpen Contribution: {importance['bullpen_contribution']:.3f}")
        print(f"   Starting Pitcher: {importance['pitcher_vs_bullpen']['starting_pitcher']:.3f}")
        print(f"   Top Features:")
        for i, (feature, score) in enumerate(importance['top_features'][:5]):
            print(f"     {i+1}. {feature}: {score:.3f}")
        
        print(f"\n💾 SAVED MODELS:")
        for name, path in results['saved_models'].items():
            print(f"   {name}: {path}")
        
        print("\n" + "="*80)

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='Enhanced Model Retraining')
    parser.add_argument('--start-date', default='2025-03-20', help='Start date for training data')
    parser.add_argument('--end-date', default='2025-08-15', help='End date for training data')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    # Run retraining
    retrainer = EnhancedModelRetrainer(verbose=args.verbose)
    results = retrainer.run_complete_retraining(args.start_date, args.end_date)
    
    # Print summary
    retrainer.print_retraining_summary(results)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to {args.output}")
    
    return results

if __name__ == "__main__":
    main()
